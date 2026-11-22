from dataclasses import asdict
import numpy as np
import torch
from .config import PPOConfig

def gae(rewards,values,terminated,gamma=1.,lam=.95,last_value=0.):
    r=np.asarray(rewards,np.float32);v=np.asarray(values,np.float32);d=np.asarray(terminated,bool)
    if r.shape!=v.shape or r.shape!=d.shape:raise ValueError('GAE shapes differ')
    adv=np.zeros_like(r);carry=0.
    for t in reversed(range(len(r))):
        next_v=last_value if t==len(r)-1 else v[t+1];nonterminal=1-float(d[t])
        delta=r[t]+gamma*next_v*nonterminal-v[t]
        carry=delta+gamma*lam*nonterminal*carry;adv[t]=carry
    return adv,adv+v

def head_diagnostics(out):
    """Per-head pre-update on-policy behavior; categorical order is -2..+2.

    Variation across observed states is diagnostic, not proof of causal
    state dependence. Inactive ROI/background factors are excluded.
    """
    result={}
    probabilities=out['factor_logits'].softmax(-1).detach().cpu().numpy()
    active=out['active_factors'].detach().cpu().numpy().astype(bool)
    executed=out['factor_actions'].detach().cpu().numpy()
    for i,name in enumerate(('roi','background')):
        selection=active[:,i]
        if not selection.any():
            result[name]={'active_frames':0,'mean_probabilities':None,'std_probabilities_across_states':None,
                          'argmax_fraction':None,'sampled_fraction':None,'mean_l1_probability_deviation':None}
            continue
        p=probabilities[selection,i,:]
        chosen=executed[selection,i]
        mean=p.mean(axis=0)
        result[name]={'active_frames':int(selection.sum()),
            'mean_probabilities':mean.tolist(),
            'std_probabilities_across_states':p.std(axis=0).tolist(),
            'argmax_fraction':(np.bincount(p.argmax(axis=1),minlength=5)/len(p)).tolist(),
            'sampled_fraction':(np.bincount(chosen,minlength=5)/len(chosen)).tolist(),
            'mean_l1_probability_deviation':float(np.abs(p-mean).sum(axis=1).mean())}
    return result

def policy_state_diagnostics(model,batch):
    """Read-only assessment of policy behavior on the supplied stored states."""
    with torch.no_grad():
        x=torch.as_tensor(batch['x'],dtype=torch.float32)
        g=torch.as_tensor(batch['g'],dtype=torch.float32)
        actions=torch.as_tensor(batch['actions'],dtype=torch.long)
        return head_diagnostics(model.evaluate(x,g,actions))

class PPO:
    def __init__(self,model,config=PPOConfig()):
        config.validate();self.model,self.config=model,config;self.optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    def update(self,batch):
        c=self.config;m=self.model
        if len(batch['done'])<2 or not batch['done'][-1]:raise ValueError('Update requires complete finite clips')
        if np.asarray(batch['actions']).dtype.kind not in 'iu':raise ValueError('PPO requires original categorical integer indices, never rounded actions')
        x=torch.as_tensor(batch['x'],dtype=torch.float32);g=torch.as_tensor(batch['g'],dtype=torch.float32)
        a=torch.as_tensor(batch['actions'],dtype=torch.long);old_log=torch.as_tensor(batch['log_prob'],dtype=torch.float32)
        adv,ret=gae(batch['reward'],batch['value'],batch['done'],c.gamma,c.gae_lambda)
        old_values=np.asarray(batch['value'],dtype=np.float32)
        rewards=np.asarray(batch['reward'],dtype=np.float32)
        done=np.asarray(batch['done'],dtype=bool)
        next_values=np.concatenate((old_values[1:],np.zeros(1,dtype=np.float32)))
        td_error=rewards+c.gamma*next_values*(~done)-old_values
        variance=float(np.var(ret))
        learning={'ppo_config':asdict(c),
                  'advantage_raw_mean':float(np.mean(adv)),'advantage_raw_std':float(np.std(adv)),
                  'return_mean':float(np.mean(ret)),'return_std':float(np.std(ret)),
                  'td_error_mean':float(np.mean(td_error)),'td_error_std':float(np.std(td_error)),
                  'critic_explained_variance':(float(1-np.var(ret-old_values)/variance) if variance>1e-12 else None),
                  'terminal_rate_reward_nonzero':int(np.count_nonzero(rewards[done]))}
        adv=torch.as_tensor(adv);ret=torch.as_tensor(ret)
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        target=torch.as_tensor(batch['aux_target'],dtype=torch.float32);weight=torch.as_tensor(batch['aux_weight'],dtype=torch.float32)
        with torch.no_grad():
            replay=m.evaluate(x,g,a)
            replay_error=float((replay['log_prob']-old_log).abs().max())
            value_error=float((replay['value']-torch.as_tensor(batch['value'])).abs().max())
            learning['head_behavior_before_update']=head_diagnostics(replay)
        if replay_error>2e-5 or value_error>2e-5:raise ValueError('Collector policy/value differs from optimizer before update')
        before_actor=m.actor.weight.detach().clone();before_critic=m.critic[-1].weight.detach().clone();metrics=[];stopped=False
        for epoch in range(c.epochs):
            for idx in torch.randperm(len(x)).split(c.minibatch):
                out=m.evaluate(x[idx],g[idx],a[idx]);log_ratio=out['log_prob']-old_log[idx];ratio=log_ratio.exp()
                kl=((ratio-1)-log_ratio).mean()
                if not torch.isfinite(kl):raise ValueError('Nonfinite PPO KL')
                if float(kl.detach())>c.target_kl:stopped=True;break
                policy=-torch.minimum(ratio*adv[idx],ratio.clamp(1-c.clip,1+c.clip)*adv[idx]).mean()
                value=.5*(out['value']-ret[idx]).square().mean()
                per=nn_bce(out['aux_logits'],target[idx]);aux=(per*weight[idx]).sum()/weight[idx].sum().clamp_min(1)
                entropy=out['entropy_mean'].mean()
                loss=policy+c.value_coef*value-c.entropy_coef*entropy+c.auxiliary_coef*aux
                if not torch.isfinite(loss):raise ValueError('Nonfinite PPO objective')
                policy_grads=torch.autograd.grad(policy,list(m.actor.parameters()),retain_graph=True,allow_unused=True)
                policy_grad_norm=sum(float(v.detach().square().sum()) for v in policy_grads if v is not None)**.5
                self.optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),c.max_grad);self.optimizer.step()
                metrics.append({'policy_loss':float(policy.detach()),'value_loss':float(value.detach()),'aux_loss':float(aux.detach()),'entropy_mean':float(entropy.detach()),'approx_kl_joint':float(kl.detach()),
                  'clip_fraction':float(((ratio-1).abs()>c.clip).float().mean().detach()),'grad_norm':float(grad),'policy_head_gradient_norm':policy_grad_norm})
            if stopped:break
        if not metrics:raise ValueError('PPO made no optimizer steps')
        stats={k:float(np.mean([r[k] for r in metrics])) for k in metrics[0]}
        stats.update(optimizer_steps=len(metrics),early_stop=stopped,replay_logprob_error=replay_error,replay_value_error=value_error,
          actor_head_change=float((m.actor.weight-before_actor).abs().max().detach()),critic_head_change=float((m.critic[-1].weight-before_critic).abs().max().detach()),learning_diagnostics=learning)
        return stats

def nn_bce(logits,target):return torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')
