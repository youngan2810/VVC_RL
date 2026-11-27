from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .config import PPOConfig,D_STATES,ISOLATED_STATES,REGION_STATES,D4_STATE,BUDGET_VALUE_SCALE

def gae(rewards,values,terminated,gamma=1.,lam=.95,last_value=0.):
    r=np.asarray(rewards,np.float32);v=np.asarray(values,np.float32);d=np.asarray(terminated,bool)
    if r.ndim!=1 or not len(r) or r.shape!=v.shape or r.shape!=d.shape:raise ValueError('GAE shapes differ')
    if not np.isfinite(r).all() or not np.isfinite(v).all() or not np.isfinite([gamma,lam,last_value]).all():raise ValueError('Nonfinite GAE input')
    flags=np.asarray(terminated)
    if flags.dtype.kind not in 'biu' or not np.isin(flags,[0,1]).all():raise ValueError('Invalid GAE terminal flags')
    adv=np.zeros_like(r);carry=0.
    for t in reversed(range(len(r))):
        next_v=last_value if t==len(r)-1 else v[t+1];nonterminal=1-float(d[t])
        delta=r[t]+gamma*next_v*nonterminal-v[t]
        carry=delta+gamma*lam*nonterminal*carry;adv[t]=carry
    returns=adv+v
    if not np.isfinite(adv).all() or not np.isfinite(returns).all():raise ValueError('Nonfinite GAE result')
    return adv,returns

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

def diagnostic_groups(model):
    groups={'actor_spatial':model.spatial,'actor_global':model.global_net,'original_head':model.actor,
            'critic_spatial':model.critic_spatial,'critic_global':model.critic_global,
            'value_head':model.critic,'auxiliary':model.auxiliary}
    if model.state_version in REGION_STATES:
        groups.update(roi_residual=model.region_heads[0],bg_residual=model.region_heads[1])
    return {k:list(v.parameters()) for k,v in groups.items()}

def norm_of(grads):
    return sum(float(g.detach().square().sum()) for g in grads if g is not None)**.5

def residual_mechanics_ok(model,stats):
    """Check policy-loss signal and parameter change, not useful learning."""
    if model.state_version not in REGION_STATES:return True
    for name,region in (('roi_residual','roi'),('bg_residual','background')):
        if stats['learning_diagnostics']['head_behavior_before_update'][region]['active_frames']:
            if not (stats['module_diagnostics'][name]['policy_loss_grad_norm_mean']>0 and
                    stats['module_diagnostics'][name]['parameter_max_change']>0):return False
    return True

class PPO:
    def __init__(self,model,config=PPOConfig()):
        config.validate();self.model,self.config=model,config
        if model.state_version==D4_STATE and config.reward_scale!=BUDGET_VALUE_SCALE:raise ValueError('D4 analytic value requires reward_scale=100')
        self.isolated=model.state_version in ISOLATED_STATES
        if model.state_version in D_STATES:
            ap,cp=model.actor_parameters(),model.critic_parameters()
            if set(map(id,ap)) & set(map(id,cp)) or set(map(id,ap+cp))!=set(map(id,model.parameters())):
                raise ValueError('Actor/critic parameter partition is not disjoint and exhaustive')
        # Separate Adam parameter groups have independent per-parameter moments.
        # D1 retains the original joint clipping to isolate representation;
        # D1-isolated and later variants use separate clipping budgets.
        params=[{'params':ap,'name':'actor'},{'params':cp,'name':'critic'}] if self.isolated else model.parameters()
        self.optimizer=torch.optim.Adam(params,lr=config.learning_rate)
    def update(self,batch,diagnostics_path=None):
        c=self.config;m=self.model
        done_flags=np.asarray(batch['done'])
        if done_flags.ndim!=1 or done_flags.size<2 or done_flags.dtype.kind not in 'biu' or not np.isin(done_flags,[0,1]).all() or not done_flags[-1]:raise ValueError('Update requires complete finite clips')
        n=len(done_flags)
        for key in ('x','g','actions','log_prob','value','reward','aux_target','aux_weight'):
            arr=np.asarray(batch[key])
            if len(arr)!=n or not np.isfinite(arr).all():raise ValueError('Nonfinite or mismatched PPO rollout: '+key)
        for key in ('log_prob','value','reward'):
            if np.asarray(batch[key]).shape!=(n,):raise ValueError('PPO vector shape mismatch: '+key)
        if np.asarray(batch['aux_target']).shape!=(n,3,4) or np.asarray(batch['aux_weight']).shape!=(n,3,4) or (np.asarray(batch['aux_target'])<0).any() or (np.asarray(batch['aux_target'])>1).any() or (np.asarray(batch['aux_weight'])<0).any():raise ValueError('Invalid auxiliary target or weight')
        if np.asarray(batch['actions']).dtype.kind not in 'iu':raise ValueError('PPO requires original categorical integer indices, never rounded actions')
        x=torch.as_tensor(batch['x'],dtype=torch.float32);g=torch.as_tensor(batch['g'],dtype=torch.float32)
        a=torch.as_tensor(batch['actions'],dtype=torch.long);old_log=torch.as_tensor(batch['log_prob'],dtype=torch.float32)
        adv,ret=gae(batch['reward'],batch['value'],batch['done'],c.gamma,c.gae_lambda)
        raw_advantage=adv.copy();raw_returns=ret.copy()
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
                  'terminal_total_reward_nonzero':int(np.count_nonzero(rewards[done]))}
        adv=torch.as_tensor(adv);ret=torch.as_tensor(ret)
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        if not torch.isfinite(adv).all() or not torch.isfinite(ret).all() or not np.isfinite(td_error).all() or not np.isfinite(variance):raise ValueError('Nonfinite PPO return/advantage')
        target=torch.as_tensor(batch['aux_target'],dtype=torch.float32);weight=torch.as_tensor(batch['aux_weight'],dtype=torch.float32)
        with torch.no_grad():
            replay=m.evaluate(x,g,a)
            replay_error=float((replay['log_prob']-old_log).abs().max())
            value_error=float((replay['value']-torch.as_tensor(batch['value'])).abs().max())
            learning['head_behavior_before_update']=head_diagnostics(replay)
            factor_probabilities=replay['factor_logits'].softmax(-1).cpu().numpy()
            factor_actions=replay['factor_actions'].cpu().numpy()
            factor_active=replay['active_factors'].cpu().numpy()
            value_offset=replay['value_budget_offset'].cpu().numpy()
            value_residual=replay['value_residual'].cpu().numpy()
        extra_signals={}
        if m.state_version==D4_STATE:
            # Diagnostics only: never substitute MC for the selected GAE.
            mc_advantage,_=gae(rewards,old_values,done,c.gamma,1.)
            residual_returns=raw_returns-value_offset
            residual_variance=float(np.var(residual_returns))
            learning['value_decomposition']={
                'convention':'V_shaped=V_residual+scale*price*positive_pre_action_pressure',
                'budget_offset_mean':float(value_offset.mean()),'budget_offset_max':float(value_offset.max()),
                'residual_return_std':float(residual_returns.std()),
                'residual_explained_variance':float(1-np.var(residual_returns-value_residual)/residual_variance) if residual_variance>1e-12 else None,
                'gae_mc_sign_disagreement':float(np.mean(np.sign(raw_advantage)!=np.sign(mc_advantage))),
                'interpretation':'Estimator sensitivity only; MC is not a counterfactual Q-value or a learning-quality claim.'}
            extra_signals=dict(value_budget_offset=value_offset,value_residual=value_residual,
                returns_residual=residual_returns,advantage_monte_carlo=mc_advantage)
        if not np.isfinite([replay_error,value_error]).all() or replay_error>2e-5 or value_error>2e-5:raise ValueError('Collector policy/value differs from optimizer before update')
        before_actor=m.actor.weight.detach().clone();before_critic=m.critic[-1].weight.detach().clone();metrics=[];stopped=False
        groups=diagnostic_groups(m) if m.state_version in D_STATES else {}
        group_before={k:[p.detach().clone() for p in ps] for k,ps in groups.items()}
        group_metrics={k:[] for k in groups}
        policy_params=m.actor_parameters() if groups else list(m.actor.parameters())
        for epoch in range(c.epochs):
            for idx in torch.randperm(len(x)).split(c.minibatch):
                out=m.evaluate(x[idx],g[idx],a[idx]);log_ratio=out['log_prob']-old_log[idx];ratio=log_ratio.exp()
                kl=((ratio-1)-log_ratio).mean()
                if not torch.isfinite(kl):raise ValueError('Nonfinite PPO KL')
                if float(kl.detach())>c.target_kl:stopped=True;break
                policy=-torch.minimum(ratio*adv[idx],ratio.clamp(1-c.clip,1+c.clip)*adv[idx]).mean()
                if m.state_version==D4_STATE:
                    # Same value error, but make the neural target explicitly
                    # exclude the known refund. Actor still uses shaped GAE.
                    target_residual=ret[idx]-out['value_budget_offset']
                    value=.5*(out['value_residual']-target_residual).square().mean()
                else:value=.5*(out['value']-ret[idx]).square().mean()
                per=nn_bce(out['aux_logits'],target[idx]);aux=(per*weight[idx]).sum()/weight[idx].sum().clamp_min(1)
                entropy=out['entropy_mean'].mean()
                loss=policy+c.value_coef*value-c.entropy_coef*entropy+c.auxiliary_coef*aux
                if not torch.isfinite(loss):raise ValueError('Nonfinite PPO objective')
                policy_grads=torch.autograd.grad(policy,policy_params,retain_graph=True,allow_unused=True)
                pg=dict(zip(map(id,policy_params),policy_grads))
                policy_grad_norm=norm_of(pg.get(id(p)) for p in m.actor.parameters())
                # Measure entropy separately, before clipping; no optimizer mutation.
                eg={}
                if groups:
                    eg=dict(zip(map(id,policy_params),torch.autograd.grad(-c.entropy_coef*entropy,policy_params,retain_graph=True,allow_unused=True)))
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                for name,ps in groups.items():
                    group_metrics[name].append((norm_of(pg.get(id(p)) for p in ps),
                        norm_of(eg.get(id(p)) for p in ps),norm_of(p.grad for p in ps)))
                actor_grad=critic_grad=None
                if self.isolated:
                    actor_grad=float(torch.nn.utils.clip_grad_norm_(m.actor_parameters(),c.max_grad,error_if_nonfinite=True))
                    critic_grad=float(torch.nn.utils.clip_grad_norm_(m.critic_parameters(),c.max_grad,error_if_nonfinite=True))
                    grad=(actor_grad**2+critic_grad**2)**.5
                else:
                    grad=torch.nn.utils.clip_grad_norm_(m.parameters(),c.max_grad,error_if_nonfinite=True)
                self.optimizer.step()
                if not np.isfinite(policy_grad_norm) or any(not torch.isfinite(p).all() for p in m.parameters()):raise ValueError('Nonfinite optimizer result')
                metrics.append({'policy_loss':float(policy.detach()),'value_loss':float(value.detach()),'aux_loss':float(aux.detach()),'entropy_mean':float(entropy.detach()),'approx_kl_joint':float(kl.detach()),
                  'clip_fraction':float(((ratio-1).abs()>c.clip).float().mean().detach()),'grad_norm':float(grad),'policy_head_gradient_norm':policy_grad_norm})
                if self.isolated:metrics[-1].update(actor_preclip_grad_norm=actor_grad,critic_preclip_grad_norm=critic_grad)
            if stopped:break
        if not metrics:raise ValueError('PPO made no optimizer steps')
        stats={k:float(np.mean([r[k] for r in metrics])) for k in metrics[0]}
        if m.state_version in D_STATES:
            stats['independent_gradient_clipping']=self.isolated
            stats['module_diagnostics']={name:{
                'policy_loss_grad_norm_mean':float(np.mean([r[0] for r in group_metrics[name]])),
                'weighted_entropy_grad_norm_mean':float(np.mean([r[1] for r in group_metrics[name]])),
                'total_preclip_grad_norm_mean':float(np.mean([r[2] for r in group_metrics[name]])),
                'parameter_max_change':max(float((p.detach()-old).abs().max()) for p,old in zip(ps,group_before[name]))
                } for name,ps in groups.items()}
        stats.update(optimizer_steps=len(metrics),early_stop=stopped,replay_logprob_error=replay_error,replay_value_error=value_error,
          actor_head_change=float((m.actor.weight-before_actor).abs().max().detach()),critic_head_change=float((m.critic[-1].weight-before_critic).abs().max().detach()),learning_diagnostics=learning)
        if diagnostics_path is not None:
            path=Path(diagnostics_path);path.parent.mkdir(parents=True,exist_ok=True)
            # Saved in Kaggle run artifacts, never in the repository or Test split.
            np.savez_compressed(path,advantage_raw=raw_advantage,returns=raw_returns,
                td_error=td_error,reward=rewards,done=done,
                factor_probabilities=factor_probabilities,factor_actions=factor_actions,
                factor_active=factor_active,**extra_signals)
            stats['learning_diagnostics']['transition_signals_file']=path.name
        return stats

def nn_bce(logits,target):return torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')
