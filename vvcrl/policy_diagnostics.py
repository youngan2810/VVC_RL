"""Read-only, deterministic validation probes. Not causal encoder experiments."""
from itertools import combinations
import numpy as np
import torch
from .config import ONLINE_STATES,uses_feedback,REGION_STATES
from .model import model_hash

HEADS=('roi','background')

def _center(a):return a-a.mean(-1,keepdim=True)

def _compare(old,new,active):
    out={}
    for i,name in enumerate(HEADS):
        mask=active[:,i]
        n=int(mask.sum())
        if not n:
            out[name]={'active_frames':0,'argmax_flip_fraction':None,'probability_l1_mean':None};continue
        a,b=old[mask,i],new[mask,i]
        out[name]={'active_frames':n,'argmax_flip_fraction':float((a.argmax(-1)!=b.argmax(-1)).float().mean()),
            'probability_l1_mean':float((a.softmax(-1)-b.softmax(-1)).abs().sum(-1).mean())}
    return out

def _decomposition(out):
    result={}
    for i,name in enumerate(HEADS):
        mask=out['active_factors'][:,i]
        if not mask.any():result[name]=None;continue
        row={}
        for key in ('base_factor_logits','residual_factor_logits','factor_logits'):
            z=_center(out[key][mask,i])
            row[key]={'centered_rms':float(z.square().mean().sqrt()),
                'across_state_rms':float((z-z.mean(0)).square().mean().sqrt())}
        z=out['factor_logits'][mask,i].topk(2,dim=-1).values
        row['margin_mean']=float((z[:,0]-z[:,1]).mean())
        row['residual_argmax_effect']=float((out['factor_logits'][mask,i].argmax(-1)!=out['base_factor_logits'][mask,i].argmax(-1)).float().mean())
        result[name]=row
    return result

@torch.no_grad()
def validation_policy_diagnostics(model,batch,rows,reward_config):
    """Feature interventions hold ROI routing fixed; off-support probes are not
    physical counterfactuals. Actual cross-rate comparison matches observed POC
    and requires both factors active, but does not isolate a rate causal effect.
    No random draws, model updates, new encoding or Test access.
    """
    before=model_hash(model);rng=torch.get_rng_state().clone()
    x_all=torch.as_tensor(batch['x'],dtype=torch.float32);g_all=torch.as_tensor(batch['g'],dtype=torch.float32)
    ends=np.flatnonzero(batch['done'])+1;starts=np.r_[0,ends[:-1]]
    if len(ends)!=len(rows) or not len(ends) or ends[-1]!=len(x_all):raise ValueError('Validation diagnostics episode boundaries differ')
    episodes=[];observed={}
    for row,start,end in zip(rows,starts,ends):
        r=row['result'];n=int(end-start);x=x_all[start:end];g=g_all[start:end]
        if r['split']!='val' or n!=r['identity']['profile']['frames']:raise ValueError('Validation diagnostic split/length differs')
        old=model.evaluate(x,g,deterministic=True);base=old['factor_logits'];active=old['active_factors']
        # Deterministic half-clip shift, avoiding a dependence on global RNG.
        perm=torch.roll(torch.arange(n),max(1,n//2))
        masks=x[:,0]>0
        probes={}
        def probe(name,xx,gg):
            new=model.evaluate(xx,gg,deterministic=True)
            if not torch.equal(active,new['active_factors']):raise ValueError('Probe changed ROI routing')
            probes[name]=_compare(base,new['factor_logits'],active)
        xx=x.clone();xx[:,1:3]=torch.where(masks[:,None],x[perm,1:3],0.)
        probe('roi_confidence_scale_shift_occupancy_fixed',xx,g)
        for name,mask in (('roi_activity_motion',masks),('bg_activity_motion',~masks)):
            xx=x.clone();xx[:,4:6]=torch.where(mask[:,None],x[perm,4:6],x[:,4:6]);probe(name,xx,g)
        xx=x.clone();xx[:,3]=x[perm,3];probe('pre_action_qp_shift',xx,g)
        if uses_feedback(model.state_version):
            xx=x.clone();gg=g.clone();xx[:,6]=x[perm,6];xx[:,8:11]=x[perm,8:11];gg[:,15:18]=g[perm,15:18]
            probe('previous_action_and_feedback_shift',xx,gg)
        # Shift cumulative bit use, recompute derived features coherently at
        # original step and target. Other historical features held fixed.
        gg=g.clone();gg[:,3]=g[perm,3];step=g[:,1]*n
        gg[:,4]=((1-gg[:,3])*n/(n-step).clamp_min(1)).clamp(-4,4)/4
        if model.state_version in ONLINE_STATES:
            tol=1+reward_config.rate_tolerance
            gg[:,18]=gg[:,3]-tol*step/n
            gg[:,19]=(tol*(step+1)-gg[:,3]*n).clamp(-4,4)/4
        probe('budget_use_shift_derived_features_recomputed',x,gg)
        # Isolated log_rate input sensitivity; not a physically consistent
        # re-encode. Actual observed pairs are separately compared below.
        for rate in (50,120,200):
            if rate!=r['rate']:
                gg=g.clone();gg[:,0]=float(np.log(rate/120));probe('log_rate_only_to_'+str(rate),x,gg)
        episodes.append({'video':row['video'],'rate':r['rate'],'frames':n,'probes':probes,'logit_decomposition':_decomposition(old)})
        order=r.get('coding_order')
        if order is not None:
            if len(order)!=n or set(order)!=set(range(n)):raise ValueError('Invalid POC order for cross-rate diagnostics')
            idx=torch.as_tensor(np.argsort(order),dtype=torch.long)
            observed[(row['video'],r['rate'])]=(base[idx],active[idx],x[idx,0]>0)
    paired=[]
    for video in sorted({v for v,_ in observed}):
        rates=sorted(r for v,r in observed if v==video)
        for a,b in combinations(rates,2):
            la,ma,ra=observed[(video,a)];lb,mb,rb=observed[(video,b)]
            same=(ra==rb).flatten(1).all(1)
            paired.append({'video':video,'rates':[a,b],'same_roi_mask_frames':int(same.sum()),
                'comparison':_compare(la,lb,ma&mb&same[:,None])})
    if model_hash(model)!=before or not torch.equal(torch.get_rng_state(),rng):raise ValueError('Read-only diagnostic changed model or RNG')
    return {'schema':'d-policy-diagnostics/1','scope':'saved validation states; perturbations are not causal native results',
        'episodes':episodes,'observed_cross_rate_pairs':paired}
