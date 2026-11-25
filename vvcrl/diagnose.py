"""Validation-only ablations; never optimize on validation or select Test videos."""
from pathlib import Path
from dataclasses import asdict
import json,torch,uuid,itertools
from .config import Profile,RewardConfig,PPOConfig,atomic_json,LEGACY_STATE
from .gates import binding,require_gate
from .model import ActorCritic,model_hash
from .train import load_checkpoint
from .ppo import PPO
from .locked.data import VideoCatalog
from .collect import collect
from .artifacts import export_run
from .validation import validation_summary

def diagnose(root,output,gate,toolchain,input_root,tools,checkpoint,profile=Profile(),reward_config=RewardConfig(),ppo_config=PPOConfig(),train_schedule='mixed',state_version=LEGACY_STATE,mode='legacy'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);bind=binding(root,toolchain,profile,reward_config=reward_config,ppo_config=ppo_config,train_schedule=train_schedule,state_version=state_version);require_gate(gate,bind)
    torch.set_num_threads(1);model=ActorCritic(state_version);state=load_checkpoint(checkpoint,model,PPO(model,ppo_config),bind);before=model_hash(model)
    catalog=VideoCatalog(input_root,mode='mini',seed=20261111);rows={};token=uuid.uuid4().hex[:8]
    if mode not in ('legacy','compact'):raise ValueError('Unknown diagnostic mode')
    rates=(50,120,200) if mode=='compact' else (120,)
    controls=('ppo','roi_0_bg_p1','roi_m1_bg_p1','roi_m2_bg_p1') if mode=='compact' else ('ppo','random','fixed_m2','fixed_p2','roi_protect','roi_redistribute','roi_0_bg_p1','roi_m1_bg_p1','roi_m2_bg_p1','roi_0_bg_m1')
    items=list(itertools.product(catalog.selected['val'],rates));summaries={}
    try:
        for control in controls:
            results=[]
            for start in range(0,len(items),2):
                specs=[dict(input=str(input_root),output=str(output),tools=str(tools),encoder=str(toolchain['encoder']),decoder=str(toolchain['decoder']),
                    snapshot=str(checkpoint),model_sha256=before,profile=asdict(profile),reward_config=asdict(reward_config),reward_scale=ppo_config.reward_scale,state_version=state_version,video=v,split='val',rate=rate,price=state['prices']['values'][str(rate)],
                    seed=20261120+i+start,device=f'cuda:{i}',episode_id=f'{token}_{control}_{i+start}',control=control) for i,(v,rate) in enumerate(items[start:start+2])]
                results+=collect(specs)
            rows[control]=[{'video':r['video'],**r['result']} for r in results]
            summaries[control]=validation_summary(results,items,reward_config)
            # Save each complete control before proceeding. Never present a
            # partial grid as a complete evaluation.
            atomic_json(output/'POLICY_CONTROLS_PARTIAL.json',{'status':'PARTIAL','mode':mode,'rates':rates,'completed_controls':list(rows),'summaries':summaries})
        if model_hash(model)!=before:raise ValueError('Diagnostics modified policy')
        report={'status':'VALIDATION_DIAGNOSTICS_COMPLETE','test_used':False,'checkpoint_model_sha256':before,'results':rows,'mode':mode,'rates':rates,'summaries':summaries,
          'development_rate_conditioned_fixed':rate_conditioned_fixed(rows,rates),
          'interpretation':'Compare PPO to its paired no-op anchor, fixed -2/+2, random, historical semantic heuristics and the four native Validation controls using actual bits. Nonzero action or a changed bitstream alone is not useful learning. Rate-specific winners are development-set descriptions, not independent Test performance or BD-rate.'}
        atomic_json(output/'POLICY_CONTROLS.json',report);return report
    finally:export_run(output,output.parent/'VVC_RL_POLICY_CONTROLS_KAGGLE_SAFE.zip')


def rate_conditioned_fixed(rows,rates):
    """Descriptive best feasible fixed control per rate on development only.
    This selection must be frozen before independent test, never used as a
    per-clip deployable oracle. No-op is included when its own cap permits it.
    """
    result={}
    fixed=[k for k in rows if k.startswith('roi_') and '_bg_' in k]
    for rate in rates:
        candidates=[]
        for name in fixed:
            rr=[r for r in rows[name] if r['rate']==rate]
            if rr and all(r['clip_objective']['within_cap'] for r in rr):
                candidates.append((sum(r['task_retention_f1_v2']-r['anchor_quality'] for r in rr)/len(rr),name))
        rr=[r for r in rows.get('ppo',[]) if r['rate']==rate]
        if rr and all(r['anchor_bits']<=r['clip_objective']['budget_cap_bits'] for r in rr):candidates.append((0.,'noop'))
        result[str(rate)]=dict(gain=None,control=None) if not candidates else dict(zip(('gain','control'),max(candidates,key=lambda t:(t[0],t[1]))))
    return result
