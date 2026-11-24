"""Validation-only ablations; never optimize on validation or select Test videos."""
from pathlib import Path
from dataclasses import asdict
import json,torch,uuid
from .config import Profile,RewardConfig,PPOConfig,atomic_json,LEGACY_STATE
from .gates import binding,require_gate
from .model import ActorCritic,model_hash
from .train import load_checkpoint
from .ppo import PPO
from .locked.data import VideoCatalog
from .collect import collect
from .artifacts import export_run

def diagnose(root,output,gate,toolchain,input_root,tools,checkpoint,profile=Profile(),reward_config=RewardConfig(),ppo_config=PPOConfig(),train_schedule='mixed',state_version=LEGACY_STATE):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);bind=binding(root,toolchain,profile,reward_config=reward_config,ppo_config=ppo_config,train_schedule=train_schedule,state_version=state_version);require_gate(gate,bind)
    torch.set_num_threads(1);model=ActorCritic(state_version);state=load_checkpoint(checkpoint,model,PPO(model,ppo_config),bind);before=model_hash(model)
    catalog=VideoCatalog(input_root,mode='mini',seed=20261111);rows={};token=uuid.uuid4().hex[:8]
    try:
        for control in ('ppo','random','fixed_m2','fixed_p2','roi_protect','roi_redistribute'):
            results=[]
            for start in (0,2):
                specs=[dict(input=str(input_root),output=str(output),tools=str(tools),encoder=str(toolchain['encoder']),decoder=str(toolchain['decoder']),
                    snapshot=str(checkpoint),model_sha256=before,profile=asdict(profile),reward_config=asdict(reward_config),reward_scale=ppo_config.reward_scale,state_version=state_version,video=v,split='val',rate=120,price=state['prices']['values']['120'],
                    seed=20261120+i+start,device=f'cuda:{i}',episode_id=f'{token}_{control}_{i+start}',control=control) for i,v in enumerate(catalog.selected['val'][start:start+2])]
                results+=collect(specs)
            rows[control]=[{'video':r['video'],**r['result']} for r in results]
        if model_hash(model)!=before:raise ValueError('Diagnostics modified policy')
        report={'status':'VALIDATION_DIAGNOSTICS_COMPLETE','test_used':False,'checkpoint_model_sha256':before,'results':rows,
          'interpretation':'Compare PPO to its paired no-op anchor, fixed -2/+2, random, and both semantic ROI heuristics using actual bits. Nonzero action or a changed bitstream alone is not useful learning. This one-rate check does not establish BD-rate.'}
        atomic_json(output/'POLICY_CONTROLS.json',report);return report
    finally:export_run(output,output.parent/'VVC_RL_POLICY_CONTROLS_KAGGLE_SAFE.zip')
