from dataclasses import asdict
from pathlib import Path
import json,platform
import numpy as np
import torch
from .config import SCHEMA,TASK,PPOConfig,RewardConfig,atomic_json,canonical,digest,fingerprint,LEGACY_STATE,state_shape
from .data import DETECTOR_ID

MANDATORY=('unit_contracts','rgb_protocol','native_noop_parity','all_five_integer_bins','spatial_mapping','actor_to_encoder',
 'au_bit_accounting','independent_decode','live_state_causality','two_gpu_collectors','real_train_rollout',
 'logprob_value_replay','actor_head_updated','critic_head_updated','reward_identity','validation_no_update','checkpoint_reload')

def environment():return {'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__}
def binding(root,toolchain,profile,workers=2,reward_config=RewardConfig(),ppo_config=PPOConfig(),train_schedule='mixed',state_version=LEGACY_STATE):
    reward_config.validate();ppo_config.validate();state_shape(state_version)
    if train_schedule not in ('blocked','mixed'):raise ValueError('Unknown training schedule')
    return {'schema':SCHEMA,'task':TASK,'source_sha256':fingerprint(root),'profile':asdict(profile),'ppo':asdict(ppo_config),'train_schedule':train_schedule,'reward':asdict(reward_config),'detector':DETECTOR_ID,
      'binary_sha256':{k:digest(toolchain[k]) for k in ('encoder','vanilla','decoder')},'workers':workers,'environment':environment(),'state_version':state_version}
def write_gate(folder,bind,checks,evidence):
    folder=Path(folder)
    if set(checks)!=set(MANDATORY):raise ValueError('Preflight checklist incomplete')
    all_ok=all(v is True for v in checks.values())
    report={'schema':SCHEMA,'status':'PASS_READY_FOR_MINI' if all_ok else 'FAIL_TRAINING_BLOCKED','binding':bind,'checks':checks,
      'evidence':{str(Path(p).relative_to(folder)):digest(p) for p in evidence},'training_performed':'one mechanical PPO update on train only; no mini training'}
    atomic_json(folder/'PREFLIGHT_GATE.json',report);return report
def require_gate(folder,expected):
    folder=Path(folder);p=folder/'PREFLIGHT_GATE.json'
    if not p.exists():raise RuntimeError('Training blocked: run the full real-data preflight first')
    r=json.loads(p.read_text())
    if r.get('status')!='PASS_READY_FOR_MINI' or set(r.get('checks',{}))!=set(MANDATORY) or not all(v is True for v in r['checks'].values()):raise RuntimeError('Training blocked: incomplete/failed preflight')
    if r['binding']!=expected:raise RuntimeError('Training blocked: source/profile/optimizer/toolchain/environment differs from preflight')
    for n,h in r['evidence'].items():
        p=(folder/n).resolve()
        if not p.is_relative_to(folder.resolve()) or not p.is_file() or digest(p)!=h:raise RuntimeError('Training blocked: preflight evidence missing/corrupt')
    if not r['evidence']:raise RuntimeError('Training blocked: empty preflight evidence')
    return r
