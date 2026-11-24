from dataclasses import asdict
from pathlib import Path
import os,subprocess
import numpy as np
import torch
from .config import Profile,FB,W,H,atomic_json,digest,LEGACY_STATE,FEEDBACK_STATE,RewardConfig
from .episode import run_episode
from .model import ActorCritic,model_hash
from .ppo import PPO
from .controls import roi_action

class FixtureDetector:
    """Only tests mechanics. Never accepted as a real detector by the training gate."""
    def __call__(self,yuv):
        y=np.asarray(yuv)[:W*H].reshape(H,W);yy,xx=np.where(y>160)
        if len(xx)<36:return np.empty((0,5),np.float64)
        return np.array([[xx.min(),yy.min(),xx.max()+1,yy.max()+1,.9]],np.float64)

def fixture(path,frames=32):
    rng=np.random.default_rng(81);yy,xx=np.indices((H,W));det=FixtureDetector();boxes=[]
    with Path(path).open('wb') as f:
        for k in range(frames):
            y=np.clip(80+((xx//12+yy//12)%2)*45+rng.integers(-4,5,size=(H,W)),16,235).astype(np.uint8)
            left=20+(k*4)%300;y[100:170,left:left+70]=180
            raw=np.concatenate([y.ravel(),np.full(FB-W*H,128,np.uint8)]);f.write(raw.tobytes());boxes.append(det(raw).tolist())
    return boxes

def run_smoke(toolchain,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);source=root/'fixture.yuv';boxes=fixture(source);detector=FixtureDetector();profile=Profile(frames=32)
    identity={'name':'SYNTHETIC_BRIGHTNESS_FIXTURE_NOT_YOLO'};results={}
    def case(name,fixed=None,model=None,anchor=None,state_version=LEGACY_STATE,reward_config=RewardConfig()):
        r,b=run_episode(encoder=toolchain['encoder'],decoder=toolchain['decoder'],source=source,boxes=boxes,detector=detector,detector_identity=identity,
          profile=profile,rate=120,folder=root/name,model=model,fixed=fixed,anchor=anchor,deterministic=True,split='train',state_version=state_version,reward_config=reward_config)
        results[name]=r;return r,b
    anchor,_=case('noop')
    env=os.environ.copy()
    if os.name=='nt' and Path('C:/msys64/ucrt64/bin').is_dir():env['PATH']='C:/msys64/ucrt64/bin;'+env.get('PATH','')
    env={k:v for k,v in env.items() if not k.startswith(('VVCRL_','VVENC_QPA_PROBE_'))}
    vanilla=root/'vanilla.266'
    with (root/'vanilla.log').open('wb') as f:subprocess.run(profile.command(toolchain['vanilla'],source,vanilla,120),stdout=f,stderr=subprocess.STDOUT,env=env,check=True,timeout=300)
    if digest(vanilla)!=anchor['bitstream_sha256']:raise ValueError('Online no-op differs from vanilla bitstream')
    for d in (-2,-1,1,2):
        r,_=case('fixed_'+str(d),fixed=lambda obs,d=d:np.full(12,d),anchor=anchor)
        hist=r['action_audit']['effective_delta_histogram']
        if r['bitstream_sha256']==anchor['bitstream_sha256'] or not hist.get(str(d)):raise ValueError('Native dummy action had no QP/bitstream effect: '+str(d))
    spatial=np.array([-2,0,0,2]*3)
    r,_=case('spatial',fixed=lambda obs:spatial,anchor=anchor)
    if not all(r['action_audit']['effective_delta_histogram'].get(str(d)) for d in (-2,0,2)):raise ValueError('Spatial mapping failed')
    probe=ActorCritic().eval()
    # A deliberately forced, untrained probe tests wiring, never learned quality.
    with torch.no_grad():probe.actor.weight.zero_();probe.actor.bias.zero_();probe.actor.bias[[3,8]]=10
    r,_=case('actor_forced_plus1',model=probe,anchor=anchor)
    if r['bitstream_sha256']!=results['fixed_1']['bitstream_sha256']:raise ValueError('Actor categories do not reproduce the fixed +1 native action')
    # The same 12-CTU encoder API must also reproduce a coherent two-region map.
    roi,_=case('roi_control',fixed=lambda obs:roi_action(boxes[obs['poc']],True),anchor=anchor)
    with torch.no_grad():probe.actor.bias.zero_();probe.actor.bias[[0,8]]=10
    r,_=case('actor_forced_roi',model=probe,anchor=anchor)
    if r['bitstream_sha256']!=roi['bitstream_sha256']:raise ValueError('Semantic factors do not reproduce ROI -2 / background +1')
    # C1 is observation only: the same forced action must produce the same stream.
    feedback_model=ActorCritic(FEEDBACK_STATE).eval()
    with torch.no_grad():feedback_model.actor.weight.zero_();feedback_model.actor.bias.zero_();feedback_model.actor.bias[[3,8]]=10
    r,b=case('feedback_forced_plus1',model=feedback_model,anchor=anchor,state_version=FEEDBACK_STATE)
    if r['bitstream_sha256']!=results['fixed_1']['bitstream_sha256'] or b['x'].shape[1]!=11 or not np.any(b['x'][1:,8]):raise ValueError('C1 observation/bitstream parity failed')
    if np.any(b['x'][0,8:]) or np.any(b['g'][0,15:]):raise ValueError('C1 feedback leaked into initial state')
    progressive,_=case('progressive_fixed_plus1',fixed=lambda obs:np.ones(12,int),anchor=anchor,reward_config=RewardConfig('task-budget-v3-progressive'))
    if progressive['bitstream_sha256']!=results['fixed_1']['bitstream_sha256'] or abs(progressive['raw_return']-results['fixed_1']['raw_return'])>1e-10:raise ValueError('C2 reward/bitstream identity failed')
    for name,version in (('soft_iou_fixed_plus1','task-budget-v2-soft-iou'),('combined_fixed_plus1','task-budget-v3-progressive-soft-iou')):
        soft,_=case(name,fixed=lambda obs:np.ones(12,int),anchor=anchor,reward_config=RewardConfig(version))
        if soft['bitstream_sha256']!=results['fixed_1']['bitstream_sha256'] or soft['task_retention_f1_v2']!=results['fixed_1']['task_retention_f1_v2'] or soft['reward_identity_error']>1e-10:raise ValueError('C3 soft utility changed native stream or legacy F1')
    # One synthetic real-codec mechanical PPO update with all three changes.
    # This is not useful learning or a real-data training gate.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(721);mechanical=ActorCritic(FEEDBACK_STATE)
        result,batch=run_episode(encoder=toolchain['encoder'],decoder=toolchain['decoder'],source=source,boxes=boxes,detector=detector,detector_identity=identity,
          profile=profile,rate=120,folder=root/'combined_ppo_mechanics',model=mechanical,anchor=anchor,split='train',state_version=FEEDBACK_STATE,reward_config=RewardConfig('task-budget-v3-progressive-soft-iou'))
        before=model_hash(mechanical);optimizer=PPO(mechanical);stats=optimizer.update(batch)
        if model_hash(mechanical)==before or stats['actor_head_change']<=0 or stats['critic_head_change']<=0:raise ValueError('Combined native PPO mechanics failed')
        results['combined_ppo_mechanics']=result
        atomic_json(root/'COMBINED_PPO_MECHANICS.json',{'training_ready':False,'synthetic_only':True,'stats':stats})
    summary={'status':'PASS_NATIVE_MECHANICS_ONLY','synthetic_only':True,'training_ready':False,
      'checks':{'native_noop_parity':True,'all_five_integer_bins':True,'spatial_mapping':True,'actor_to_encoder':True,'au_bit_accounting':True,'independent_decode':True,'live_state_causality':True},
      'cases':{n:{k:r[k] for k in ('bits','actual_kbps','psnr_y_10bit','bitstream_sha256','decoder_native_parity','action_audit')} for n,r in results.items()}}
    atomic_json(root/'NATIVE_SMOKE.json',summary);return summary
