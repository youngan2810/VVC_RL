"""Fresh on-policy episodes in independent detector/native processes."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,multiprocessing as mp,os,time
import numpy as np
import torch
from .config import Profile,RewardConfig,atomic_json,canonical,digest,LEGACY_STATE,UTILITY_SCHEMA
from .model import ActorCritic,model_hash
from .data import DETECTOR_ID,open_detector,prepared_clip
from .episode import run_episode,verified_result
from .controls import roi_action

def anchor_episode(spec,detector,source,boxes,profile):
    key=hashlib.sha256(canonical({'pixel':digest(source),'teacher_sha256':hashlib.sha256(canonical(boxes).encode()).hexdigest(),'frames':profile.frames,'rate':spec['rate'],'encoder':digest(spec['encoder']),'detector':DETECTOR_ID,'profile':asdict(profile),'task':'task_retention_f1_v2','utility_schema':UTILITY_SCHEMA,'reward':spec.get('reward_config',{}),'reward_scale':spec.get('reward_scale',100.)}).encode()).hexdigest()
    base=Path(spec['output'])/'anchors'/key;base.mkdir(parents=True,exist_ok=True)
    for p in sorted(base.glob('attempt*')):
        r=verified_result(p)
        if r is not None:return r
    folder=base/f'attempt{len(list(base.glob("attempt*"))):03}'
    r,_=run_episode(encoder=spec['encoder'],decoder=spec['decoder'],source=source,boxes=boxes,detector=detector,detector_identity=DETECTOR_ID,
      profile=profile,rate=spec['rate'],folder=folder,split=spec['split'],deterministic=True,reward_config=RewardConfig(**spec.get('reward_config',{})),reward_scale=spec.get('reward_scale',100.))
    return r

def _worker(spec):
    torch.set_num_threads(1);torch.manual_seed(spec['seed']);np.random.seed(spec['seed']%(2**32))
    version=spec.get('state_version',LEGACY_STATE)
    checkpoint=torch.load(spec['snapshot'],map_location='cpu',weights_only=True)
    if checkpoint.get('state_version',checkpoint.get('binding',{}).get('state_version',LEGACY_STATE))!=version:raise ValueError('Collector state schema mismatch')
    model=ActorCritic(version);model.load_state_dict(checkpoint['model']);model.eval()
    if model_hash(model)!=spec['model_sha256']:raise ValueError('Collector checkpoint differs from parent')
    profile=Profile(**spec['profile']);detector=open_detector(spec['tools'],spec['device'])
    try:
        cache=Path(spec['output'])/'source_cache'/spec['device'].replace(':','_')
        source,boxes,key=prepared_clip(spec['input'],cache,spec['video'],profile.frames,detector,spec['split'])
        anchor=anchor_episode(spec,detector,source,boxes,profile)
        folder=Path(spec['output'])/'episodes'/spec['episode_id']
        control=spec.get('control','ppo');fixed=None;policy=model
        if control=='random':
            rng=np.random.default_rng(spec['seed']);fixed=lambda obs:rng.integers(-2,3,12);policy=None
        elif control=='fixed_m2':fixed=lambda obs:np.full(12,-2);policy=None
        elif control=='fixed_p2':fixed=lambda obs:np.full(12,2);policy=None
        elif control=='noop':fixed=lambda obs:np.zeros(12,int);policy=None
        elif control in ('roi_protect','roi_redistribute'):
            fixed=lambda obs:roi_action(boxes[obs['poc']],control=='roi_redistribute');policy=None
        elif control!='ppo':raise ValueError('Unknown diagnostic control')
        result,batch=run_episode(encoder=spec['encoder'],decoder=spec['decoder'],source=source,boxes=boxes,detector=detector,detector_identity=DETECTOR_ID,
          profile=profile,rate=spec['rate'],folder=folder,model=policy,fixed=fixed,anchor=anchor,price=spec['price'],deterministic=spec['split']!='train',split=spec['split'],reward_config=RewardConfig(**spec.get('reward_config',{})),state_version=version,reward_scale=spec.get('reward_scale',100.))
        cache_meta=json.loads((cache/key/'cache.json').read_text())
        atomic_json(folder/'collector.json',{'video':spec['video'],'split':spec['split'],'device':spec['device'],'model_sha256':model_hash(model),'input_cache_key':key,'seed':spec['seed'],
          'control':control,'source_stage_timing':cache_meta.get('timing')})
        return {'folder':str(folder),'result':result,'pid':os.getpid(),'device':spec['device'],'model_sha256':model_hash(model),'video':spec['video']}
    finally:detector.close()

def collect(specs):
    if not specs:raise ValueError('Empty collection')
    if len({(s['video'],s['rate'],s['split']) for s in specs})!=len(specs):raise ValueError('Concurrent duplicate anchor key')
    if len({s['device'] for s in specs})!=len(specs):raise ValueError('One episode per device per batch is required')
    ctx=mp.get_context('spawn')
    with ctx.Pool(len(specs)) as pool:results=pool.map(_worker,specs)
    if len({x['pid'] for x in results})!=len(specs):raise ValueError('Collectors were not independent processes')
    return results

def load_batches(results):
    records=[]
    for r in results:
        proof=verified_result(r['folder'])
        if proof is None or proof!=r['result']:raise ValueError('Rollout evidence does not match collector')
        with np.load(Path(r['folder'])/'rollout.npz',allow_pickle=False) as z:records.append({k:z[k].copy() for k in z.files})
    return {k:np.concatenate([r[k] for r in records]) for k in records[0]}
