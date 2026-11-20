"""One real-codec PPO mechanism update on a synthetic fixture, never a training gate."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from vvcrl.config import Profile,atomic_json
from vvcrl.native_smoke import fixture,FixtureDetector
from vvcrl.episode import run_episode
from vvcrl.model import ActorCritic,model_hash
from vvcrl.ppo import PPO

def probe(toolchain,folder,frames=32):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);source=folder/'fixture.yuv';boxes=fixture(source,frames);torch.set_num_threads(1);torch.manual_seed(721)
    common=dict(encoder=toolchain['encoder'],decoder=toolchain['decoder'],source=source,boxes=boxes,detector=FixtureDetector(),
      detector_identity={'name':'SYNTHETIC_BRIGHTNESS_FIXTURE_NOT_YOLO'},profile=Profile(frames=frames),rate=120,split='train')
    anchor,_=run_episode(**common,folder=folder/'anchor',deterministic=True)
    model=ActorCritic();result,batch=run_episode(**common,folder=folder/'stochastic_actor',model=model,anchor=anchor)
    before=model_hash(model);stats=PPO(model).update(batch);after=model_hash(model)
    path=folder/'mechanical_checkpoint.pt';torch.save({'model':model.state_dict()},path)
    restored=ActorCritic();restored.load_state_dict(torch.load(path,weights_only=True)['model'])
    if before==after or stats['actor_head_change']<=0 or stats['critic_head_change']<=0 or model_hash(restored)!=after:raise ValueError('Real-codec PPO mechanical probe failed')
    report={'status':'PASS_SYNTHETIC_REAL_CODEC_SINGLE_UPDATE','training_ready':False,'learning_quality_claim':False,
      'before_model':before,'after_model':after,'checkpoint_reload_exact':True,'stats':stats,'rollout':result}
    atomic_json(folder/'NATIVE_PPO_PROBE.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--toolchain',required=True);p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=32);a=p.parse_args();print(json.dumps(probe(json.loads(Path(a.toolchain).read_text()),a.output,a.frames),indent=2))
