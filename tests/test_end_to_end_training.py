"""CPU orchestration regression. No fake fixture can open the real-data gate."""
import json,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from vvcrl import train
from vvcrl.artifacts import import_run
from vvcrl.config import Profile,PPOConfig,RewardConfig,atomic_json,digest,state_shape,LEGACY_STATE,FEEDBACK_STATE
from vvcrl.episode import verified_result
from vvcrl.collect import load_batches
from vvcrl.model import ActorCritic,model_hash
from vvcrl.reward import clip_objective

class MockCatalog:
    selected={'train':['train_a','train_b','train_c','train_d'],'val':['val_a','val_b']}
    def __init__(self,*a,**kw):pass
    def check_available(self,*a,**kw):pass

def fixture_collect(specs):
    """Mock only native/detector collection. Actual policy, seals, PPO and I/O."""
    rows=[]
    for spec in specs:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec['seed'])
            model=ActorCritic(spec['state_version'])
            model.load_state_dict(torch.load(spec['snapshot'],weights_only=True)['model'])
            assert model_hash(model)==spec['model_sha256']
            n=spec['profile']['frames'];sc,gc=state_shape(spec['state_version'])
            x=np.zeros((n,sc,3,4),np.float32);x[:,7]=1;x[:,0,0,0]=1
            x[:,3]=np.linspace(.1,.9,n)[:,None,None]
            g=np.zeros((n,gc),np.float32);g[:,0]=np.log(spec['rate']/120)
            deterministic=spec['split']=='val'
            with torch.no_grad():out=model.evaluate(torch.from_numpy(x),torch.from_numpy(g),deterministic=deterministic)
            batch={k:out[k].numpy() for k in ('actions','log_prob','value')}
        reward=np.linspace(-.1,.2,n,dtype=np.float32)
        batch.update(x=x,g=g,reward=reward,done=np.array([False]*(n-1)+[True]),aux_target=np.ones((n,3,4),np.float32),aux_weight=np.ones((n,3,4),np.float32))
        folder=Path(spec['output'])/'episodes'/spec['episode_id'];folder.mkdir(parents=True)
        for f in ('frames.json','trace.csv','command.json','encoder.log','decoder.log'):(folder/f).write_text('CPU_ORCHESTRATION_FIXTURE')
        (folder/'stream.266').write_bytes(bytes(100));np.savez(folder/'rollout.npz',**batch)
        cfg=RewardConfig(**spec['reward_config'])
        result=dict(split=spec['split'],rate=spec['rate'],bits=800,budget=1000,anchor_bits=800,actual_kbps=spec['rate'],
          task_retention_f1_v2=.81,anchor_quality=.8,task_utility=.81,anchor_task_utility=.8,utility_version=cfg.utility,
          matched_iou_f1_v1=.81,anchor_matched_iou_f1_v1=.8,
          clip_objective=clip_objective(.81,.8,800,800,1000,spec['price'],cfg),raw_return=.01,mean_action_entropy=1.,
          requested_action_distribution={'0':1.},action_audit=dict(effective_delta_histogram={},nonzero_requests_zeroed=0,qg_mapping_verified=True,winning_coverage_verified=True),
          identity={'profile':spec['profile']},state_version=spec['state_version'],reward_config=spec['reward_config'],reward_scale=spec['reward_scale'],
          model_sha256=spec['model_sha256'],deterministic=deterministic,bitstream_sha256=digest(folder/'stream.266'),decoder_native_parity=True)
        atomic_json(folder/'result.json',result)
        atomic_json(folder/'SEAL.json',{p.name:digest(p) for p in folder.iterdir()})
        rows.append(dict(folder=str(folder),result=result,video=spec['video']))
    return rows

class EndToEndTests(unittest.TestCase):
    def test_incomplete_seal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);atomic_json(p/'result.json',{});atomic_json(p/'SEAL.json',{})
            with self.assertRaisesRegex(ValueError,'mandatory evidence'):verified_result(p)
    def test_duplicate_rollout_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate rollout'):load_batches([{'folder':'same'},{'folder':'same'}])
    def test_training_export_import_resume_exact(self):
        torch.set_num_threads(1)
        variants=[(LEGACY_STATE,'task-budget-v2'),(FEEDBACK_STATE,'task-budget-v2'),(LEGACY_STATE,'task-budget-v3-progressive'),(LEGACY_STATE,'task-budget-v2-soft-iou'),(FEEDBACK_STATE,'task-budget-v3-progressive-soft-iou')]
        for version,reward_version in variants:
            with self.subTest(version=version,reward=reward_version),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg=RewardConfig(reward_version);bind={'state_version':version,'reward':asdict(cfg),'ppo':asdict(PPOConfig())}
                def run(out,updates,resume=None):
                    return train.mini_train(root,out,root/'gate',{},root,root,updates=updates,val_every=1,profile=Profile(frames=32),seed=20261120,authorized=True,resume=resume,reward_config=cfg,state_version=version)
                with patch('vvcrl.train.binding',return_value=bind),patch('vvcrl.train.require_gate'),patch('vvcrl.train.VideoCatalog',MockCatalog),patch('vvcrl.train.collect',side_effect=fixture_collect):
                    a=root/'uninterrupted';self.assertEqual(run(a,2)['iteration'],2)
                    b=root/'first';self.assertEqual(run(b,1)['iteration'],1)
                    restored=root/'restored';import_run(root/'VVC_RL_MINI_first_SAFE.zip',restored)
                    self.assertEqual(run(restored,1,restored/'checkpoints/latest.pt')['iteration'],2)
                ca=torch.load(a/'checkpoints/latest.pt',weights_only=True);cb=torch.load(restored/'checkpoints/latest.pt',weights_only=True)
                for k in ('model_sha256','prices','iteration','episodes','best_validation','seed'):self.assertEqual(ca[k],cb[k])
                self.assertTrue(torch.equal(ca['torch_rng'],cb['torch_rng']))
                for key in ca['optimizer']['state']:
                    for k,v in ca['optimizer']['state'][key].items():
                        if torch.is_tensor(v):self.assertTrue(torch.equal(v,cb['optimizer']['state'][key][k]))
                        else:self.assertEqual(v,cb['optimizer']['state'][key][k])
                for step in ('000001','000002'):
                    self.assertEqual(json.loads((a/'updates'/f'{step}.json').read_text()),json.loads((restored/'updates'/f'{step}.json').read_text()))
                    self.assertEqual(json.loads((a/'validations'/f'{step}.json').read_text()),json.loads((restored/'validations'/f'{step}.json').read_text()))
                self.assertFalse(json.loads((restored/'PROGRESS.json').read_text())['validation_pending'])
