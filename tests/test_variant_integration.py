import json,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from vvcrl.config import LEGACY_STATE,FEEDBACK_STATE,RewardConfig,PPOConfig,CHECKPOINT_SCHEMA,atomic_json,digest
from vvcrl.model import ActorCritic,model_hash
from vvcrl.ppo import PPO
from vvcrl.reward import RatePrices
from vvcrl.train import save_checkpoint,load_checkpoint
from vvcrl.collect import load_batches
from vvcrl.controls import roi_pair_action,roi_action

class VariantIntegrationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def batch(self,m):
        sc,gc=(11,18) if m.state_version==FEEDBACK_STATE else (8,15)
        x=np.zeros((8,sc,3,4),np.float32);g=np.zeros((8,gc),np.float32);x[:,7]=1.;x[:,0,0,0]=1
        x[:,3]=np.linspace(.1,.9,8)[:,None,None]
        if sc>8:x[:,8]=np.linspace(.1,.8,8)[:,None,None]
        with torch.no_grad():out=m.evaluate(torch.from_numpy(x),torch.from_numpy(g))
        b={k:out[k].numpy() for k in ('actions','log_prob','value')}
        b.update(x=x,g=g,reward=np.arange(8,dtype=np.float32)/4,done=np.array([False]*3+[True]+[False]*3+[True]),aux_target=np.zeros((8,3,4),np.float32),aux_weight=np.ones((8,3,4),np.float32))
        return b
    def test_shared_initialization_and_rng(self):
        torch.manual_seed(123);old=ActorCritic();rng=torch.get_rng_state()
        torch.manual_seed(123);new=ActorCritic(FEEDBACK_STATE)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,w in old.state_dict().items():
            q=new.state_dict()[k]
            if k=='spatial.0.weight':self.assertTrue(torch.equal(w,q[:,:8]));self.assertFalse(q[:,8:].count_nonzero())
            elif k=='global_net.0.weight':self.assertTrue(torch.equal(w,q[:,:15]));self.assertFalse(q[:,15:].count_nonzero())
            else:self.assertTrue(torch.equal(w,q),k)
    def test_checkpoint_resume_exact_and_mismatch_before_mutation(self):
        for version,reward_version in ((LEGACY_STATE,'task-budget-v3-progressive'),(LEGACY_STATE,'task-budget-v2-soft-iou'),(FEEDBACK_STATE,'task-budget-v3-progressive-soft-iou')):
            with self.subTest(version=version,reward=reward_version),tempfile.TemporaryDirectory() as d:
                torch.manual_seed(83);m=ActorCritic(version);p=PPO(m);p.update(self.batch(m))
                bind=dict(state_version=version,reward=asdict(RewardConfig(reward_version)),ppo=asdict(PPOConfig()),source_sha256='fixture')
                path=Path(d)/'latest.pt';prices=RatePrices()
                save_checkpoint(path,m,p,prices,1,2,bind,0.,83)
                batch=self.batch(m);p.update(batch);want=model_hash(m);rng=torch.get_rng_state()
                restored=ActorCritic(version);rp=PPO(restored);state=load_checkpoint(path,restored,rp,bind)
                self.assertEqual(state['schema'],CHECKPOINT_SCHEMA)
                again=self.batch(restored)
                for k in batch:np.testing.assert_array_equal(batch[k],again[k])
                rp.update(again);self.assertEqual(model_hash(restored),want);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                before=model_hash(restored)
                with self.assertRaises(ValueError):load_checkpoint(path,restored,rp,dict(bind,reward=asdict(RewardConfig())))
                self.assertEqual(before,model_hash(restored))
    def test_rollout_split_and_version_gate(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);m=ActorCritic(FEEDBACK_STATE);batch=self.batch(m)
            batch={k:v[:4] for k,v in batch.items()};np.savez(path/'rollout.npz',**batch)
            cfg=RewardConfig('task-budget-v3-progressive')
            proof=dict(split='val',state_version=FEEDBACK_STATE,reward_config=asdict(cfg),reward_scale=100.,model_sha256=model_hash(m),identity={'profile':{'frames':4}})
            rows=[dict(folder=str(path),result=proof)]
            with patch('vvcrl.collect.verified_result',return_value=proof):
                with self.assertRaisesRegex(ValueError,'split mismatch'):load_batches(rows,expected_split='train')
                with self.assertRaisesRegex(ValueError,'state schema'):load_batches(rows,expected_split='val',state_version=LEGACY_STATE)
                with self.assertRaisesRegex(ValueError,'reward binding'):load_batches(rows,expected_split='val',reward_config=RewardConfig())
                self.assertEqual(load_batches(rows,expected_split='val',state_version=FEEDBACK_STATE,reward_config=cfg)['x'].shape,(4,11,3,4))
    def test_exact_fixed_controls_and_legacy_empty_behavior(self):
        self.assertTrue((roi_pair_action([],0,1)==1).all())
        self.assertTrue((roi_action([],True)==0).all())
        a=roi_pair_action([[260,140,300,180,.8]],-1,1).reshape(3,4)
        self.assertEqual(a[1,2],-1);self.assertEqual(a[0,0],1)
