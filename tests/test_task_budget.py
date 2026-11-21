"""Behavioral regressions for the measured V1 failure, not reward mirrors."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from vvcrl.config import RewardConfig,digest,atomic_json
from vvcrl.reward import reward,clip_objective,bitrate_cap,RatePrices
from vvcrl.validation import validation_summary
from vvcrl.model import ActorCritic,model_hash
from vvcrl.ppo import PPO
from vvcrl.train import save_checkpoint,load_checkpoint

class TaskBudgetTests(unittest.TestCase):
    def test_measured_failure_no_longer_gets_positive_reward(self):
        # Frozen check: F1 loss + bitrate savings gave positive V1 reward.
        q=.8-.02046517552606697;b0=100000.;b=b0*(1-.022500178007737404);budget=100000.
        old=clip_objective(q,.8,b,b0,budget,.989920703125,RewardConfig('paired-linear-v1'))
        new=clip_objective(q,.8,b,b0,budget,.989920703125)
        self.assertGreater(old['raw'],0);self.assertLess(new['raw'],0)
    def test_savings_do_not_improve_fixed_quality_return(self):
        vals=[clip_objective(.8,.8,b,100000,100000,1)['raw'] for b in (60000,80000,100000,102000)]
        self.assertEqual(vals,[0.,0.,0.,0.])
    def test_better_task_within_cap_is_preferred_to_worse_task_saving_bits(self):
        a=clip_objective(.82,.8,101000,100000,100000,1)
        b=clip_objective(.79,.8,80000,100000,100000,1)
        self.assertGreater(a['raw'],0);self.assertLess(b['raw'],0)
    def test_cap_restricts_both_baseline_and_target(self):
        self.assertEqual(bitrate_cap(100,80),81.6);self.assertEqual(bitrate_cap(100,120),102)
    def test_zero_tolerance_has_no_budget_slack(self):
        self.assertEqual(bitrate_cap(100,90,RewardConfig(rate_tolerance=0)),90)
    def test_terminal_penalty_once_and_not_intermediate_burst(self):
        # First AU may exceed its average budget; only complete-clip violation matters.
        bits=[800,20,20,20];anchor=[200]*4
        rows=[reward(.8,.7,b,a,4,1000,2,terminal=i==3,total_bits=sum(bits) if i==3 else None,anchor_total_bits=sum(anchor) if i==3 else None) for i,(b,a) in enumerate(zip(bits,anchor))]
        self.assertTrue(all(r['bit_component']==0 for r in rows[:-1]))
        self.assertAlmostEqual(rows[-1]['bit_component'],-2*(860-816)/1000)
        self.assertAlmostEqual(sum(r['raw'] for r in rows),clip_objective(.8,.7,860,800,1000,2)['raw'])
    def test_random_complete_episode_identity_and_coding_order(self):
        rng=np.random.default_rng(81)
        for frames in (32,64,256):
            q=rng.random(frames);a=rng.random(frames);b=rng.integers(10,1000,frames);ba=rng.integers(10,1000,frames);order=rng.permutation(frames)
            for version in ('task-budget-v2','paired-linear-v1'):
                cfg=RewardConfig(version);rs=[]
                for k,poc in enumerate(order):rs.append(reward(q[poc],a[poc],b[poc],ba[poc],frames,frames*400,1.3,config=cfg,terminal=k==frames-1,total_bits=b.sum() if k==frames-1 else None,anchor_total_bits=ba.sum() if k==frames-1 else None))
                objective=clip_objective(q.mean(),a.mean(),b.sum(),ba.sum(),frames*400,1.3,cfg)
                self.assertAlmostEqual(sum(r['raw'] for r in rs),objective['raw'])
                self.assertAlmostEqual(sum(r['train'] for r in rs),100*objective['raw'])
    def test_missing_terminal_totals_rejected(self):
        with self.assertRaises(ValueError):reward(.8,.8,100,100,32,1000,1,terminal=True)
    def test_nan_and_invalid_scores_rejected(self):
        for q in (np.nan,np.inf,-.1,1.1):
            with self.assertRaises(ValueError):reward(q,.8,100,100,32,1000,1)
        with self.assertRaises(ValueError):RewardConfig(rate_tolerance=np.nan).validate()
    def test_prices_follow_same_cap_and_never_validation(self):
        p=RatePrices();p.update([dict(rate=120,bits=100,budget=100,anchor_bits=80)])
        self.assertAlmostEqual(p.values['120'],1+.05*.184)
        before=p.state()
        with self.assertRaises(ValueError):p.update([],training=False)
        self.assertEqual(before,p.state())
    def test_invalid_dual_batch_is_atomic(self):
        p=RatePrices();before=p.state()
        with self.assertRaises(ValueError):p.update([dict(rate=50,bits=150,budget=100,anchor_bits=100),dict(rate=120,bits=float('nan'),budget=100,anchor_bits=100)])
        self.assertEqual(before,p.state())
    def test_checkpoint_cannot_resume_another_objective(self):
        with tempfile.TemporaryDirectory() as d:
            m=ActorCritic();p=Path(d)/'model.pt';bind={'reward':{'version':'paired-linear-v1'}}
            save_checkpoint(p,m,PPO(m),RatePrices(),40,80,bind,0,20261120)
            restored=ActorCritic();before=model_hash(restored)
            with self.assertRaisesRegex(ValueError,'frozen controls'):load_checkpoint(p,restored,PPO(restored),{'reward':{'version':'task-budget-v2'}})
            self.assertEqual(before,model_hash(restored))
    def test_legacy_schema_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'old.pt';torch.save({'schema':'vvc-rl-checkpoint/1'},p);atomic_json(p.with_suffix('.identity.json'),{'sha256':digest(p)})
            m=ActorCritic()
            with self.assertRaisesRegex(ValueError,'frozen controls'):load_checkpoint(p,m,PPO(m),{})

class ValidationSelectionTests(unittest.TestCase):
    def row(self,gain=.01,bits=101,video='v'):
        return dict(video=video,result=dict(split='val',rate=120,task_retention_f1_v2=.8+gain,anchor_quality=.8,bits=bits,anchor_bits=100,budget=100,actual_kbps=bits))
    def test_savings_with_quality_loss_is_not_best(self):
        r=validation_summary([self.row(-.01,80)],[('v',120)])
        self.assertFalse(r['eligible_checkpoint']);self.assertEqual(r['quality_regressions'],1)
    def test_task_gain_over_cap_is_not_best(self):
        self.assertFalse(validation_summary([self.row(.05,103)],[('v',120)])['eligible_checkpoint'])
    def test_task_gain_inside_cap_can_be_best(self):
        self.assertTrue(validation_summary([self.row()],[('v',120)])['eligible_checkpoint'])
    def test_partial_validation_cannot_select_checkpoint(self):
        self.assertFalse(validation_summary([self.row()],[('v',120),('w',120)])['eligible_checkpoint'])
    def test_duplicates_and_training_rows_rejected(self):
        with self.assertRaises(ValueError):validation_summary([self.row(),self.row()],[('v',120)])
        r=self.row();r['result']['split']='train'
        with self.assertRaises(ValueError):validation_summary([r],[('v',120)])

if __name__=='__main__':unittest.main()
