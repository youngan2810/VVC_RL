import json,tempfile,unittest,zipfile,hashlib
from pathlib import Path
import numpy as np
import torch
from vvcrl.config import D2_STATE,D3_STATE,D4_STATE,ONLINE_STATES,RewardConfig,COMBINED_REWARD,ONLINE_REWARD
from vvcrl.model import ActorCritic,model_hash
from vvcrl.ppo import PPO,residual_mechanics_ok
from vvcrl.policy_diagnostics import validation_policy_diagnostics
from vvcrl.artifacts import export_run,import_run
from vvcrl.diagnose import rate_conditioned_fixed

class DiagnosticsTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(43)
    def batch(self,m):
        n=16;x=torch.rand(n,11,3,4);x[:,0]=0;x[:,0,0,0]=.5;x[:,7]=1
        g=torch.zeros(n,20 if m.state_version in ONLINE_STATES else 18);g[:,1]=torch.arange(n)%8/8;g[:,3]=g[:,1]*.8
        with torch.no_grad():o=m.evaluate(x,g)
        b={k:o[k].numpy() for k in ('actions','log_prob','value')}
        b.update(x=x.numpy(),g=g.numpy(),reward=np.linspace(-1,1,n,dtype=np.float32),done=np.array(([False]*7+[True])*2),aux_target=np.zeros((n,3,4),np.float32),aux_weight=np.ones((n,3,4),np.float32))
        return b
    def test_all_module_diagnostics(self):
        m=ActorCritic(D2_STATE);stats=PPO(m).update(self.batch(m));d=stats['module_diagnostics']
        self.assertEqual(set(d),{'actor_spatial','actor_global','original_head','critic_spatial','critic_global','value_head','auxiliary','roi_residual','bg_residual'})
        self.assertTrue(residual_mechanics_ok(m,stats))
        for key in ('critic_spatial','critic_global','value_head','auxiliary'):
            self.assertEqual(d[key]['policy_loss_grad_norm_mean'],0.)
            self.assertEqual(d[key]['weighted_entropy_grad_norm_mean'],0.)
        for key in ('roi_residual','bg_residual'):
            self.assertGreater(d[key]['parameter_max_change'],0.)
    def test_validation_probes_no_mutation(self):
        m=ActorCritic(D3_STATE);b=self.batch(m)
        rows=[dict(video='v',result=dict(split='val',rate=r,identity={'profile':{'frames':8}},coding_order=o)) for r,o in ((50,list(range(8))),(200,list(reversed(range(8)))))]
        before=model_hash(m);rng=torch.get_rng_state().clone()
        result=validation_policy_diagnostics(m,b,rows,RewardConfig(ONLINE_REWARD))
        self.assertEqual(before,model_hash(m));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(len(result['episodes']),2);self.assertEqual(len(result['observed_cross_rate_pairs']),1)
        self.assertEqual(result['observed_cross_rate_pairs'][0]['same_roi_mask_frames'],8)
        self.assertEqual(result['episodes'][0]['logit_decomposition']['roi']['residual_factor_logits']['centered_rms'],0.)
        json.dumps(result,allow_nan=False)
    def test_streaming_import_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'run';src.mkdir();(src/'a').write_bytes(b'a'*2200000)
            pkg=export_run(src,root/'safe.zip')
            import_run(pkg,root/'restored');self.assertEqual((root/'restored/a').read_bytes(),(src/'a').read_bytes())
            corrupt=root/'bad.bin'
            with zipfile.ZipFile(corrupt,'w') as z:
                z.writestr('a',b'bad');z.writestr('EXPORT_MANIFEST.json',json.dumps({'schema':'vvc-rl-safe/1','files':{'a':'0'*64}}))
            with self.assertRaises(ValueError):import_run(corrupt,root/'failed')
            self.assertFalse((root/'failed').exists())

    def test_d4_diagnostics_do_not_change_policy_or_rng(self):
        m=ActorCritic(D4_STATE);b=self.batch(m);before=model_hash(m);rng=torch.get_rng_state().clone()
        rows=[dict(video='v',result=dict(split='val',rate=r,identity={'profile':{'frames':8}},coding_order=list(range(8)))) for r in (50,200)]
        result=validation_policy_diagnostics(m,b,rows,RewardConfig(ONLINE_REWARD))
        self.assertEqual(before,model_hash(m));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(len(result['episodes']),2);json.dumps(result,allow_nan=False)
    def test_rate_fixed_selection(self):
        def row(gain,ok):
            return dict(rate=50,task_retention_f1_v2=.6+gain,anchor_quality=.6,anchor_bits=100,clip_objective={'within_cap':ok,'budget_cap_bits':102})
        got=rate_conditioned_fixed({'ppo':[row(.2,True)],'roi_m1_bg_p1':[row(.1,True)],'roi_m2_bg_p1':[row(.3,False)]},[50,200])
        self.assertEqual(got['50']['control'],'roi_m1_bg_p1');self.assertIsNone(got['200']['control'])
