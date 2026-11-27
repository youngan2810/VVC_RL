import json,tempfile,unittest,zipfile,hashlib
from pathlib import Path
import numpy as np
import torch
from vvcrl.config import D2_STATE,D3_STATE,D4_STATE,ONLINE_STATES,RewardConfig,COMBINED_REWARD,ONLINE_REWARD
from vvcrl.model import ActorCritic,model_hash
from vvcrl.ppo import PPO,residual_mechanics_ok,gae,joint_diagnostics
from vvcrl.config import PPOConfig
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

    def test_full_clip_credit_keeps_delayed_reward_and_ignores_future_values(self):
        # Exact delayed-credit arithmetic, NOT a claim about real encoder Q.
        r=np.zeros(256,np.float32);r[-1]=1.;v=np.zeros_like(r)
        done=np.zeros(256,bool);done[-1]=True
        short,_=gae(r,v,done,1.,.95);full,ret=gae(r,v,done,1.,1.)
        self.assertLess(short[0],3e-6);np.testing.assert_array_equal(ret,np.ones(256,np.float32))
        np.testing.assert_array_equal(full,ret)
        v[1:]=np.linspace(-100,100,255)
        a,target=gae(r,v,done,1.,1.)
        np.testing.assert_array_equal(target,ret);self.assertEqual(a[0],full[0])
        # No return/terminal cost may leak into the preceding clip.
        _,ret=gae([0,1,0,-2],[7,8,9,10],[False,True,False,True],1.,1.,999)
        np.testing.assert_array_equal(ret,[1,1,-2,-2])
        _,ret=gae([1,2],[9,8],[False,False],.5,1.,4)
        np.testing.assert_array_equal(ret,[3,4])
        for gamma,lam in ((-1,1),(1,2)):
            with self.assertRaises(ValueError):gae([1],[0],[True],gamma,lam)
        with self.assertRaises(ValueError):gae([3e38,3e38],[0,0],[False,True],1.,1.)

    def test_joint_diagnostics_exclude_inactive_heads_and_preserve_gradient_rng(self):
        m=ActorCritic(D4_STATE);b=self.batch(m);x=torch.tensor(b['x']);g=torch.tensor(b['g'])
        x[0,0]=0;x[1,0]=1 # background-only, then ROI-only
        out=m.evaluate(x,g);rng=torch.get_rng_state().clone()
        d=joint_diagnostics(out,np.arange(len(x),dtype=np.float32))
        self.assertEqual(d['both_active_frames'],14);self.assertEqual(d['single_active_frames'],2)
        self.assertEqual(np.sum(d['sampled_counts']),14)
        self.assertAlmostEqual(np.sum(d['mean_probabilities']),1.,places=6)
        self.assertFalse(d['adaptation_proven']);json.dumps(d,allow_nan=False)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        out['log_prob'].sum().backward();self.assertGreater(m.actor.weight.grad.abs().sum(),0)
        none=joint_diagnostics(m.evaluate(x[:1],g[:1],deterministic=True))
        self.assertIsNone(none['sampled_counts'])
        with self.assertRaises(ValueError):joint_diagnostics(out,np.zeros(3))

    def test_full_return_d4_optimizer_uses_saved_target_and_joint_action(self):
        m=ActorCritic(D4_STATE);b=self.batch(m);cfg=PPOConfig(gae_lambda=1.)
        before=model_hash(m)
        with tempfile.TemporaryDirectory() as tmp:
            stats=PPO(m,cfg).update(b,Path(tmp)/'signals.npz')
            with np.load(Path(tmp)/'signals.npz',allow_pickle=False) as z:
                a,target=gae(b['reward'],b['value'],b['done'],1.,1.)
                np.testing.assert_array_equal(z['returns'],target)
                np.testing.assert_array_equal(z['advantage_raw'],a)
                np.testing.assert_allclose(z['advantage_normalized'].std(),1.,atol=1e-6)
                np.testing.assert_array_equal(z['old_value'],b['value'])
                self.assertEqual(z['advantage_gae095'].shape,(16,))
            credit=stats['learning_diagnostics']['credit_assignment']
            self.assertEqual(credit['selected'],'full_clip_return');self.assertEqual(credit['complete_clips'],2)
            self.assertEqual(stats['learning_diagnostics']['joint_behavior_before_update']['both_active_frames'],16)
            self.assertTrue(residual_mechanics_ok(m,stats));self.assertNotEqual(before,model_hash(m))

    def test_cli_binds_d4_full_credit_and_explicit_legacy_control(self):
        from unittest.mock import patch
        from vvcrl.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            tools=Path(tmp)/'tools.json';tools.write_text('{}')
            for extra,wanted in (([],1.),(['--gae-lambda','.95'],.95)):
                args=['vvcrl','preflight','--output',str(Path(tmp)/'out'),'--toolchain',str(tools),
                      '--state-version',D4_STATE,'--reward-version',ONLINE_REWARD,*extra]
                with patch('sys.argv',args),patch('vvcrl.preflight.full_preflight') as run:
                    main()
                    self.assertEqual(run.call_args.kwargs['ppo_config'].gae_lambda,wanted)
