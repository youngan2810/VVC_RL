"""D-series contracts: synthetic tests, not evidence of useful learning."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from vvcrl.config import (FEEDBACK_STATE,D1_STATE,D1_ISOLATED_STATE,D2_STATE,D2_NEUTRAL_STATE,D3_STATE,D4_STATE,ONLINE_STATES,
    D_STATES,REGION_STATES,ISOLATED_STATES,COMBINED_REWARD,ONLINE_REWARD,RewardConfig,PPOConfig,state_shape,validate_experiment,FB)
from vvcrl.model import ActorCritic,model_hash
from vvcrl.ppo import PPO,gae
from vvcrl.reward import ProgressiveBudget,reward,clip_objective,online_cap
from vvcrl.state import StateBuilder

class DSeriesTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def state(self,version):
        sc,gc=state_shape(version)
        x=torch.zeros(8,sc,3,4);g=torch.randn(8,gc)
        x[:,7]=1;x[1:7,0,0,0]=1;x[7,0]=1
        x[:,3]=torch.linspace(.1,.9,8)[:,None,None]
        return x,g
    def test_initialization_equivalence(self):
        torch.manual_seed(17);base=ActorCritic(FEEDBACK_STATE);rng=torch.get_rng_state().clone()
        x,g=self.state(FEEDBACK_STATE);reference=base.evaluate(x,g,deterministic=True)
        for v in (D1_STATE,D1_ISOLATED_STATE,D2_STATE,D3_STATE):
            torch.manual_seed(17);m=ActorCritic(v)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()),v)
            gv=torch.cat([g,torch.zeros(8,2)],1) if v==D3_STATE else g
            out=m.evaluate(x,gv,deterministic=True)
            for k in ('factor_logits','value','actions','log_prob','aux_logits'):
                torch.testing.assert_close(reference[k],out[k],atol=1e-6,rtol=1e-6)
    def test_gradient_partition_and_update(self):
        for v in D_STATES:
            torch.manual_seed(22);m=ActorCritic(v);x,g=self.state(v);out=m.evaluate(x,g)
            ap,cp=m.actor_parameters(),m.critic_parameters()
            self.assertFalse(set(map(id,ap))&set(map(id,cp)))
            for loss,params in ((out['log_prob'].mean(),cp),(out['value'].mean()+out['aux_logits'].mean(),ap)):
                grads=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
                self.assertTrue(all(a is None for a in grads),v)
            replay=m.evaluate(x,g,out['actions'])
            torch.testing.assert_close(replay['log_prob'],out['log_prob'])
            self.assertEqual(int(out['factor_actions'][0,0]),2)
            self.assertEqual(int(out['factor_actions'][7,1]),2)
            b={k:out[k].detach().numpy() for k in ('actions','log_prob','value')}
            b.update(x=x.numpy(),g=g.numpy(),reward=np.linspace(-1,1,8,dtype=np.float32),done=np.array([False]*3+[True]+[False]*3+[True]),
                aux_target=np.zeros((8,3,4),np.float32),aux_weight=np.ones((8,3,4),np.float32))
            before=model_hash(m);p=PPO(m);stats=p.update(b)
            self.assertNotEqual(before,model_hash(m));self.assertEqual(stats['independent_gradient_clipping'],v in ISOLATED_STATES)
            self.assertGreater(stats['actor_head_change'],0);self.assertGreater(stats['critic_head_change'],0)
    def test_online_terminal_conservation_and_sorted_roundtrip(self):
        cfg=RewardConfig(ONLINE_REWARD)
        a={'bits':1000,'coding_order':[3,1,0,2],'by_poc':{str(p):{'bits':b} for p,b in zip([3,1,0,2],[500,200,200,100])}}
        a=json.loads(json.dumps(a,sort_keys=True))
        for bits in ([700,100,100,100],[500,200,200,300],[100,100,100,100]):
            tracker=ProgressiveBudget(4,1100,cfg,a);used=0;rs=[]
            for k,(p,b) in enumerate(zip(a['coding_order'],bits)):
                used+=b;ctx=tracker.step(p,b,used)
                if k<3:self.assertAlmostEqual(ctx['cumulative_cap_bits'],online_cap(1100,k+1,4,cfg))
                rs.append(reward(.8,.7,b,a['by_poc'][str(p)]['bits'],4,1100,2,config=cfg,terminal=k==3,total_bits=used,anchor_total_bits=1000,progress=ctx))
            self.assertAlmostEqual(sum(r['raw'] for r in rs),clip_objective(.8,.7,used,1000,1100,2,cfg)['raw'],places=12)
        with self.assertRaises(ValueError):ProgressiveBudget(4,1100,cfg,{k:v for k,v in a.items() if k!='coding_order'})
        with self.assertRaises(ValueError):ProgressiveBudget(4,1100,cfg,dict(a,coding_order=[3,3,0,2]))
        legacy=ProgressiveBudget(4,1100,RewardConfig(COMBINED_REWARD),a)
        self.assertEqual(legacy.order,(3,1,0,2))
    def test_variant_binding(self):
        for v in D_STATES:
            cfg=RewardConfig(ONLINE_REWARD if v in ONLINE_STATES else COMBINED_REWARD)
            validate_experiment(v,cfg)
            with self.assertRaises(ValueError):validate_experiment(v,RewardConfig())
        with self.assertRaises(ValueError):validate_experiment(FEEDBACK_STATE,RewardConfig(ONLINE_REWARD))
    def test_online_state_is_causal(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'fixture.yuv';source.write_bytes(bytes(FB*2))
            cfg=RewardConfig(ONLINE_REWARD);s=StateBuilder(source,[[],[]],2,120,1.,D3_STATE,cfg)
            e=dict(schema='vvc-rl-online/1',k=0,bits_before=0,width=480,height=320,ctu=128,poc=1,qp=[30]*12,activity=[1]*12,slice_type=2,slice_qp=30,temporal_layer=0,reference=True)
            try:
                x,g=s.build(e);self.assertEqual(g.shape,(20,));self.assertEqual(g[-2],0.)
                s.advance(e,[0]*12,dict(k=0,poc=1,bits=5000,bits_total=5000,winning_qp=np.full((3,4),30)))
                e.update(k=1,poc=0,bits_before=5000)
                x,g=s.build(e)
                self.assertAlmostEqual(float(g[-2]),(5000-online_cap(s.budget,1,2,cfg))/s.budget,places=6)
            finally:s.close()

    def test_d4_preserves_actor_rng_auxiliary_and_state(self):
        torch.manual_seed(17);old=ActorCritic(D3_STATE);rng=torch.get_rng_state().clone()
        torch.manual_seed(17);new=ActorCritic(D4_STATE)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        x,g=self.state(D3_STATE);g[:,14]=.12;g[:,18]=torch.linspace(-.1,.2,len(g))
        a=old.evaluate(x,g,deterministic=True);b=new.evaluate(x,g,a['actions'])
        for key in ('factor_logits','log_prob','entropy_joint','aux_logits'):
            torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
        torch.testing.assert_close(b['value'],a['value']+120*g[:,18].clamp_min(0))
        # An auxiliary/value update still has no direct gradient into Actor.
        grad=torch.autograd.grad(b['value'].sum(),new.actor_parameters(),allow_unused=True)
        self.assertTrue(all(v is None for v in grad))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.yuv';p.write_bytes(bytes(FB*2))
            states=[StateBuilder(p,[[],[]],2,120,1.2,v) for v in ONLINE_STATES]
            e=dict(schema='vvc-rl-online/1',k=0,bits_before=0,width=480,height=320,ctu=128,poc=1,
                   qp=[30]*12,activity=[1]*12,slice_type=2,slice_qp=30,temporal_layer=0,reference=True)
            try:
                for a,b in zip(states[0].build(e),states[1].build(e)):np.testing.assert_array_equal(a,b)
            finally:
                for s in states:s.close()

    def test_d4_potential_gae_equivalence_and_terminal_boundaries(self):
        # Two complete episodes: anchor cap below target, then target below
        # anchor, with different prices and a real terminal cap correction.
        rewards=[];terminal_rewards=[];values=[];residual_values=[];done=[]
        cfg=RewardConfig(ONLINE_REWARD);m=ActorCritic(D4_STATE)
        for budget,anchor_bits,price,bits in ((1000,800,1.3,[400,40,40,40]),(700,900,.7,[500,100,100,100])):
            anchor={'bits':anchor_bits,'coding_order':[3,1,0,2],
                    'by_poc':{str(p):{'bits':anchor_bits/4} for p in [3,1,0,2]}}
            tracker=ProgressiveBudget(4,budget,cfg,anchor);used=0
            for k,(p,b) in enumerate(zip(anchor['coding_order'],bits)):
                g=torch.zeros(1,20);g[0,14]=price/10;g[0,18]=(used-online_cap(budget,k,4,cfg))/budget
                offset=float(m.value_budget_offset(g));u=.02*(k-1)
                used+=b;context=tracker.step(p,b,used)
                r=reward(.75,.7,b,anchor_bits/4,4,budget,price,config=cfg,terminal=k==3,
                         total_bits=used,anchor_total_bits=anchor_bits,progress=context)
                terminal_cost=-100*price*max(0,used-1.02*min(budget,anchor_bits))/budget if k==3 else 0.
                rewards.append(r['train']);terminal_rewards.append(100*r['task_component']+terminal_cost)
                values.append(u+offset);residual_values.append(u);done.append(k==3)
        for lam in (.95,1.):
            shaped,_=gae(rewards,values,done,1.,lam)
            terminal,_=gae(terminal_rewards,residual_values,done,1.,lam)
            np.testing.assert_allclose(shaped,terminal,atol=2e-5,rtol=2e-5)
        self.assertAlmostEqual(sum(rewards),sum(terminal_rewards),places=10)
        zero=torch.zeros(3,20);zero[:,14]=.1;zero[:,18]=torch.tensor([-.2,0.,.02])
        torch.testing.assert_close(m.value_budget_offset(zero),torch.tensor([0.,0.,2.]))

    def test_d4_scale_guard_before_collection_and_optimizer(self):
        from vvcrl.episode import run_episode
        from vvcrl.gates import binding
        from vvcrl.config import Profile
        with self.assertRaisesRegex(ValueError,'reward_scale=100'):
            PPO(ActorCritic(D4_STATE),PPOConfig(reward_scale=10.))
        with self.assertRaisesRegex(ValueError,'reward_scale=100'):
            binding('.',{},Profile(),reward_config=RewardConfig(ONLINE_REWARD),
                    ppo_config=PPOConfig(reward_scale=10.),state_version=D4_STATE)
        with self.assertRaisesRegex(ValueError,'reward_scale=100'):
            run_episode(encoder=None,decoder=None,source=None,boxes=None,detector=None,detector_identity=None,
                        profile=None,rate=50,folder=None,state_version=D4_STATE,reward_scale=10.)

    def test_d4_update_signals_replay_and_checkpoint_semantic_guard(self):
        from dataclasses import asdict
        from vvcrl.train import save_checkpoint,load_checkpoint
        from vvcrl.reward import RatePrices
        torch.manual_seed(111);m=ActorCritic(D4_STATE);x,g=self.state(D4_STATE)
        g[:,14]=.1;g[:,18]=torch.tensor([0.,.1,.08,.02,0.,.05,.01,0.])
        with torch.no_grad():out=m.evaluate(x,g)
        b={k:out[k].numpy() for k in ('actions','log_prob','value')}
        b.update(x=x.numpy(),g=g.numpy(),reward=np.linspace(-1,1,8,dtype=np.float32),
          done=np.array(([False]*3+[True])*2),aux_target=np.zeros((8,3,4),np.float32),aux_weight=np.ones((8,3,4),np.float32))
        ppo=PPO(m)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);stats=ppo.update(b,path/'signals.npz')
            self.assertGreater(stats['actor_head_change'],0);self.assertGreater(stats['critic_head_change'],0)
            self.assertLessEqual(stats['replay_value_error'],2e-5)
            with np.load(path/'signals.npz') as z:
                np.testing.assert_allclose(z['value_budget_offset'],np.maximum(g[:,18].numpy(),0)*100)
                np.testing.assert_allclose(z['returns_residual']+z['value_budget_offset'],z['returns'],atol=2e-6)
                a,_=gae(b['reward'],b['value'],b['done'],1.,1.)
                np.testing.assert_array_equal(z['advantage_monte_carlo'],a)
            bind=dict(state_version=D4_STATE,reward=asdict(RewardConfig(ONLINE_REWARD)),ppo=asdict(ppo.config),source_sha256='fixture')
            save_checkpoint(path/'latest.pt',m,ppo,RatePrices(),1,2,bind,0.,111)
            restored=ActorCritic(D4_STATE);rp=PPO(restored);load_checkpoint(path/'latest.pt',restored,rp,bind)
            self.assertEqual(model_hash(m),model_hash(restored))
            torch.testing.assert_close(m.evaluate(x,g,out['actions'])['value'],restored.evaluate(x,g,out['actions'])['value'],atol=0,rtol=0)
            old=ActorCritic(D3_STATE);before=model_hash(old)
            with self.assertRaises(ValueError):load_checkpoint(path/'latest.pt',old,PPO(old),dict(bind,state_version=D3_STATE))
            self.assertEqual(before,model_hash(old))

    def test_d4_episode_checks_value_against_actual_reward_context(self):
        # Mock native/detector execution only. Use the real state builder,
        # model, reward tracker, episode journal and offset identity check.
        from unittest.mock import patch
        from vvcrl.config import Profile,digest
        from vvcrl.episode import run_episode
        class EncoderFixture:
            def __init__(self,encoder,source,folder,profile,rate,**kw):
                self.folder=Path(folder);self.folder.mkdir();(self.folder/'recon').mkdir()
                for n in ('encoder.log','decoder.log','trace.csv','command.json'):(self.folder/n).write_text('CPU_FIXTURE_ONLY')
                self.k=self.bits=0
            def observation(self):
                return dict(schema='vvc-rl-online/1',k=self.k,bits_before=self.bits,width=480,height=320,
                    ctu=128,poc=31-self.k,qp=[30]*12,activity=[1]*12,slice_qp=30,temporal_layer=0,reference=True,slice_type=2)
            def step(self,d):
                bits=20000 if self.k==0 else 1000;self.bits+=bits
                out=dict(k=self.k,poc=31-self.k,bits=bits,bits_total=self.bits,winning_qp=np.full((3,4),30.).tolist())
                self.k+=1;return out,np.full(FB,511,dtype=np.uint16)
            def finish(self):
                p=self.folder/'stream.266';p.write_bytes(bytes(self.bits//8))
                return dict(bits=self.bits,actual_kbps=self.bits*30/32/1000,bitstream_sha256=digest(p))
            def close(self):pass
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'source.yuv';source.write_bytes(bytes([128])*FB*32)
            encoder=p/'encoder';encoder.write_bytes(b'CPU_FIXTURE_ONLY')
            m=ActorCritic(D4_STATE)
            kwargs=dict(encoder=encoder,decoder=None,source=source,boxes=[[[0,0,80,80,.9]]]*32,
                detector=lambda yuv:[],detector_identity={'fixture':True},profile=Profile(frames=32),rate=120,
                model=m,state_version=D4_STATE,reward_config=RewardConfig(ONLINE_REWARD))
            with patch('vvcrl.episode.NativeProcess',EncoderFixture),patch('vvcrl.episode.verify_decode',return_value=True),patch('vvcrl.episode.audit_trace',return_value={'fixture_only':True}):
                result,batch=run_episode(folder=p/'valid',**kwargs)
                self.assertLess(result['value_budget_identity_error'],2e-5)
                self.assertEqual(batch['done'].sum(),1)
                frames=json.loads((p/'valid/frames.json').read_text())
                self.assertGreater(max(f['value_budget_offset'] for f in frames),0)
                original=m.value_budget_offset
                with patch.object(m,'value_budget_offset',side_effect=lambda g:original(g)+.1):
                    with self.assertRaisesRegex(ValueError,'causal reward budget context'):
                        run_episode(folder=p/'invalid',**kwargs)
