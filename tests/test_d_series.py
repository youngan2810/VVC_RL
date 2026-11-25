"""D-series contracts: synthetic tests, not evidence of useful learning."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from vvcrl.config import (FEEDBACK_STATE,D1_STATE,D1_ISOLATED_STATE,D2_STATE,D2_NEUTRAL_STATE,D3_STATE,
    D_STATES,REGION_STATES,ISOLATED_STATES,COMBINED_REWARD,ONLINE_REWARD,RewardConfig,state_shape,validate_experiment,FB)
from vvcrl.model import ActorCritic,model_hash
from vvcrl.ppo import PPO
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
            cfg=RewardConfig(ONLINE_REWARD if v==D3_STATE else COMBINED_REWARD)
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
