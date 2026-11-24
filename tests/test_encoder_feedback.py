import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from vvcrl.config import FB,FEEDBACK_STATE
from vvcrl.state import StateBuilder,areas
from vvcrl.feedback import winning_map,WinningTrace
from vvcrl.model import ActorCritic,model_hash
from vvcrl.ppo import PPO

class FeedbackTests(unittest.TestCase):
    def test_geometry_and_partial_rejection(self):
        rows=[dict(stage='winning_cu',poc=31,x=0,y=0,w=480,h=128,qp_after=20),
              dict(stage='winning_cu',poc=31,x=0,y=128,w=480,h=192,qp_after=40)]
        qp=winning_map(rows,31)
        np.testing.assert_array_equal(qp,np.array([[20]*4,[40]*4,[40]*4]))
        with self.assertRaises(ValueError):winning_map(rows[:1],31)
        with self.assertRaises(ValueError):winning_map(rows+rows,31)
        with self.assertRaises(ValueError):winning_map(rows,15)
    def test_incremental_trace(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.csv'
            p.write_text('stage,poc,x,y,w,h,qp_after\nwinning_cu,31,0,0,480,320,20\n')
            t=WinningTrace(p);np.testing.assert_array_equal(t.completed(31),np.full((3,4),20))
            with p.open('a') as f:f.write('winning_cu,15,0,0,480,320,32\n')
            np.testing.assert_array_equal(t.completed(15),np.full((3,4),32))
            with self.assertRaises(ValueError):t.completed(15)
    def test_causal_feedback_and_roi_mask(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.yuv';p.write_bytes(bytes([90])*FB*32)
            boxes=[[] for _ in range(32)];boxes[31]=[[0,0,128,128,.9]]
            s=StateBuilder(p,boxes,32,120,1,FEEDBACK_STATE)
            try:
                e=dict(schema='vvc-rl-online/1',k=0,bits_before=0,width=480,height=320,ctu=128,poc=31,qp=[30]*12,activity=[1]*12,slice_qp=30,temporal_layer=0,reference=True,slice_type=2)
                x,g=s.build(e);self.assertEqual(x.shape,(11,3,4));self.assertEqual(g.shape,(18,))
                self.assertFalse(x[8:].any());self.assertFalse(g[15:].any())
                qp=np.full((3,4),40.);qp[0,0]=20
                with self.assertRaises(ValueError):s.advance(e,np.zeros(12),dict(k=0,poc=31,bits=100,bits_total=100))
                s.advance(e,np.zeros(12),dict(k=0,poc=31,bits=100,bits_total=100,winning_qp=qp))
                e.update(k=1,poc=15,bits_before=100)
                x,g=s.build(e);self.assertFalse(x[0].any());self.assertEqual(x[10,0,0],1)
                np.testing.assert_allclose(x[8],qp/63);np.testing.assert_allclose(x[9],(qp-30)/63)
                self.assertAlmostEqual(float(g[15]),-20/63);self.assertEqual(g[16:].tolist(),[1,1])
            finally:s.close()
    def test_model_replay_and_ppo(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        m=ActorCritic(FEEDBACK_STATE);x=np.zeros((4,11,3,4),np.float32);x[:,7]=areas()/128**2;x[:,0,0,0]=1
        g=np.zeros((4,18),np.float32)
        out=m.evaluate(torch.from_numpy(x),torch.from_numpy(g))
        batch={k:out[k].detach().numpy() for k in ('actions','log_prob','value')}
        batch.update(x=x,g=g,reward=np.array([1.,0.,-1.,2.],np.float32),done=np.array([False,True,False,True]),aux_target=np.zeros((4,3,4),np.float32),aux_weight=np.ones((4,3,4),np.float32))
        before=model_hash(m);stats=PPO(m).update(batch);self.assertNotEqual(before,model_hash(m));self.assertLessEqual(stats['replay_logprob_error'],2e-5)
        with self.assertRaises(ValueError):ActorCritic().evaluate(torch.from_numpy(x),torch.from_numpy(g))
