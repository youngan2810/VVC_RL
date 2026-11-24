import unittest
import numpy as np
from vvcrl.config import RewardConfig
from vvcrl.reward import ProgressiveBudget,reward,clip_objective
from vvcrl.ppo import gae

class ProgressiveTests(unittest.TestCase):
    def test_objective_and_refunds(self):
        cfg=RewardConfig('task-budget-v3-progressive')
        anchor={'bits':1000,'by_poc':{str(p):{'bits':b} for p,b in zip([0,3,1,2],[500,200,200,100])}}
        for bits in ([700,100,100,100],[500,200,200,120],[700,200,200,100],[100,100,100,100]):
            schedule=ProgressiveBudget(4,1100,cfg,anchor);used=0;rows=[]
            for k,(p,b) in enumerate(zip([0,3,1,2],bits)):
                used+=b;ctx=schedule.step(p,b,used)
                rows.append(reward(.8,.7,b,anchor['by_poc'][str(p)]['bits'],4,1100,2,config=cfg,terminal=k==3,total_bits=used,anchor_total_bits=1000,progress=ctx))
            expected=clip_objective(.8,.7,sum(bits),1000,1100,2,RewardConfig())['raw']
            self.assertAlmostEqual(sum(r['raw'] for r in rows),expected,places=12)
            self.assertAlmostEqual(sum(r['train'] for r in rows),100*expected,places=10)
            adv,ret=gae([r['train'] for r in rows],np.zeros(4),[False,False,False,True],lam=1.)
            self.assertAlmostEqual(float(ret[0]),100*expected,places=5)
            if bits[0]==700 and sum(bits)==1000:
                self.assertLess(rows[0]['bit_component'],0);self.assertTrue(any(r['bit_component']>0 for r in rows))
                self.assertAlmostEqual(sum(r['bit_component'] for r in rows),0)
    def test_reference_without_known_total(self):
        cfg=RewardConfig('task-budget-v3-progressive')
        s=ProgressiveBudget(2,1000,cfg);rows=[];used=0
        for k,b in enumerate([700,100]):
            used+=b
            rows.append(reward(.7,.7,b,b,2,1000,1,config=cfg,terminal=k==1,total_bits=used,anchor_total_bits=used if k==1 else None,progress=s.step(k,b,used)))
        self.assertAlmostEqual(sum(r['raw'] for r in rows),0)
    def test_invalid_order_and_totals(self):
        cfg=RewardConfig('task-budget-v3-progressive');a={'bits':200,'by_poc':{'1':{'bits':100},'0':{'bits':100}}}
        s=ProgressiveBudget(2,200,cfg,a)
        with self.assertRaises(ValueError):s.step(0,100,100)
        with self.assertRaises(ValueError):s.step(1,100,101)
        with self.assertRaises(ValueError):ProgressiveBudget(2,200,cfg,dict(a,bits=201))
        with self.assertRaises(ValueError):reward(.8,.7,100,100,2,200,1,config=cfg)
    def test_bad_terminal_cap(self):
        cfg=RewardConfig('task-budget-v3-progressive')
        with self.assertRaises(ValueError):reward(.8,.7,100,100,1,100,1,config=cfg,terminal=True,total_bits=100,anchor_total_bits=100,progress={'previous_bits':0,'cumulative_bits':100,'previous_cap_bits':0,'cumulative_cap_bits':99})
