import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
import numpy as np
import torch
from vvcrl.config import atomic_json,digest,RewardConfig
from vvcrl.train import run_protocol,resume_progress
from vvcrl.ppo import gae,PPO
from vvcrl.model import ActorCritic,model_hash
from vvcrl.reward import RatePrices
from vvcrl.validation import validation_summary

class PretrainingAuditTests(unittest.TestCase):
    def test_resume_latest_cadence_and_pending_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);contract=run_protocol(root,7,5,None)
            atomic_json(root/'RUN_PROTOCOL.json',contract)
            latest=root/'checkpoints/latest.pt';latest.parent.mkdir();latest.write_bytes(b'checkpoint')
            state=dict(iteration=1,episodes=2,best_validation=0.)
            atomic_json(root/'updates/000001.json',dict(iteration=1))
            (root/'updates/000001_signals.npz').touch()
            atomic_json(root/'PROGRESS.json',dict(state,checkpoint='checkpoints/latest.pt',checkpoint_sha256=digest(latest),best_validation_gain=0.,validation_pending=True))
            self.assertEqual(run_protocol(root,7,5,latest),contract)
            self.assertTrue(resume_progress(root,state))
            with self.assertRaises(ValueError):run_protocol(root,7,10,latest)
            with self.assertRaises(ValueError):run_protocol(root,8,5,latest)
            with self.assertRaises(ValueError):run_protocol(root,7,5,root/'checkpoints/best_validation.pt')
            with self.assertRaises(ValueError):run_protocol(root,7,5,None)
            atomic_json(root/'updates/000002.json',dict(iteration=2))
            with self.assertRaises(ValueError):resume_progress(root,state)
            (root/'updates/000002.json').unlink();latest.write_bytes(b'changed')
            with self.assertRaises(ValueError):resume_progress(root,state)
    def test_invalid_flags_and_nonfinite_gae(self):
        for flags in ([0,2],[0.,float('nan')],[False,False]):
            if flags==[False,False]:continue
            with self.assertRaises(ValueError):gae([0,1],[0,0],flags)
        with self.assertRaises(ValueError):gae([0,float('nan')],[0,0],[False,True])
    def test_invalid_aux_and_done_before_mutation(self):
        torch.set_num_threads(1);m=ActorCritic();p=PPO(m)
        x=np.zeros((2,8,3,4),np.float32);x[:,7]=1;g=np.zeros((2,15),np.float32)
        with torch.no_grad():out=m.evaluate(torch.from_numpy(x),torch.from_numpy(g))
        batch={k:out[k].numpy() for k in ('actions','log_prob','value')}
        batch.update(x=x,g=g,reward=np.array([1.,0.],np.float32),done=np.array([False,True]),aux_target=np.zeros((2,3,4),np.float32),aux_weight=np.ones((2,3,4),np.float32))
        before=model_hash(m)
        for change in (dict(done=np.array([0,2])),dict(aux_weight=-batch['aux_weight']),dict(aux_target=batch['aux_target']+2),dict(value=np.zeros((2,1))),dict(reward=np.array([0.,np.inf]))):
            with self.assertRaises(ValueError):p.update(dict(batch,**change))
            self.assertEqual(before,model_hash(m));self.assertFalse(p.optimizer.state)
    def test_validation_nonfinite_never_selects_and_price_split(self):
        row=dict(video='v',result=dict(split='val',rate=120,task_retention_f1_v2=.8,anchor_quality=.7,bits=100,anchor_bits=100,budget=100,actual_kbps=120))
        for field,value in (('bits',float('nan')),('actual_kbps',0),('task_retention_f1_v2',1.1),('anchor_quality',float('inf'))):
            with self.assertRaises(ValueError):validation_summary([dict(row,result=dict(row['result'],**{field:value}))],[('v',120)])
        p=RatePrices();before=p.state()
        with self.assertRaises(ValueError):p.update([row['result']])
        self.assertEqual(before,p.state())
