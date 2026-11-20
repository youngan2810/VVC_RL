import io,json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
import torch
from vvcrl.config import Profile,PPOConfig,FB,canonical,digest
from vvcrl.state import StateBuilder,areas,pool
from vvcrl.model import ActorCritic,model_hash
from vvcrl.ppo import PPO,gae
from vvcrl.metrics import task
from vvcrl.reward import reward,RatePrices
from vvcrl.artifacts import export_run,import_run
from vvcrl.gates import MANDATORY,write_gate,require_gate
from vvcrl.train import save_checkpoint,load_checkpoint
from vvcrl.locked.data import ASSETS,DATASET_SHA
from vvcrl.controls import roi_action

torch.set_num_threads(1)

def batch(model,n=16):
    rng=np.random.default_rng(42);x=rng.normal(size=(n,8,3,4)).astype('float32');x[:,7]=1
    g=rng.normal(size=(n,15)).astype('float32')
    with torch.no_grad():o=model.evaluate(torch.tensor(x),torch.tensor(g))
    return dict(x=x,g=g,actions=o['actions'].numpy(),log_prob=o['log_prob'].numpy(),value=o['value'].numpy(),
      reward=np.linspace(-2,3,n,dtype='float32'),done=np.array([False]*(n-1)+[True]),aux_target=np.ones((n,3,4),'float32'),aux_weight=np.ones((n,3,4),'float32'))

class ActionPPOTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(123);self.m=ActorCritic()
    def test_joint_log_probability(self):
        b=batch(self.m);o=self.m.evaluate(torch.tensor(b['x']),torch.tensor(b['g']),torch.tensor(b['actions']))
        dist=torch.distributions.Categorical(logits=o['logits'])
        torch.testing.assert_close(o['log_prob'],dist.log_prob(torch.tensor(b['actions'])).sum(-1))
    def test_integer_category_domain(self):
        b=batch(self.m);self.assertEqual(b['actions'].dtype,np.int64);self.assertTrue(np.isin(b['actions']-2,[-2,-1,0,1,2]).all())
    def test_argmax_can_select_nonzero(self):
        with torch.no_grad():self.m.actor.weight.zero_();self.m.actor.bias[4]=10
        b=batch(self.m);o=self.m.act(b['x'][0],b['g'][0],True);self.assertTrue((o['actions']==4).all())
    def test_entropy_joint_and_mean(self):
        b=batch(self.m);o=self.m.evaluate(torch.tensor(b['x']),torch.tensor(b['g']))
        torch.testing.assert_close(o['entropy_joint'],12*o['entropy_mean'])
    def test_invalid_categories_rejected(self):
        b=batch(self.m);b['actions'][0,0]=5
        with self.assertRaises(ValueError):self.m.evaluate(torch.tensor(b['x']),torch.tensor(b['g']),torch.tensor(b['actions']))
    def test_act_does_not_modify_weights(self):
        h=model_hash(self.m);b=batch(self.m);self.m.act(b['x'][0],b['g'][0]);self.assertEqual(h,model_hash(self.m))
    def test_actor_critic_receive_real_gradients(self):
        b=batch(self.m);s=PPO(self.m).update(b)
        self.assertGreater(s['actor_head_change'],0);self.assertGreater(s['critic_head_change'],0);self.assertGreater(s['policy_head_gradient_norm'],0)
        self.assertLess(s['replay_logprob_error'],2e-5)
    def test_replay_detects_changed_log_probability(self):
        b=batch(self.m);b['log_prob']+=.1
        with self.assertRaises(ValueError):PPO(self.m).update(b)
    def test_incomplete_rollout_rejected(self):
        b=batch(self.m);b['done'][-1]=False
        with self.assertRaises(ValueError):PPO(self.m).update(b)
    def test_projected_float_actions_rejected(self):
        b=batch(self.m);b['actions']=b['actions'].astype('float32')
        with self.assertRaises(ValueError):PPO(self.m).update(b)
    def test_gae_stops_at_each_episode(self):
        a,r=gae([1,2,3,4],[0]*4,[False,True,False,True],lam=1)
        np.testing.assert_equal(r,[3,2,7,4])
    def test_gae_bootstrap(self):
        a,r=gae([1],[2],[False],gamma=.9,last_value=3);self.assertAlmostEqual(float(r[0]),3.7,places=5)
    def test_checkpoint_exact_reload(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt';ppo=PPO(self.m);ppo.update(batch(self.m));prices=RatePrices()
            save_checkpoint(p,self.m,ppo,prices,1,2,{'source':'test'},0.,7)
            restored=ActorCritic();c=load_checkpoint(p,restored,PPO(restored),{'source':'test'})
            self.assertEqual(c['episodes'],2);self.assertEqual(model_hash(self.m),model_hash(restored))
            with self.assertRaises(ValueError):load_checkpoint(p,restored,PPO(restored),{'source':'changed'})

class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)/'s.yuv';self.p.write_bytes(bytes([90])*FB*32)
        self.s=StateBuilder(self.p,[[]]*32,32,120,1)
        self.e=dict(schema='vvc-rl-online/1',k=0,bits_before=0,width=480,height=320,ctu=128,poc=31,qp=[30]*12,activity=[1]*12,slice_qp=30,temporal_layer=0,reference=True,slice_type=2)
    def tearDown(self):self.s.close();self.tmp.cleanup()
    def test_shapes_and_finiteness(self):
        x,g=self.s.build(self.e);self.assertEqual(x.shape,(8,3,4));self.assertEqual(g.shape,(15,));self.assertTrue(np.isfinite(g).all())
    def test_edge_area(self):self.assertEqual(int(areas().sum()),480*320);self.assertEqual(areas()[-1,-1],64*96)
    def test_union_occupancy_not_double_counted(self):
        self.s.boxes=[[[0,0,128,128,.9],[0,0,128,128,.8]]]*32;x,g=self.s.build(self.e);self.assertEqual(x[0,0,0],1)
    def test_coding_order_history(self):
        self.s.advance(self.e,np.ones(12),dict(k=0,poc=31,bits=100,bits_total=100));self.e.update(k=1,poc=15,bits_before=100)
        x,g=self.s.build(self.e);self.assertTrue((x[6]==.5).all());self.assertEqual(g[13],1)
    def test_noncausal_bits_rejected(self):
        self.e['bits_before']=12
        with self.assertRaises(ValueError):self.s.build(self.e)
    def test_invalid_native_geometry_rejected(self):
        self.e['ctu']=32
        with self.assertRaises(ValueError):self.s.build(self.e)
    def test_source_state_determinism(self):
        x,g=self.s.build(self.e);y,h=self.s.build(self.e);np.testing.assert_equal(x,y);np.testing.assert_equal(g,h)
    def test_lock_byte_exact(self):self.assertEqual(digest(ASSETS/'dataset_lock.json'),DATASET_SHA)
    def test_profile_rejects_parallel_pictures(self):
        with self.assertRaises(ValueError):Profile(threads=4).validate()

class RewardTests(unittest.TestCase):
    def test_roi_heuristic_uses_same_semantic_boxes(self):
        a=roi_action([[260,140,300,180,.8]],True).reshape(3,4);self.assertEqual(a[1,2],-2);self.assertEqual(a[0,0],1)
    def test_roi_empty_frame_noop(self):self.assertTrue((roi_action([],True)==0).all())
    def test_empty_teacher_and_prediction(self):self.assertEqual(task([],[])['score'],1);self.assertEqual(task([],[])['legacy_f1'],0)
    def test_false_positive_on_negative_frame(self):self.assertEqual(task([],[[1,1,10,10,.8]])['score'],0)
    def test_identical_detection(self):self.assertEqual(task([[1,1,10,10,.8]],[[1,1,10,10,.8]])['score'],1)
    def test_auxiliary_centers_and_empty_mask(self):
        r=task([[260,140,300,180,.8]],[]);self.assertEqual(r['aux_weight'][1,2],1);self.assertEqual(r['aux_weight'].sum(),1);self.assertEqual(r['aux_target'].sum(),0)
    def test_reward_telescope(self):
        rows=[reward(.8,.7,1000,950,32,128000,2) for _ in range(32)]
        self.assertAlmostEqual(sum(r['raw'] for r in rows),.1-2*1600/128000)
    def test_rate_price_moves_with_budget(self):
        p=RatePrices();p.update([dict(rate=120,bits=200,budget=100)]);self.assertAlmostEqual(p.values['120'],1.05)
    def test_validation_cannot_change_prices(self):
        with self.assertRaises(ValueError):RatePrices().update([],training=False)
    def test_noop_reward_exact_zero(self):self.assertEqual(reward(.2,.2,800,800,32,128000,1)['raw'],0)

class EvidenceTests(unittest.TestCase):
    def test_safe_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);r=d/'run';r.mkdir();(r/'model.txt').write_text('proof');p=export_run(r,d/'safe.zip');import_run(p,d/'restored');self.assertEqual((d/'restored/model.txt').read_text(),'proof')
    def test_safe_corruption_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'bad.bin'
            with zipfile.ZipFile(p,'w') as z:z.writestr('data','x');z.writestr('EXPORT_MANIFEST.json',json.dumps({'files':{'data':'wrong'}}))
            with self.assertRaises(ValueError):import_run(p,d/'restored')
            self.assertFalse((d/'restored').exists())
    def test_archive_traversal_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'bad.bin'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','x');z.writestr('EXPORT_MANIFEST.json',json.dumps({'files':{'../escape':hashlib.sha256(b'x').hexdigest()}}))
            with self.assertRaises(ValueError):import_run(p,d/'restored')
    def test_failed_gate_blocks_training(self):
        with tempfile.TemporaryDirectory() as d:
            write_gate(d,{},dict.fromkeys(MANDATORY,False),[])
            with self.assertRaises(RuntimeError):require_gate(d,{})
    def test_gate_binds_source_and_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'proof';p.write_text('evidence');write_gate(d,{'source':'a'},dict.fromkeys(MANDATORY,True),[p]);require_gate(d,{'source':'a'})
            with self.assertRaises(RuntimeError):require_gate(d,{'source':'b'})
            p.write_text('changed')
            with self.assertRaises(RuntimeError):require_gate(d,{'source':'a'})
    def test_synthetic_gate_not_sufficient(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'NATIVE_SMOKE.json').write_text('{"status":"PASS_NATIVE_MECHANICS_ONLY"}')
            with self.assertRaises(RuntimeError):require_gate(d,{})
    def test_empty_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            write_gate(d,{},dict.fromkeys(MANDATORY,True),[])
            with self.assertRaises(RuntimeError):require_gate(d,{})

if __name__=='__main__':unittest.main()
