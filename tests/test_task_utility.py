import unittest
import numpy as np
from vvcrl.config import RewardConfig,TASK
from vvcrl.metrics import task,summary
from vvcrl.reward import clip_objective,reward,ProgressiveBudget
from vvcrl.validation import validation_summary

class UtilityTests(unittest.TestCase):
    def test_empty_duplicate_localization_confidence(self):
        a=[[0,0,100,100,.8]]
        self.assertEqual(task([],[])['matched_iou_f1_v1'],1.)
        self.assertEqual(task(a,[])['matched_iou_f1_v1'],0.)
        self.assertEqual(task([],a)['matched_iou_f1_v1'],0.)
        good=task(a,a);shift=task(a,[[10,0,110,100,.8]])
        self.assertEqual(good['score'],shift['score'])
        self.assertAlmostEqual(shift['matched_iou_f1_v1'],90/110)
        self.assertLess(shift['matched_iou_f1_v1'],good['matched_iou_f1_v1'])
        self.assertAlmostEqual(task(a,a+a)['matched_iou_f1_v1'],2/3)
        self.assertEqual(task(a,[[0,0,100,100,.99]])['matched_iou_f1_v1'],1.)
        self.assertEqual(task(a,[[200,0,300,100,.8]])['matched_iou_f1_v1'],0.)
    def test_reward_uses_same_utility_on_both_sides(self):
        a=[[0,0,100,100,.8]];u=task(a,[[10,0,110,100,.8]]);ref=task(a,a)
        for version in ('task-budget-v2-soft-iou','task-budget-v3-progressive-soft-iou'):
            cfg=RewardConfig(version);self.assertEqual(cfg.utility,'matched_iou_f1_v1')
            context=ProgressiveBudget(1,100,cfg,{'bits':100,'by_poc':{'0':{'bits':100}}}).step(0,100,100) if cfg.progressive else None
            r=reward(u[cfg.utility],ref[cfg.utility],100,100,1,100,1,config=cfg,terminal=True,total_bits=100,anchor_total_bits=100,progress=context)
            self.assertAlmostEqual(r['raw'],90/110-1)
            self.assertAlmostEqual(r['raw'],clip_objective(u[cfg.utility],ref[cfg.utility],100,100,100,1,cfg)['raw'])
        self.assertEqual(RewardConfig().utility,TASK)
    def test_summary_and_selection_keep_legacy_f1(self):
        a=[[0,0,100,100,.8]];r=task(a,[[10,0,110,100,.8]])
        s=summary([r]);self.assertEqual(s['task_retention_f1_v2'],1)
        self.assertAlmostEqual(s['matched_iou_f1_v1'],90/110)
        cfg=RewardConfig('task-budget-v2-soft-iou')
        result=dict(split='val',rate=120,budget=100,anchor_bits=100,bits=100,actual_kbps=120,
                    task_retention_f1_v2=.8,anchor_quality=.7,task_utility=.5,anchor_task_utility=.6,utility_version=cfg.utility)
        report=validation_summary([dict(video='v',result=result)],[('v',120)],cfg)
        self.assertTrue(report['eligible_checkpoint']);self.assertAlmostEqual(report['macro_delta_utility'],-.1)
        with self.assertRaises(ValueError):validation_summary([dict(video='v',result=dict(result,utility_version=TASK))],[('v',120)],cfg)

    def test_utility_gain_f1_loss_is_visible_but_never_primary_success(self):
        cfg=RewardConfig('task-budget-v4-online-soft-iou')
        result=dict(split='val',rate=120,budget=100,anchor_bits=100,bits=100,actual_kbps=120,
                    task_retention_f1_v2=.69,anchor_quality=.7,task_utility=.61,anchor_task_utility=.6,utility_version=cfg.utility)
        report=validation_summary([dict(video='v',result=result)],[('v',120)],cfg)
        self.assertEqual(report['utility_gain_f1_loss_pairs'],1)
        self.assertFalse(report['eligible_checkpoint']);self.assertTrue(report['eligible_utility_checkpoint'])


class GroundTruthRDTests(unittest.TestCase):
    def test_coco_confidence_ranking_negative_frames_and_iou(self):
        from vvcrl.metrics import coco_ap
        gt=[dict(image_id=1,boxes=[[0,0,100,100]]),dict(image_id=2,boxes=[])]
        perfect=[dict(image_id=1,boxes=[[0,0,100,100,.8]]),dict(image_id=2,boxes=[])]
        self.assertAlmostEqual(coco_ap(gt,perfect)['ap50_95'],1)
        noisy=[perfect[0],dict(image_id=2,boxes=[[0,0,100,100,.99]])]
        self.assertLess(coco_ap(gt,noisy)['ap50'],1)
        shift=[dict(image_id=1,boxes=[[10,0,110,100,.8]]),perfect[1]]
        ap=coco_ap(gt,shift);self.assertGreater(ap['ap50'],ap['ap50_95'])
        self.assertEqual(coco_ap(gt,[dict(image_id=i,boxes=[]) for i in (1,2)])['ap50'],0)
        self.assertIsNone(coco_ap([dict(image_id=1,boxes=[])],[dict(image_id=1,boxes=[])])['ap50'])
        for frames,pred in ((gt,perfect[:1]),(gt+gt[:1],perfect),([],[])):
            with self.assertRaises(ValueError):coco_ap(frames,pred)

    def test_bd_actual_rate_scale_overlap_and_frontier(self):
        from vvcrl.metrics import bd_metrics,frontier
        a=[(50,.3),(80,.4),(120,.5),(160,.6),(200,.7)]
        b=[(r*.9,q) for r,q in a];rd=bd_metrics(a,b)
        self.assertAlmostEqual(rd['bd_rate_percent'],-10,places=10)
        self.assertGreater(rd['bd_ap_pp'],0)
        self.assertIsNone(bd_metrics(a,[(50,.8),(100,.9)])['bd_rate_percent'])
        self.assertIsNone(bd_metrics([(1,.5),(2,.5)],a)['bd_rate_percent'])
        self.assertEqual(frontier([(1,.2),(2,.3),(2,.8),(3,.4),(4,.9)]),[(1.,.2),(2.,.8),(4.,.9)])
        with self.assertRaises(ValueError):frontier([(1,float('nan'))])

    @staticmethod
    def rd_rows():
        import hashlib
        from vvcrl.config import RATES,VALIDATION_SELECTION,canonical
        rows=[]
        # Real COCO pooling, no mocked AP or synthetic success claims about VVC.
        for vi in range(4):
            for ri,rate in enumerate(RATES):
                gt=[dict(image_id=vi*100+j,video_id=f'v{vi}',poc=j*6,boxes=[[0,0,100,100]]) for j in range(10)]
                def proof(count):
                    pred=[dict(image_id=f['image_id'],boxes=[[0,0,100,100,.9]] if j<count else []) for j,f in enumerate(gt)]
                    return dict(protocol=VALIDATION_SELECTION,gt_sha256=hashlib.sha256(canonical(gt).encode()).hexdigest(),ground_truth=gt,predictions=pred)
                bits=rate*1000*256/30
                rows.append(dict(video=f'v{vi}',result=dict(split='val',rate=rate,bits=bits*.9,anchor_bits=bits,budget=bits,
                    actual_kbps=rate*.9,task_retention_f1_v2=.6,anchor_quality=.8,
                    identity={'profile':{'frames':256,'fps':30}},validation_ap=proof(ri+3),anchor_validation_ap=proof(ri+3))))
        return rows,[(r['video'],r['result']['rate']) for r in rows]

    def test_rd_selection_never_falls_back_to_teacher_and_checks_all_caps(self):
        import copy
        from vvcrl.validation import rd_validation_summary,better_rd_checkpoint
        rows,pairs=self.rd_rows();report=rd_validation_summary(rows,pairs)
        # Lower teacher F1 cannot veto an improving GT RD curve.
        self.assertLess(report['macro_delta_quality'],0)
        self.assertTrue(report['eligible_checkpoint'])
        self.assertAlmostEqual(report['bd_ap50_95']['bd_rate_percent'],-10,places=8)
        self.assertAlmostEqual(report['rd_curves'][0]['ppo']['actual_kbps'],45)
        self.assertTrue(better_rd_checkpoint(report,None))
        self.assertFalse(better_rd_checkpoint(report,report['bd_ap50_95']))
        bad=copy.deepcopy(rows);bad[0]['result']['bits']*=2;bad[0]['result']['actual_kbps']*=2
        self.assertFalse(rd_validation_summary(bad,pairs)['eligible_checkpoint'])
        self.assertFalse(rd_validation_summary(rows[:-1],pairs)['eligible_checkpoint'])
        bad=copy.deepcopy(rows);del bad[0]['result']['validation_ap']
        with self.assertRaisesRegex(ValueError,'Missing real-label'):rd_validation_summary(bad,pairs)
        bad=copy.deepcopy(rows);bad[0]['result']['actual_kbps']+=1
        with self.assertRaisesRegex(ValueError,'accounting'):rd_validation_summary(bad,pairs)
        for row in rows:
            row['result']['bits']=row['result']['anchor_bits'];row['result']['actual_kbps']=row['result']['rate']
            row['result']['task_retention_f1_v2']=.9
        self.assertFalse(rd_validation_summary(rows,pairs)['eligible_checkpoint'])

    def test_ap_inference_restores_teacher_confidence_even_on_failure(self):
        from types import SimpleNamespace
        from vvcrl.data import validation_ap_prediction
        class FakeDetector:
            model=SimpleNamespace(conf=.25,iou=.45,max_det=300,classes=[2],amp=False)
            def __call__(self,x):
                assert self.model.conf==.001
                raise RuntimeError('fixture inference failed')
        detector=FakeDetector()
        with self.assertRaisesRegex(RuntimeError,'fixture inference'):validation_ap_prediction(detector,None)
        self.assertEqual(detector.model.conf,.25)

    def test_gt_loader_filters_val_maps_timestamps_and_authenticates_labels(self):
        import tempfile,json,pandas as pd
        from pathlib import Path
        from unittest.mock import patch
        from vvcrl.data import validation_ground_truth
        from vvcrl.config import VALIDATION_SELECTION
        from vvcrl.locked.data import VideoCatalog,ASSETS
        ids=VideoCatalog('/missing',mode='mini',seed=20261111).selected['val']
        lock=json.loads((ASSETS/'dataset_lock.json').read_text(encoding='utf8'))
        rows=[]
        for vid in ids+['ffffffff-ffffffff']:
            for index in range(12):
                rows.append(dict(video=vid,frameIndex=index,category='car' if index else 'person',x1=0,y1=0,x2=128,y2=72))
            rows.append(dict(video=vid,frameIndex=12,category=None,x1=None,y1=None,x2=None,y2=None))
        df=pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'mot_labels.parquet').write_bytes(b'CPU_FIXTURE_NOT_REAL_LABELS')
            with patch('vvcrl.data.digest',return_value=VALIDATION_SELECTION['annotation_sha256']),patch('pandas.read_parquet',return_value=df):
                gt=validation_ground_truth(temp,256)
                self.assertEqual(set(gt),set(ids))
                for vid in ids:
                    f=gt[vid]['frames'];self.assertEqual(len(f),12);self.assertEqual(f[0]['boxes'],[])
                    self.assertEqual(f[1]['poc'],6)
                    dw,dh=next(x['display_size'] for x in lock['video_records'] if x['video_id']==vid)
                    np.testing.assert_allclose(f[1]['boxes'][0],[0,0,128*480/dw,72*320/dh])
            with self.assertRaises(FileNotFoundError):validation_ground_truth(temp,256)


    def test_rd_gate_cannot_reuse_legacy_mechanism_gate(self):
        import tempfile
        from pathlib import Path
        from vvcrl.gates import MANDATORY,required_checks,write_gate,require_gate
        from vvcrl.config import VALIDATION_SELECTION
        legacy={'source_sha256':'fixture'};rd=dict(legacy,validation_selection=VALIDATION_SELECTION)
        self.assertEqual(len(required_checks(rd)),len(MANDATORY)+1)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'proof.txt').write_text('fixture')
            write_gate(p,legacy,{k:True for k in MANDATORY},[p/'proof.txt'])
            with self.assertRaises(RuntimeError):require_gate(p,rd)
            with self.assertRaises(ValueError):write_gate(p,rd,{k:True for k in MANDATORY},[p/'proof.txt'])
            write_gate(p,rd,{k:True for k in required_checks(rd)},[p/'proof.txt'])
            self.assertEqual(require_gate(p,rd)['status'],'PASS_READY_FOR_MINI')


    def test_real_episode_ap_is_poc_aligned_and_never_replaces_teacher_reward(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from vvcrl.config import FB,Profile,D4_STATE,ONLINE_REWARD,VALIDATION_SELECTION,canonical,digest
        from vvcrl.episode import run_episode
        from vvcrl.model import ActorCritic
        class Native:
            def __init__(self,encoder,source,folder,profile,rate,**kw):
                self.folder=Path(folder);self.folder.mkdir();(self.folder/'recon').mkdir()
                self.k=self.bits=0
            def observation(self):
                return dict(schema='vvc-rl-online/1',k=self.k,bits_before=self.bits,width=480,height=320,
                    ctu=128,poc=31-self.k,qp=[30]*12,activity=[1]*12,slice_qp=30,temporal_layer=0,reference=True,slice_type=2)
            def step(self,d):
                self.bits+=1000;out=dict(k=self.k,poc=31-self.k,bits=1000,bits_total=self.bits,winning_qp=np.full((3,4),30.).tolist())
                self.k+=1;return out,np.full(FB,512,np.uint16)
            def finish(self):
                stream=self.folder/'stream.266';stream.write_bytes(bytes(self.bits//8))
                return dict(bits=self.bits,actual_kbps=self.bits*30/32/1000,bitstream_sha256=digest(stream))
            def close(self):pass
        class Detector:
            model=SimpleNamespace(conf=.25,iou=.45,max_det=300,classes=[2],amp=False)
            def __init__(self):self.thresholds=[]
            def __call__(self,yuv):
                self.thresholds.append(self.model.conf)
                return np.array([[0,0,100,100,.1]]) if self.model.conf==.001 else np.empty((0,5))
        frames=[dict(image_id=1,video_id='fixture',poc=0,boxes=[[0,0,100,100]])]
        gt=dict(protocol=VALIDATION_SELECTION,frames=frames,gt_sha256=hashlib.sha256(canonical(frames).encode()).hexdigest())
        detector=Detector()
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source=p/'source.yuv';source.write_bytes(bytes([128])*FB*32);encoder=p/'encoder';encoder.write_bytes(b'CPU_FIXTURE_ONLY')
            with patch('vvcrl.episode.NativeProcess',Native),patch('vvcrl.episode.verify_decode',return_value=True),patch('vvcrl.episode.audit_trace',return_value={'fixture':True}):
                result,_=run_episode(encoder=encoder,decoder=None,source=source,boxes=[[[0,0,100,100,.9]]]*32,
                    detector=detector,detector_identity={'fixture':True},profile=Profile(frames=32),rate=120,
                    folder=p/'val',model=ActorCritic(D4_STATE),split='val',deterministic=True,
                    reward_config=RewardConfig(ONLINE_REWARD),state_version=D4_STATE,validation_gt=gt)
            self.assertEqual(detector.thresholds.count(.25),32);self.assertEqual(detector.thresholds.count(.001),1)
            self.assertEqual(detector.model.conf,.25)
            self.assertAlmostEqual(result['validation_ap']['metrics']['ap50_95'],1)
            self.assertEqual(result['task_retention_f1_v2'],0)
            self.assertEqual([x['image_id'] for x in result['validation_ap']['predictions']],[1])
            self.assertTrue(all(not f['decoded_boxes'] for f in json.loads((p/'val/frames.json').read_text())))
