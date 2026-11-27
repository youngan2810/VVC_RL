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
