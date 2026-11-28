"""Useful-learning diagnostics use the same cap as the training objective."""
import numpy as np
from .config import DELTAS,RewardConfig
from .reward import bitrate_cap

def validation_summary(rows,expected_pairs,reward_config=RewardConfig()):
    reward_config.validate();pairs=list(expected_pairs);expected=set(pairs);seen=set();detail=[]
    if len(pairs)!=len(expected):raise ValueError('Duplicate expected validation pair')
    for row in rows:
        r=row['result'];key=(row['video'],r['rate'])
        if key not in expected or key in seen:raise ValueError('Unexpected or duplicate validation pair')
        if r.get('split')!='val':raise ValueError('Checkpoint selection requires validation only')
        seen.add(key);cap=bitrate_cap(r['budget'],r['anchor_bits'],reward_config)
        scores=[r['task_retention_f1_v2'],r['anchor_quality'],r.get('task_utility',r['task_retention_f1_v2']),r.get('anchor_task_utility',r['anchor_quality'])]
        if not np.isfinite(scores+[r['bits'],r['actual_kbps']]).all() or any(not 0<=s<=1 for s in scores) or min(r['bits'],r['actual_kbps'])<=0:raise ValueError('Invalid validation quality or bit accounting')
        for name in ('matched_iou_f1_v1','anchor_matched_iou_f1_v1'):
            if name in r and (not np.isfinite(r[name]) or not 0<=r[name]<=1):raise ValueError('Invalid validation localization score')
        if 'utility_version' in r and r['utility_version']!=reward_config.utility:raise ValueError('Validation objective binding differs')
        if reward_config.utility!='task_retention_f1_v2' and (r.get('utility_version')!=reward_config.utility or 'task_utility' not in r or 'anchor_task_utility' not in r):raise ValueError('Validation utility missing')
        detail.append({'video':row['video'],'rate':r['rate'],'delta_quality':r['task_retention_f1_v2']-r['anchor_quality'],
            'actual_kbps':r['actual_kbps'],'delta_bitrate_pct':100*(r['bits']/r['anchor_bits']-1),
            'budget_cap_bits':cap,'within_cap':r['bits']<=cap,'no_extra_bits':r['bits']<=r['anchor_bits']*(1+reward_config.rate_tolerance),
            'budget_ok':r['bits']<=r['budget']*(1+reward_config.rate_tolerance),
            'delta_utility':r.get('task_utility',r['task_retention_f1_v2'])-r.get('anchor_task_utility',r['anchor_quality']),
            'delta_matched_iou_f1':r['matched_iou_f1_v1']-r['anchor_matched_iou_f1_v1'] if 'matched_iou_f1_v1' in r and 'anchor_matched_iou_f1_v1' in r else None})
    complete=bool(expected) and seen==expected;feasible=bool(detail) and all(p['within_cap'] for p in detail)
    gain=float(np.mean([p['delta_quality'] for p in detail])) if detail else None
    return {'complete':complete,'feasible':feasible,'macro_delta_quality':gain,'points':detail,'utility_version':reward_config.utility,
        'macro_delta_utility':float(np.mean([p['delta_utility'] for p in detail])) if detail else None,
        'selection_metric':'legacy teacher F1 retention (unchanged; not GT-mAP)',
        'quality_regressions':sum(p['delta_quality']<0 for p in detail),
        'utility_gain_f1_loss_pairs':sum(p['delta_utility']>1e-12 and p['delta_quality']< -1e-12 for p in detail),
        'eligible_checkpoint':complete and feasible and gain is not None and gain>0,
        'eligible_utility_checkpoint':complete and feasible and bool(detail) and float(np.mean([p['delta_utility'] for p in detail]))>0,
        'interpretation':'A checkpoint needs complete validation, positive macro F1 gain and every clip within the shared bitrate cap. This is not a Test or ground-truth mAP claim.'}

def rollout_summary(rows):
    if not rows:return {'clips':0}
    results=[x['result'] for x in rows]
    return {'clips':len(rows),'mean_task_gain':float(np.mean([r['clip_objective']['task_component'] for r in results])),
        'mean_bitrate_component':float(np.mean([r['clip_objective']['bit_component'] for r in results])),
        'mean_raw_return':float(np.mean([r['raw_return'] for r in results])),
        'within_cap_fraction':float(np.mean([r['clip_objective']['within_cap'] for r in results])),
        'utility_gain_f1_loss_clips':sum(r['clip_objective']['task_component']>1e-12 and r['task_retention_f1_v2']<r['anchor_quality']-1e-12 for r in results),
        'mean_entropy':float(np.mean([r['mean_action_entropy'] for r in results if r['mean_action_entropy'] is not None])) if any(r['mean_action_entropy'] is not None for r in results) else None,
        'requested_action_distribution':{str(d):float(np.mean([r['requested_action_distribution'].get(str(d),0) for r in results])) for d in DELTAS},
        'effective_qg_candidate_counts':{str(d):sum(r['action_audit']['effective_delta_histogram'].get(str(d),0) for r in results) for d in DELTAS},
        'nonzero_requests_zeroed':sum(r['action_audit']['nonzero_requests_zeroed'] for r in results)}


def rd_validation_summary(rows,expected_pairs,reward_config=RewardConfig()):
    """Select on pooled real-label RD curves; teacher metrics remain diagnostics.

    AP pools confidence-ranked predictions across four videos at each rate.
    Rate is SUM(bits)/SUM(duration), never target rate or AP averaged by frame.
    Both curves and all cap decisions are retained for auditing.
    """
    import hashlib
    from .config import RATES,VALIDATION_SELECTION,canonical
    from .metrics import coco_ap,bd_metrics
    rows=list(rows);pairs=list(expected_pairs)
    report=validation_summary(rows,pairs,reward_config)
    videos=sorted({v for v,_ in pairs})
    if len(videos)!=4 or set(pairs)!={(v,r) for v in videos for r in RATES}:
        raise ValueError('RD selection requires four Validation videos at all five rates')
    report.update(selection_metric=VALIDATION_SELECTION,eligible_checkpoint=False,
                  eligible_utility_checkpoint=False,rd_curves=[],bd_ap50=None,bd_ap50_95=None,
                  interpretation='Validation-only real-GT RD selection. Teacher F1/utility are diagnostic, never fallback selection.',
                  selection_reason='incomplete_validation')
    gt_by_video={}
    for row in rows:
        result=row['result'];ap=result.get('validation_ap');anchor=result.get('anchor_validation_ap')
        if ap is None or anchor is None:raise ValueError('Missing real-label Validation AP; never fall back to F1')
        for proof in (ap,anchor):
            frames=proof['ground_truth']
            if proof['protocol']!=VALIDATION_SELECTION or not frames or any(f['video_id']!=row['video'] for f in frames):
                raise ValueError('Validation GT/protocol mismatch')
            if hashlib.sha256(canonical(frames).encode()).hexdigest()!=proof['gt_sha256']:
                raise ValueError('Validation GT hash mismatch')
        if any(ap[k]!=anchor[k] for k in ('protocol','gt_sha256','ground_truth')):
            raise ValueError('Unpaired Validation GT')
        if row['video'] in gt_by_video and gt_by_video[row['video']]!=ap['ground_truth']:
            raise ValueError('GT coverage changed between rates')
        gt_by_video[row['video']]=ap['ground_truth']
        profile=result['identity']['profile'];duration=profile['frames']/profile['fps']
        if not np.isfinite(duration) or duration<=0 or not np.isclose(result['actual_kbps'],result['bits']/duration/1000,rtol=0,atol=1e-8):
            raise ValueError('Validation rate/duration accounting mismatch')
    if not report['complete']:return report
    for rate in RATES:
        current=sorted((x for x in rows if x['result']['rate']==rate),key=lambda x:x['video'])
        gt=sum([x['result']['validation_ap']['ground_truth'] for x in current],[])
        duration=sum(x['result']['identity']['profile']['frames']/x['result']['identity']['profile']['fps'] for x in current)
        point={'target_kbps':rate,'duration_seconds':duration,'videos':len(current)}
        for name,proof_key,bits_key in (('qpa','anchor_validation_ap','anchor_bits'),('ppo','validation_ap','bits')):
            predictions=sum([x['result'][proof_key]['predictions'] for x in current],[])
            point[name]={'actual_kbps':sum(x['result'][bits_key] for x in current)/duration/1000,
                         **coco_ap(gt,predictions)}
        report['rd_curves'].append(point)
    for metric,key in (('ap50','bd_ap50'),('ap50_95','bd_ap50_95')):
        report[key]=bd_metrics([(p['qpa']['actual_kbps'],p['qpa'][metric]) for p in report['rd_curves']],
                               [(p['ppo']['actual_kbps'],p['ppo'][metric]) for p in report['rd_curves']])
    primary=report['bd_ap50_95'];rate,quality=primary['bd_rate_percent'],primary['bd_ap_pp']
    valid=rate is not None and quality is not None and np.isfinite([rate,quality]).all()
    report['eligible_checkpoint']=bool(report['feasible'] and valid and rate< -1e-9 and quality>1e-9)
    report['selection_reason']=('bitrate_cap_failed' if not report['feasible'] else
        'undefined_rd_overlap' if not valid else 'improving_rd' if report['eligible_checkpoint'] else 'no_joint_rd_improvement')
    return report


def better_rd_checkpoint(report,best):
    if not report['eligible_checkpoint']:return False
    candidate=report['bd_ap50_95']
    if best is None:return True
    delta=candidate['bd_rate_percent']-best['bd_rate_percent']
    return delta< -1e-9 or (abs(delta)<=1e-9 and candidate['bd_ap_pp']>best['bd_ap_pp']+1e-9)


def selection_progress(best,bind):
    return {'best_validation_rd':best} if 'validation_selection' in bind else {'best_validation_gain':best}
