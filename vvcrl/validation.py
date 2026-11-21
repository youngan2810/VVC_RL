"""Useful-learning diagnostics use the same cap as the training objective."""
import numpy as np
from .config import DELTAS,RewardConfig
from .reward import bitrate_cap

def validation_summary(rows,expected_pairs,reward_config=RewardConfig()):
    expected=set(expected_pairs);seen=set();detail=[]
    for row in rows:
        r=row['result'];key=(row['video'],r['rate'])
        if key not in expected or key in seen:raise ValueError('Unexpected or duplicate validation pair')
        if r.get('split')!='val':raise ValueError('Checkpoint selection requires validation only')
        seen.add(key);cap=bitrate_cap(r['budget'],r['anchor_bits'],reward_config)
        detail.append({'video':row['video'],'rate':r['rate'],'delta_quality':r['task_retention_f1_v2']-r['anchor_quality'],
            'actual_kbps':r['actual_kbps'],'delta_bitrate_pct':100*(r['bits']/r['anchor_bits']-1),
            'budget_cap_bits':cap,'within_cap':r['bits']<=cap,'no_extra_bits':r['bits']<=r['anchor_bits']*(1+reward_config.rate_tolerance),
            'budget_ok':r['bits']<=r['budget']*(1+reward_config.rate_tolerance)})
    complete=bool(expected) and seen==expected;feasible=bool(detail) and all(p['within_cap'] for p in detail)
    gain=float(np.mean([p['delta_quality'] for p in detail])) if detail else None
    return {'complete':complete,'feasible':feasible,'macro_delta_quality':gain,'points':detail,
        'quality_regressions':sum(p['delta_quality']<0 for p in detail),
        'eligible_checkpoint':complete and feasible and gain is not None and gain>0,
        'interpretation':'A checkpoint needs complete validation, positive macro F1 gain and every clip within the shared bitrate cap. This is not a Test or ground-truth mAP claim.'}

def rollout_summary(rows):
    if not rows:return {'clips':0}
    results=[x['result'] for x in rows]
    return {'clips':len(rows),'mean_task_gain':float(np.mean([r['clip_objective']['task_component'] for r in results])),
        'mean_bitrate_component':float(np.mean([r['clip_objective']['bit_component'] for r in results])),
        'mean_raw_return':float(np.mean([r['raw_return'] for r in results])),
        'within_cap_fraction':float(np.mean([r['clip_objective']['within_cap'] for r in results])),
        'mean_entropy':float(np.mean([r['mean_action_entropy'] for r in results if r['mean_action_entropy'] is not None])) if any(r['mean_action_entropy'] is not None for r in results) else None,
        'requested_action_distribution':{str(d):float(np.mean([r['requested_action_distribution'].get(str(d),0) for r in results])) for d in DELTAS},
        'effective_qg_candidate_counts':{str(d):sum(r['action_audit']['effective_delta_histogram'].get(str(d),0) for r in results) for d in DELTAS},
        'nonzero_requests_zeroed':sum(r['action_audit']['nonzero_requests_zeroed'] for r in results)}
