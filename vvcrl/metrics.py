import numpy as np
from .locked.method import detection_assignments,boxes_checked
from .config import CTU,NX,NY

def task(source,reconstructed):
    ref,pred,correct,matched=detection_assignments(source,reconstructed)
    tp=int(correct.sum());fp=len(pred)-tp;fn=len(ref)-tp
    denom=2*tp+fp+fn
    score=2*tp/denom if denom else 1.
    # Auxiliary target describes source-object retention, not a causal CTU reward.
    counts=np.zeros((NY,NX),np.float32);kept=np.zeros_like(counts)
    matched_ids=set(matched[correct].tolist())
    for j,b in enumerate(ref):
        x=min(NX-1,int((b[0]+b[2])/2)//CTU);y=min(NY-1,int((b[1]+b[3])/2)//CTU)
        counts[y,x]+=1;kept[y,x]+=int(j in matched_ids)
    target=np.divide(kept,counts,out=np.zeros_like(kept),where=counts>0)
    return {'score':score,'legacy_f1':2*tp/denom if denom else 0.,'tp':tp,'fp':fp,'fn':fn,
            'aux_target':target,'aux_weight':counts}

def summary(frames):
    if not frames:raise ValueError('No completed frames')
    tp,fp,fn=[sum(x[k] for x in frames) for k in ('tp','fp','fn')]
    return {'task_retention_f1_v2':float(np.mean([x['score'] for x in frames])),
      'legacy_mean_f1':float(np.mean([x['legacy_f1'] for x in frames])),
      'micro_precision':tp/(tp+fp) if tp+fp else None,'micro_recall':tp/(tp+fn) if tp+fn else None,
      'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'tp':tp,'fp':fp,'fn':fn}
