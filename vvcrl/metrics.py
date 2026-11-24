import numpy as np
from .locked.method import detection_assignments,boxes_checked
from .config import CTU,NX,NY

def task(source,reconstructed):
    ref,pred,correct,matched=detection_assignments(source,reconstructed)
    tp=int(correct.sum());fp=len(pred)-tp;fn=len(ref)-tp
    denom=2*tp+fp+fn
    score=2*tp/denom if denom else 1.
    # Same frozen confidence-ordered one-to-one assignments and IoU>=.5.
    # Matched localization quality replaces the unit TP credit, not its count.
    # Unmatched and duplicate detections remain in the denominator.
    pairs=pred[correct];refs=ref[matched[correct]]
    intersection=np.prod(np.maximum(0.,np.minimum(pairs[:,2:4],refs[:,2:4])-np.maximum(pairs[:,:2],refs[:,:2])),axis=1)
    union=np.prod(pairs[:,2:4]-pairs[:,:2],axis=1)+np.prod(refs[:,2:4]-refs[:,:2],axis=1)-intersection
    iou=intersection/union if len(pairs) else np.empty(0,np.float64)
    soft_iou=float(np.clip(2*iou.sum()/denom,0.,1.)) if denom else 1.
    # Auxiliary target describes source-object retention, not a causal CTU reward.
    counts=np.zeros((NY,NX),np.float32);kept=np.zeros_like(counts)
    matched_ids=set(matched[correct].tolist())
    for j,b in enumerate(ref):
        x=min(NX-1,int((b[0]+b[2])/2)//CTU);y=min(NY-1,int((b[1]+b[3])/2)//CTU)
        counts[y,x]+=1;kept[y,x]+=int(j in matched_ids)
    target=np.divide(kept,counts,out=np.zeros_like(kept),where=counts>0)
    return {'score':score,'matched_iou_f1_v1':soft_iou,'legacy_f1':2*tp/denom if denom else 0.,'tp':tp,'fp':fp,'fn':fn,
            'aux_target':target,'aux_weight':counts}

def summary(frames):
    if not frames:raise ValueError('No completed frames')
    tp,fp,fn=[sum(x[k] for x in frames) for k in ('tp','fp','fn')]
    result={'task_retention_f1_v2':float(np.mean([x['score'] for x in frames])),
      'legacy_mean_f1':float(np.mean([x['legacy_f1'] for x in frames])),
      'micro_precision':tp/(tp+fp) if tp+fp else None,'micro_recall':tp/(tp+fn) if tp+fn else None,
      'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'tp':tp,'fp':fp,'fn':fn}
    if all('matched_iou_f1_v1' in x for x in frames):result['matched_iou_f1_v1']=float(np.mean([x['matched_iou_f1_v1'] for x in frames]))
    return result
