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


# Ground-truth Validation metrics are separate from teacher reward metrics.
import contextlib,io
from scipy.interpolate import PchipInterpolator

def coco_ap(frames, predictions):
    """Pool confidence-ranked detections over frames, never average frame AP.

    frames: [{image_id, boxes:[[x1,y1,x2,y2],...]}].
    predictions: [{image_id, boxes:[[x1,y1,x2,y2,score],...]}].
    COCO IoU .50:.05:.95, 101 recall thresholds, area all, maxDets 300.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not frames:raise ValueError("Missing annotated frames")
    ids = [int(x['image_id']) for x in frames]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate evaluation image IDs')
    if sorted(ids) != sorted(int(x['image_id']) for x in predictions):
        raise ValueError('Predictions must cover exactly the evaluated frames')
    anns, dets = [], []
    for frame in frames:
        for box in frame['boxes']:
            x1,y1,x2,y2=map(float,box)
            if not np.isfinite(box).all() or x2<=x1 or y2<=y1:
                raise ValueError('Invalid ground truth box')
            anns.append(dict(id=len(anns)+1,image_id=int(frame['image_id']),category_id=1,
                             bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1),iscrowd=0))
    if not anns:
        return dict(ap50=None,ap50_95=None,images=len(ids),gt_boxes=0,reason='no_car_ground_truth')
    for frame in predictions:
        for box in frame['boxes']:
            x1,y1,x2,y2,score=map(float,box)
            if not np.isfinite(box).all() or x2<=x1 or y2<=y1 or not 0<=score<=1:
                raise ValueError('Invalid prediction box')
            dets.append(dict(image_id=int(frame['image_id']),category_id=1,
                             bbox=[x1,y1,x2-x1,y2-y1],score=score))
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO()
        gt.dataset=dict(info={},images=[dict(id=i,width=480,height=320) for i in ids],
                        annotations=anns,categories=[dict(id=1,name='car')])
        gt.createIndex()
        if dets:
            dt=gt.loadRes(dets)
        else:
            dt=COCO();dt.dataset=dict(info={},images=gt.dataset['images'],annotations=[],
                                     categories=gt.dataset['categories']);dt.createIndex()
        ev=COCOeval(gt,dt,'bbox')
        ev.params.imgIds=ids;ev.params.catIds=[1];ev.params.maxDets=[1,10,300]
        ev.evaluate();ev.accumulate()
    precision=ev.eval['precision'][:,:,0,0,-1]
    valid=precision[precision>=0]
    at50=precision[0];at50=at50[at50>=0]
    return dict(ap50=float(at50.mean()) if len(at50) else None,
                ap50_95=float(valid.mean()) if len(valid) else None,
                images=len(ids),gt_boxes=len(anns),detections=len(dets))


def frontier(points):
    """Deterministic Pareto frontier, with duplicate rates merged before pruning."""
    rates={}
    for rate,quality in points:
        if quality is None:continue
        rate,quality=float(rate),float(quality)
        if not np.isfinite([rate,quality]).all() or rate<=0 or not 0<=quality<=1:
            raise ValueError('Invalid RD point')
        rates[rate]=max(quality,rates.get(rate,-1.))
    out=[]
    for rate,quality in sorted(rates.items()):
        if not out or quality>out[-1][1]+1e-12:out.append((rate,quality))
    return out


def bd_metrics(anchor, test):
    a,b=frontier(anchor),frontier(test)
    result=dict(bd_rate_percent=None,bd_ap_pp=None,anchor_frontier=len(a),test_frontier=len(b))
    if min(len(a),len(b))<2:return dict(result,reason='insufficient_monotone_frontier')
    ar,aq=np.asarray(a).T;br,bq=np.asarray(b).T
    lo,hi=max(aq[0],bq[0]),min(aq[-1],bq[-1])
    result['ap_overlap_pp']=float(max(0.,hi-lo)*100)
    if hi>lo+1e-12:
        ai=PchipInterpolator(aq,np.log(ar),extrapolate=False)
        bi=PchipInterpolator(bq,np.log(br),extrapolate=False)
        delta=(bi.integrate(lo,hi)-ai.integrate(lo,hi))/(hi-lo)
        result['bd_rate_percent']=float(np.expm1(delta)*100)
    lo,hi=max(np.log(ar[0]),np.log(br[0])),min(np.log(ar[-1]),np.log(br[-1]))
    if hi>lo+1e-12:
        ai=PchipInterpolator(np.log(ar),aq,extrapolate=False)
        bi=PchipInterpolator(np.log(br),bq,extrapolate=False)
        result['bd_ap_pp']=float((bi.integrate(lo,hi)-ai.integrate(lo,hi))/(hi-lo)*100)
    result['reason']='ok' if result['bd_rate_percent'] is not None else 'no_common_ap_range'
    return result
