"""Frozen confidence-ordered detector matching, extracted without RL legacy code."""
import numpy as np
W,H=480,320
FRAME_BYTES=W*H*3//2

def boxes_checked(boxes):
    boxes=np.asarray(boxes,dtype=np.float64)
    if boxes.size==0:return np.empty((0,5),dtype=np.float64)
    if boxes.ndim!=2 or boxes.shape[1]!=5 or not np.isfinite(boxes).all():raise ValueError("Expected finite [x1,y1,x2,y2,confidence]")
    if ((boxes[:,2:4]<=boxes[:,:2]).any() or (boxes[:,:2]<0).any()
            or (boxes[:,[0,2]]>W).any() or (boxes[:,[1,3]]>H).any()
            or (boxes[:,4]<0).any() or (boxes[:,4]>1).any()):raise ValueError("Invalid detection coordinates/confidence")
    return boxes

def detection_assignments(raw,reconstructed):
    """Canonical confidence-ordered one-to-one matches, shared by metrics and auxiliary labels."""
    raw,pred=boxes_checked(raw),boxes_checked(reconstructed)
    available=set(range(len(raw)))
    correct=np.zeros(len(pred),dtype=bool)
    matched=np.full(len(pred),-1,dtype=np.int64)
    for i in np.argsort(-pred[:,4],kind="stable"):
        best,maximum=None,-1.
        for j in sorted(available):
            lo=np.maximum(pred[i,:2],raw[j,:2]);hi=np.minimum(pred[i,2:4],raw[j,2:4])
            overlap=float(np.prod(np.maximum(0,hi-lo)))
            union=float(np.prod(pred[i,2:4]-pred[i,:2])+np.prod(raw[j,2:4]-raw[j,:2])-overlap)
            iou=overlap/union if union>0 else 0.
            if iou>maximum:best,maximum=j,iou
        if best is not None and maximum>=.5:correct[i]=True;matched[i]=best;available.remove(best)
    return raw,pred,correct,matched
