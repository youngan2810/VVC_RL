"""Fixed semantic baselines with the same source detections as PPO.

These maps are not bitrate-matched by construction; compare actual bits and
task quality, never label zero-mean delta as a rate guarantee.
"""
import numpy as np
from .locked.method import boxes_checked

def roi_action(boxes,redistribute=False):
    roi=np.zeros((3,4),bool)
    for b in boxes_checked(boxes):
        left,top=int(np.floor(b[0]/128)),int(np.floor(b[1]/128))
        right,bottom=int(np.ceil(b[2]/128)),int(np.ceil(b[3]/128))
        roi[top:bottom,left:right]=True
    if not roi.any():return np.zeros(12,np.int64)
    return np.where(roi,-2,1 if redistribute else 0).astype(np.int64).ravel()
