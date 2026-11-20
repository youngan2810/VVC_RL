import math
import numpy as np
from .config import W,H,CTU,NX,NY,FB
from .locked.method import boxes_checked

def areas():return np.outer([min(CTU,H-y*CTU) for y in range(NY)],[min(CTU,W-x*CTU) for x in range(NX)])
def pool(image):
    return np.asarray([[image[y*CTU:min(H,(y+1)*CTU),x*CTU:min(W,(x+1)*CTU)].mean() for x in range(NX)] for y in range(NY)],np.float32)

class StateBuilder:
    def __init__(self,source,boxes,frames,rate,price):
        self.frames,self.rate,self.price=frames,rate,float(price);self.boxes=boxes
        self.raw=np.memmap(source,dtype=np.uint8,mode='r',shape=(frames,FB))
        self.budget=rate*1000*frames/30
        self.last_action=np.zeros((NY,NX),np.float32);self.last_bits=0;self.last_poc=0;self.used=0;self.k=0
    def build(self,event):
        if event['schema']!='vvc-rl-online/1' or event['k']!=self.k or event['bits_before']!=self.used:raise ValueError('Noncausal or mismatched live observation')
        if (event['width'],event['height'],event['ctu'])!=(W,H,CTU):raise ValueError('Native grid changed')
        p=event['poc']
        if not 0<=p<self.frames:raise ValueError('POC out of window')
        occupancy=np.zeros((H,W),np.float32);confidence=np.zeros_like(occupancy);scale=np.zeros((NY,NX),np.float32)
        for b in boxes_checked(self.boxes[p]):
            l,t=int(math.floor(b[0])),int(math.floor(b[1]));r,bot=int(math.ceil(b[2])),int(math.ceil(b[3]))
            occupancy[t:bot,l:r]=1;confidence[t:bot,l:r]=np.maximum(confidence[t:bot,l:r],b[4])
            cue=float(b[4]*(1-math.sqrt((b[2]-b[0])*(b[3]-b[1])/(W*H))))
            for y in range(t//CTU,min(NY,(bot+CTU-1)//CTU)):
                for x in range(l//CTU,min(NX,(r+CTU-1)//CTU)):scale[y,x]=max(scale[y,x],cue)
        qp=np.asarray(event['qp'],np.float32).reshape(NY,NX)
        activity=np.asarray(event['activity'],np.float32).reshape(NY,NX)
        if (activity<0).any() or not np.isfinite(activity).all():raise ValueError('Invalid live QPA activity')
        activity=np.clip(np.log((activity+1e-6)/(np.median(activity)+1e-6)),-4,4)/4
        y=self.raw[p,:W*H].reshape(H,W).astype(np.float32)
        difference=np.abs(y-self.raw[p-1,:W*H].reshape(H,W))/219 if p else np.zeros_like(y)
        x=np.stack([pool(occupancy),pool(confidence),scale,qp/63,activity,pool(difference),self.last_action/2,areas()/CTU**2]).astype(np.float32)
        onehot=[int(event['slice_type']==v) for v in (2,1,0)]
        g=np.asarray([math.log(self.rate/120),self.k/self.frames,p/max(self.frames-1,1),self.used/self.budget,
          np.clip(((self.budget-self.used)/max(self.frames-self.k,1))/(self.budget/self.frames),-4,4)/4,
          event['slice_qp']/63,event['temporal_layer']/7,int(event['reference']),*onehot,
          min(4,math.log1p(self.last_bits/(self.budget/self.frames)))/4,
          (p-self.last_poc)/self.frames if self.k else 0,int(self.k>0),self.price/10],np.float32)
        if x.shape!=(8,NY,NX) or g.shape!=(15,) or not np.isfinite(x).all() or not np.isfinite(g).all():raise ValueError('Bad state tensor')
        return x,g
    def advance(self,event,deltas,outcome):
        if outcome['k']!=self.k or outcome['poc']!=event['poc'] or outcome['bits_total']!=self.used+outcome['bits']:raise ValueError('Outcome accounting mismatch')
        self.last_action=np.asarray(deltas,np.float32).reshape(NY,NX);self.last_bits=outcome['bits'];self.used=outcome['bits_total'];self.last_poc=event['poc'];self.k+=1
    def close(self):self.raw._mmap.close()
