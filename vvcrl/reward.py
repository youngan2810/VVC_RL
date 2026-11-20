import numpy as np
from .config import RATES

def reward(score,anchor_score,bits,anchor_bits,frames,budget,price,scale=100.):
    if frames<=0 or budget<=0 or bits<=0 or anchor_bits<=0 or price<0:raise ValueError('Invalid reward accounting')
    quality=(score-anchor_score)/frames;bit_cost=(bits-anchor_bits)/budget
    raw=quality-price*bit_cost
    return {'task_component':quality,'bit_component':-price*bit_cost,'raw':raw,'train':scale*raw}

class RatePrices:
    def __init__(self,values=None,eta=.05,upper=10):
        self.values={str(r):1. for r in RATES} if values is None else {str(k):float(v) for k,v in values.items()}
        self.eta,self.upper=eta,upper
        if set(self.values)!=set(map(str,RATES)) or not all(np.isfinite(v) and 0<=v<=upper for v in self.values.values()) or eta<=0 or upper<=0:raise ValueError('Invalid rate price state')
    def update(self,completed,training=True):
        if not training:raise ValueError('Never adapt prices on validation/Test')
        for r in RATES:
            rows=[x for x in completed if x['rate']==r]
            if rows:self.values[str(r)]=float(np.clip(self.values[str(r)]+self.eta*np.mean([x['bits']/x['budget']-1 for x in rows]),0,self.upper))
    def state(self):return {'values':self.values,'eta':self.eta,'upper':self.upper}
