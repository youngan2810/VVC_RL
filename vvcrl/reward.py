"""Versioned clip objectives. V2 never rewards spare bitrate below the cap."""
import numpy as np
from .config import RATES,RewardConfig

def bitrate_cap(budget,anchor_bits,config=RewardConfig()):
    config.validate()
    if not np.isfinite([budget,anchor_bits]).all() or min(budget,anchor_bits)<=0:raise ValueError('Invalid bitrate cap')
    return (1+config.rate_tolerance)*min(budget,anchor_bits)

def clip_objective(score,anchor_score,bits,anchor_bits,budget,price,config=RewardConfig()):
    config.validate()
    if not np.isfinite([score,anchor_score,bits,anchor_bits,budget,price]).all() or min(bits,anchor_bits,budget)<=0 or price<0 or not 0<=score<=1 or not 0<=anchor_score<=1:
        raise ValueError('Invalid clip objective accounting')
    quality=score-anchor_score;cap=bitrate_cap(budget,anchor_bits,config)
    bit_component=-price*((bits-anchor_bits)/budget if config.version=='paired-linear-v1' else max(0.,(bits-cap)/budget))
    return {'task_component':quality,'bit_component':bit_component,'raw':quality+bit_component,
        'budget_cap_bits':cap,'overshoot_fraction':max(0.,(bits-cap)/budget),'within_cap':bool(bits<=cap)}

def reward(score,anchor_score,bits,anchor_bits,frames,budget,price,scale=100.,*,config=RewardConfig(),terminal=False,total_bits=None,anchor_total_bits=None):
    config.validate()
    if not np.isfinite([score,anchor_score,bits,anchor_bits,frames,budget,price,scale]).all() or frames<=0 or int(frames)!=frames or min(budget,bits,anchor_bits,scale)<=0 or price<0 or not 0<=score<=1 or not 0<=anchor_score<=1:
        raise ValueError('Invalid reward accounting')
    quality=(score-anchor_score)/frames
    if config.version=='paired-linear-v1':bit_component=-price*(bits-anchor_bits)/budget
    elif terminal:
        if total_bits is None or anchor_total_bits is None:raise ValueError('Terminal V2 reward requires complete clip bit counts')
        if not np.isfinite([total_bits,anchor_total_bits]).all() or total_bits<bits or anchor_total_bits<anchor_bits:raise ValueError('Invalid terminal clip totals')
        cap=bitrate_cap(budget,anchor_total_bits,config)
        bit_component=-price*max(0.,(total_bits-cap)/budget)
    else:bit_component=0.
    raw=quality+bit_component
    return {'task_component':quality,'bit_component':bit_component,'raw':raw,'train':scale*raw}

class RatePrices:
    def __init__(self,values=None,eta=.05,upper=10):
        self.values={str(r):1. for r in RATES} if values is None else {str(k):float(v) for k,v in values.items()}
        self.eta,self.upper=eta,upper
        if set(self.values)!=set(map(str,RATES)) or not all(np.isfinite(v) and 0<=v<=upper for v in self.values.values()) or not np.isfinite([eta,upper]).all() or eta<=0 or upper<=0:raise ValueError('Invalid rate price state')
    def update(self,completed,training=True,config=RewardConfig()):
        if not training:raise ValueError('Never adapt prices on validation/Test')
        config.validate()
        deviations={str(r):[] for r in RATES}
        for x in completed:
            if x['rate'] not in RATES or not np.isfinite([x['bits'],x['budget']]).all() or min(x['bits'],x['budget'])<=0:raise ValueError('Invalid completed training row')
            cap=x['budget'] if config.version=='paired-linear-v1' else bitrate_cap(x['budget'],x['anchor_bits'],config)
            deviations[str(x['rate'])].append((x['bits']-cap)/x['budget'])
        for r,rows in deviations.items():
            if rows:self.values[r]=float(np.clip(self.values[r]+self.eta*np.mean(rows),0,self.upper))
    def state(self):return {'values':dict(self.values),'eta':self.eta,'upper':self.upper}
