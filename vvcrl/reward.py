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

def reward(score,anchor_score,bits,anchor_bits,frames,budget,price,scale=100.,*,config=RewardConfig(),terminal=False,total_bits=None,anchor_total_bits=None,progress=None):
    config.validate()
    if not np.isfinite([score,anchor_score,bits,anchor_bits,frames,budget,price,scale]).all() or frames<=0 or int(frames)!=frames or min(budget,bits,anchor_bits,scale)<=0 or price<0 or not 0<=score<=1 or not 0<=anchor_score<=1:
        raise ValueError('Invalid reward accounting')
    quality=(score-anchor_score)/frames
    if config.progressive:
        if progress is None or total_bits is None:raise ValueError('Progressive reward requires causal budget context')
        previous_cap,cap=progress['previous_cap_bits'],progress['cumulative_cap_bits']
        if not np.isfinite([previous_cap,cap,total_bits]).all() or min(previous_cap,cap)<0 or total_bits<bits:raise ValueError('Invalid progressive budget context')
        if progress['cumulative_bits']!=total_bits or progress['previous_bits']!=total_bits-bits:raise ValueError('Progressive cumulative bits mismatch')
        if terminal and (anchor_total_bits is None or not np.isclose(cap,bitrate_cap(budget,anchor_total_bits,config),rtol=0,atol=1e-8)):
            raise ValueError('Progressive terminal cap differs from clip objective')
        previous_excess=max(0.,total_bits-bits-previous_cap);excess=max(0.,total_bits-cap)
        bit_component=-price*(excess-previous_excess)/budget
    elif progress is not None:raise ValueError('Unexpected progressive context for legacy reward')
    elif config.version=='paired-linear-v1':bit_component=-price*(bits-anchor_bits)/budget
    elif terminal:
        if total_bits is None or anchor_total_bits is None:raise ValueError('Terminal V2 reward requires complete clip bit counts')
        if not np.isfinite([total_bits,anchor_total_bits]).all() or total_bits<bits or anchor_total_bits<anchor_bits:raise ValueError('Invalid terminal clip totals')
        cap=bitrate_cap(budget,anchor_total_bits,config)
        bit_component=-price*max(0.,(total_bits-cap)/budget)
    else:bit_component=0.
    raw=quality+bit_component
    result={'task_component':quality,'bit_component':bit_component,'raw':raw,'train':scale*raw}
    if config.progressive:result.update(progress,previous_excess_bits=previous_excess,excess_bits=excess,price=float(price))
    return result

class ProgressiveBudget:
    """Predetermined coding-order anchor schedule; telescoping clip excess.

    With no anchor (a no-op reference run), use a provisional target schedule
    and correct the final cap against actual bits. Never observe future RL bits.
    """
    def __init__(self,frames,budget,config,anchor=None):
        config.validate()
        if not config.progressive or int(frames)!=frames or frames<=0 or not np.isfinite(budget) or budget<=0:raise ValueError('Invalid progressive schedule')
        self.frames=int(frames);self.budget=float(budget);self.config=config;self.k=0;self.used=0;self.previous_cap=0.;self.order=None
        if anchor is None:
            self.anchor_bits=None;self.fractions=np.arange(1,self.frames+1,dtype=np.float64)/self.frames
        else:
            self.anchor_bits=anchor['bits'];rows=anchor['by_poc'];self.order=tuple(int(p) for p in rows)
            if len(self.order)!=self.frames or set(self.order)!=set(range(self.frames)):raise ValueError('Invalid anchor coding-order schedule')
            bits=np.asarray([row['bits'] for row in rows.values()],np.float64)
            if not np.isfinite(bits).all() or (bits<=0).any() or (bits!=np.floor(bits)).any() or bits.sum()!=self.anchor_bits:raise ValueError('Anchor AU accounting mismatch')
            self.fractions=np.cumsum(bits)/self.anchor_bits
    def step(self,poc,bits,total_bits):
        if self.k>=self.frames or not np.isfinite([bits,total_bits]).all() or bits<=0 or int(bits)!=bits or total_bits!=self.used+bits:raise ValueError('Progressive AU accounting mismatch')
        if self.order is not None and self.order[self.k]!=poc:raise ValueError('Anchor and rollout coding order differ')
        terminal=self.k==self.frames-1
        reference=self.anchor_bits if self.anchor_bits is not None else (total_bits if terminal else self.budget)
        cap=bitrate_cap(self.budget,reference,self.config)*float(self.fractions[self.k])
        context={'budget_step':self.k,'previous_bits':self.used,'cumulative_bits':int(total_bits),'previous_cap_bits':self.previous_cap,'cumulative_cap_bits':cap}
        self.k+=1;self.used=int(total_bits);self.previous_cap=cap
        return context

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
