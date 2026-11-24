import hashlib,io
import torch
from torch import nn
from torch.distributions import Categorical
from .config import NY,NX,N,LEGACY_STATE,state_shape

class ActorCritic(nn.Module):
    def __init__(self,state_version=LEGACY_STATE):
        super().__init__()
        self.state_version=state_version
        spatial_channels,global_channels=state_shape(state_version)
        self.spatial=nn.Sequential(nn.Conv2d(8,32,3,padding=1),nn.GroupNorm(4,32),nn.Tanh(),nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.Tanh())
        self.global_net=nn.Sequential(nn.Linear(15,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        # Source-ROI/background factors coordinate residuals under a frame
        # reward. The native action stays 12 integer CTU QPs.
        self.actor=nn.Conv2d(64,5,1)
        self.critic=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        self.auxiliary=nn.Sequential(nn.Conv2d(69,32,1),nn.Tanh(),nn.Conv2d(32,1,1))
        nn.init.orthogonal_(self.actor.weight,gain=.01);nn.init.zeros_(self.actor.bias)
        with torch.no_grad():self.actor.bias[2]=.15
        # Duplicate the old initialized head into independently trainable region
        # heads without shifting shared/Critic/Aux initialization or CPU RNG.
        original=self.actor
        with torch.random.fork_rng(devices=[]):
            self.actor=nn.Conv2d(64,10,1)
            with torch.no_grad():
                self.actor.weight.copy_(original.weight.repeat(2,1,1,1))
                self.actor.bias.copy_(original.bias.repeat(2))
        # Keep all common parameters and RNG identical to C0 at the same seed.
        # New input weights start at zero, remain trainable, and cannot change
        # initial action preference just because the state dimension changed.
        if state_version!=LEGACY_STATE:
            with torch.random.fork_rng(devices=[]):
                conv=nn.Conv2d(spatial_channels,32,3,padding=1);linear=nn.Linear(global_channels,32)
                with torch.no_grad():
                    conv.weight.zero_();conv.weight[:,:8].copy_(self.spatial[0].weight);conv.bias.copy_(self.spatial[0].bias)
                    linear.weight.zero_();linear.weight[:,:15].copy_(self.global_net[0].weight);linear.bias.copy_(self.global_net[0].bias)
                self.spatial[0]=conv;self.global_net[0]=linear
    def features(self,x,g):
        sc,gc=state_shape(self.state_version)
        if x.ndim!=4 or x.shape[1:]!=(sc,NY,NX) or g.shape!=(len(x),gc):raise ValueError('State schema/model shape mismatch')
        z=self.spatial(x);global_z=self.global_net(g);joined=torch.cat([z,global_z[:,:,None,None].expand(-1,-1,NY,NX)],1)
        weights=x[:,7:8].clamp_min(1e-8);pooled=(z*weights).sum((2,3))/weights.sum((2,3))
        return joined,torch.cat([pooled,global_z],1)
    def evaluate(self,x,g,actions=None,deterministic=False):
        z,pool=self.features(x,g);batch=len(x)
        roi=(x[:,0]>0).reshape(batch,N)
        masks=torch.stack([roi,~roi],1)
        area=x[:,7].reshape(batch,N).clamp_min(1e-8)
        weights=masks.to(z.dtype)*area[:,None,:]
        active=masks.any(-1)
        raw=self.actor(z).reshape(batch,2,5,N)
        factor_logits=(raw*weights[:,:,None,:]).sum(-1)/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        dist=Categorical(logits=factor_logits)
        if actions is None:
            factors=factor_logits.argmax(-1) if deterministic else dist.sample()
            factors=torch.where(active,factors,torch.full_like(factors,2))
            actions=torch.where(roi,factors[:,0,None],factors[:,1,None])
        else:
            if actions.dtype!=torch.long or actions.shape!=(batch,N) or ((actions<0)|(actions>4)).any():raise ValueError('Invalid categorical actions')
            first=masks.to(torch.int64).argmax(-1)
            factors=actions.gather(1,first)
            factors=torch.where(active,factors,torch.full_like(factors,2))
            expected=torch.where(roi,factors[:,0,None],factors[:,1,None])
            if not torch.equal(actions,expected):raise ValueError('Actions are not coherent ROI/background factors; never replay independent-CTU rollouts')
        # Inactive factors are marginalized, not counted once per CTU. CTU maps
        # uniquely encode the active factors; no rounding/projection enters PPO.
        logp=(dist.log_prob(factors)*active).sum(-1)
        entropy=dist.entropy()*active
        logits=torch.where(roi[:,:,None],factor_logits[:,0,None,:],factor_logits[:,1,None,:])
        onehot=nn.functional.one_hot(actions,5).reshape(-1,NY,NX,5).permute(0,3,1,2).to(z.dtype)
        aux=self.auxiliary(torch.cat([z,onehot],1)).squeeze(1)
        return {'actions':actions,'log_prob':logp,'value':self.critic(pool).squeeze(-1),
          'entropy_joint':entropy.sum(-1),'entropy_mean':entropy.sum(-1)/active.sum(-1).clamp_min(1),
          'logits':logits,'factor_logits':factor_logits,'factor_actions':factors,'active_factors':active,'aux_logits':aux}
    def act(self,x,g,deterministic=False):
        with torch.inference_mode():
            result=self.evaluate(torch.as_tensor(x).unsqueeze(0),torch.as_tensor(g).unsqueeze(0),deterministic=deterministic)
        return {k:v[0].cpu().numpy() for k,v in result.items()}

def state_dict_hash(weights):
    h=hashlib.sha256()
    for k,v in sorted(weights.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def model_hash(model):return state_dict_hash(model.state_dict())
