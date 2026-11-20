import hashlib,io
import torch
from torch import nn
from torch.distributions import Categorical
from .config import NY,NX,N

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial=nn.Sequential(nn.Conv2d(8,32,3,padding=1),nn.GroupNorm(4,32),nn.Tanh(),nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.Tanh())
        self.global_net=nn.Sequential(nn.Linear(15,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.actor=nn.Conv2d(64,5,1)
        self.critic=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        self.auxiliary=nn.Sequential(nn.Conv2d(69,32,1),nn.Tanh(),nn.Conv2d(32,1,1))
        nn.init.orthogonal_(self.actor.weight,gain=.01);nn.init.zeros_(self.actor.bias)
        with torch.no_grad():self.actor.bias[2]=.15
    def features(self,x,g):
        z=self.spatial(x);global_z=self.global_net(g);joined=torch.cat([z,global_z[:,:,None,None].expand(-1,-1,NY,NX)],1)
        weights=x[:,7:8].clamp_min(1e-8);pooled=(z*weights).sum((2,3))/weights.sum((2,3))
        return joined,torch.cat([pooled,global_z],1)
    def evaluate(self,x,g,actions=None,deterministic=False):
        z,pool=self.features(x,g);logits=self.actor(z).permute(0,2,3,1).reshape(-1,N,5);dist=Categorical(logits=logits)
        if actions is None:actions=logits.argmax(-1) if deterministic else dist.sample()
        if actions.dtype!=torch.long or actions.shape!=logits.shape[:2] or ((actions<0)|(actions>4)).any():raise ValueError('Invalid categorical actions')
        logp=dist.log_prob(actions).sum(-1);entropy=dist.entropy()
        onehot=nn.functional.one_hot(actions,5).reshape(-1,NY,NX,5).permute(0,3,1,2).to(z.dtype)
        aux=self.auxiliary(torch.cat([z,onehot],1)).squeeze(1)
        return {'actions':actions,'log_prob':logp,'value':self.critic(pool).squeeze(-1),
          'entropy_joint':entropy.sum(-1),'entropy_mean':entropy.mean(-1),'logits':logits,'aux_logits':aux}
    def act(self,x,g,deterministic=False):
        with torch.inference_mode():
            result=self.evaluate(torch.as_tensor(x).unsqueeze(0),torch.as_tensor(g).unsqueeze(0),deterministic=deterministic)
        return {k:v[0].cpu().numpy() for k,v in result.items()}

def model_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
