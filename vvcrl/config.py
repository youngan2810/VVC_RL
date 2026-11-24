from dataclasses import dataclass, asdict
import hashlib,json,math
from pathlib import Path

SCHEMA='vvc-rl-qpa-ppo/3'
TASK='task_retention_f1_v2'
UPSTREAM='1a4f83bd9feeb91c6792fb8e0676b1c8f3683d4f'
DELTAS=(-2,-1,0,1,2)
RATES=(50,80,120,160,200)
W,H,FPS,CTU=480,320,30,128
NY,NX,N=3,4,12
FB=W*H*3//2
LEGACY_STATE='legacy-v1'
FEEDBACK_STATE='encoder-feedback-v1'
STATE_SHAPES={LEGACY_STATE:(8,15),FEEDBACK_STATE:(11,18)}

def state_shape(version=LEGACY_STATE):
    if version not in STATE_SHAPES:raise ValueError('Unknown state schema: '+str(version))
    return STATE_SHAPES[version]

@dataclass(frozen=True)
class RewardConfig:
    version:str='task-budget-v2'
    rate_tolerance:float=.02
    def validate(self):
        if self.version not in ('task-budget-v2','paired-linear-v1') or not 0<=self.rate_tolerance<=.1:
            raise ValueError('Unknown reward version or invalid bitrate tolerance')
        return self

@dataclass(frozen=True)
class PPOConfig:
    learning_rate:float=1e-4
    epochs:int=3
    minibatch:int=64
    clip:float=.2
    gamma:float=1.
    gae_lambda:float=.95
    value_coef:float=.5
    entropy_coef:float=.01
    auxiliary_coef:float=.1
    max_grad:float=.5
    target_kl:float=.05
    reward_scale:float=100.
    def validate(self):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError('PPO configuration must be finite')
        if self.gamma!=1.0 or self.gae_lambda not in (0.95,1.0):
            raise ValueError('Locked experiment supports gamma=1 and GAE lambda in {0.95, 1.0}')
        if self.learning_rate<=0 or self.epochs<1 or self.minibatch<1 or not 0<self.clip<1 or self.max_grad<=0 or self.target_kl<=0 or self.reward_scale<=0:
            raise ValueError('Invalid PPO optimizer configuration')
        return self

@dataclass(frozen=True)
class Profile:
    frames:int=256
    preset:str='medium'
    threads:int=2
    ctu:int=128
    width:int=480
    height:int=320
    fps:int=30
    qpa:bool=True
    gop:int=32
    lookahead:int=1
    bitdepth:int=10
    def validate(self):
        if self.frames<32 or self.frames%32 or (self.width,self.height,self.fps,self.ctu,self.gop,self.lookahead,self.bitdepth)!=(480,320,30,128,32,1,10) or not self.qpa or self.threads!=2 or self.preset!='medium':
            raise ValueError('V1 supports only the locked medium/CTU128/QPA/serial-picture profile and full GOP windows')
        return self
    def command(self,encoder,source,stream,rate):
        self.validate()
        if rate not in RATES:raise ValueError('Unknown locked rate')
        return list(map(str,[encoder,'-i',source,'-s','480x320','-r','30','-f',self.frames,'-c','yuv420','--preset',self.preset,
          '--internal-bitdepth','10','--bitrate',rate*1000,'--passes','1','--qpa','1','--intraperiod','32','--refreshtype','cra',
          '--sdr','sdr_709','--threads','2','--mtprofile','0','--additional','GOPSize=32:LookAhead=1:CTUSize=128:MaxCuDQPSubdiv=2:MaxParallelFrames=0','-o',stream]))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)
def fingerprint(root):
    root=Path(root);h=hashlib.sha256()
    for directory in ('vvcrl','native','vendor/vvenc/source','vendor/vvenc/include','configs','scripts','tests'):
        # Path ordering is case-folded on Windows but case-sensitive on POSIX.
        # Sort portable relative path strings, never OS-specific Path objects.
        for p in sorted((root/directory).rglob('*'),key=lambda p:p.relative_to(root).as_posix()):
            if p.is_file() and p.suffix in ('.py','.h','.cpp','.json'):
                h.update(p.relative_to(root).as_posix().encode());h.update(p.read_bytes())
    h.update((root/'vendor/vvenc/CMakeLists.txt').read_bytes())
    return h.hexdigest()
def atomic_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
