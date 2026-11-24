from dataclasses import asdict
from pathlib import Path
import itertools,json,time,uuid
import numpy as np
import torch
from .config import Profile,PPOConfig,RewardConfig,RATES,atomic_json,digest,LEGACY_STATE
from .locked.data import VideoCatalog
from .model import ActorCritic,model_hash
from .ppo import PPO,policy_state_diagnostics
from .reward import RatePrices
from .validation import validation_summary,rollout_summary
from .collect import collect,load_batches
from .gates import binding,require_gate
from .artifacts import export_run

def mixed_train_schedule(videos,rates=RATES,*,seed=20261120):
    """Deterministic seeded interleaving: one video per round and every video/rate once.

    Each round spans all source videos; adjacent collector pairs cannot share a
    video. Use a private NumPy RNG so Actor/Critic initialization and PPO RNG
    remain exactly as in the blocked-order baseline.
    """
    videos=tuple(videos);rates=tuple(rates)
    if len(videos)<2 or len(videos)%2 or len(set(videos))!=len(videos):
        raise ValueError('Mini schedule requires an even number of distinct videos')
    if not rates or len(set(rates))!=len(rates):
        raise ValueError('Mini schedule requires distinct bitrate levels')
    rng=np.random.default_rng(seed)
    schedule=[]
    for round_index in range(len(rates)):
        for index in rng.permutation(len(videos)):
            i=int(index)
            schedule.append((videos[i],rates[(i+round_index)%len(rates)]))
    return schedule

def save_checkpoint(path,model,ppo,prices,iteration,episodes,bind,best,seed):
    if bind.get('state_version',LEGACY_STATE)!=model.state_version:raise ValueError('Checkpoint state schema differs from model')
    if bind.get('ppo')!=asdict(ppo.config):raise ValueError('Checkpoint PPO config differs from training binding')
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    torch.save({'schema':'vvc-rl-checkpoint/2','model':model.state_dict(),'optimizer':ppo.optimizer.state_dict(),
      'prices':prices.state(),'iteration':iteration,'episodes':episodes,'binding':bind,'best_validation':best,'seed':seed,
      'torch_rng':torch.get_rng_state(),'model_sha256':model_hash(model),'ppo_config':asdict(ppo.config),'state_version':model.state_version},tmp)
    tmp.replace(p);atomic_json(p.with_suffix('.identity.json'),{'sha256':digest(p),'model_sha256':model_hash(model),'iteration':iteration,'binding':bind})

def load_checkpoint(path,model,ppo,expected):
    p=Path(path);identity=json.loads(p.with_suffix('.identity.json').read_text())
    if digest(p)!=identity['sha256']:raise ValueError('Checkpoint digest differs')
    c=torch.load(p,map_location='cpu',weights_only=True)
    if c.get('schema')!='vvc-rl-checkpoint/2':
        raise ValueError('Legacy frozen controls cannot resume a current PPO experiment')
    if (c.get('binding')!=expected
            or c.get('ppo_config')!=asdict(ppo.config) or expected.get('ppo')!=asdict(ppo.config)):
        raise ValueError('Checkpoint PPO/GAE/schedule/reward/source/environment binding differs; resume blocked')
    if c.get('state_version',LEGACY_STATE)!=expected.get('state_version',LEGACY_STATE) or model.state_version!=expected.get('state_version',LEGACY_STATE):raise ValueError('Checkpoint state schema differs; resume blocked')
    model.load_state_dict(c['model']);ppo.optimizer.load_state_dict(c['optimizer'])
    if model_hash(model)!=c['model_sha256']:raise ValueError('Checkpoint weight identity differs')
    torch.set_rng_state(c['torch_rng']);return c

def mini_train(root,output,preflight,toolchain,input_root,tools,*,updates=10,val_every=5,profile=Profile(),seed=20261120,resume=None,authorized=False,hours=10.,reward_config=RewardConfig(),ppo_config=PPOConfig(),train_schedule='mixed',state_version=LEGACY_STATE):
    if not authorized:raise RuntimeError('Training disabled: explicitly pass --start-mini-training after reviewing full preflight')
    root=Path(root);output=Path(output);bind=binding(root,toolchain,profile,2,reward_config,ppo_config,train_schedule,state_version);require_gate(preflight,bind)
    if not 1<=updates or val_every<1:raise ValueError('Invalid update budget')
    output.mkdir(parents=True,exist_ok=True);protocol=output/'TRAIN_BINDING.json'
    if (output/'checkpoints/latest.pt').exists() and not resume:raise ValueError('Existing training checkpoint: explicitly resume instead of restarting')
    invocation=uuid.uuid4().hex[:12]
    if protocol.exists() and json.loads(protocol.read_text())!=bind:raise ValueError('Output contains a different experiment')
    atomic_json(protocol,bind);torch.set_num_threads(1);torch.manual_seed(seed)
    model=ActorCritic(state_version);ppo=PPO(model,ppo_config);prices=RatePrices();iteration=episodes=0;best=0.
    if resume:
        state=load_checkpoint(resume,model,ppo,bind);iteration=state['iteration'];episodes=state['episodes'];seed=state['seed'];best=state['best_validation']
        prices=RatePrices(**state['prices'])
    catalog=VideoCatalog(input_root,mode='mini',seed=20261111);catalog.check_available(('train','val'))
    schedule=mixed_train_schedule(catalog.selected['train'],seed=seed) if train_schedule=='mixed' else list(itertools.product(catalog.selected['train'],RATES))
    if episodes<0 or iteration*2!=episodes or episodes+2*updates>len(schedule):
        raise ValueError('Mini schedule/resume exceeds the unique 40-update train budget')
    deadline=time.monotonic()+hours*3600
    def specs(items,split,snapshot,tag):
        return [dict(input=str(input_root),output=str(output),tools=str(tools),encoder=str(toolchain['encoder']),decoder=str(toolchain['decoder']),
          snapshot=str(snapshot),model_sha256=model_hash(model),profile=asdict(profile),reward_config=asdict(reward_config),reward_scale=ppo_config.reward_scale,state_version=state_version,video=v,split=split,rate=r,price=prices.values[str(r)],
          seed=seed+episodes+i,device=f'cuda:{i}',episode_id=f'{invocation}_{tag}_{i}') for i,(v,r) in enumerate(items)]
    try:
        for _ in range(updates):
            if time.monotonic()>deadline-900:break
            # Collect complete clips with fixed weights/prices; no off-policy reuse.
            snapshot=output/'snapshots'/f'collector_{iteration:06}.pt';snapshot.parent.mkdir(exist_ok=True)
            torch.save({'model':model.state_dict(),'state_version':state_version},snapshot)
            items=schedule[episodes:episodes+2]  # Two distinct videos; no wrap or replay
            rows=collect(specs(items,'train',snapshot,f'train_update{iteration:06}'))
            stats=ppo.update(load_batches(rows),diagnostics_path=output/'updates'/f'{iteration+1:06}_signals.npz');episodes+=2;iteration+=1
            prices.update([r['result'] for r in rows],training=True,config=reward_config)
            atomic_json(output/'updates'/f'{iteration:06}.json',{'iteration':iteration,'episodes':episodes,'stats':stats,'prices':prices.state(),'rollout_diagnostics':rollout_summary(rows),
              'schedule_mode':train_schedule,'ppo_config':asdict(ppo_config),'completed':[{'video':r['video'],'rate':r['result']['rate'],'bits':r['result']['bits'],'budget':r['result']['budget'],'quality':r['result']['task_retention_f1_v2'],'task_utility':r['result']['task_utility'],'utility_version':r['result']['utility_version']} for r in rows]})
            latest=output/'checkpoints/latest.pt';save_checkpoint(latest,model,ppo,prices,iteration,episodes,bind,best,seed)
            atomic_json(output/'PROGRESS.json',{'iteration':iteration,'episodes':episodes,'checkpoint':'checkpoints/latest.pt','best_validation_gain':best,'checkpoint_sha256':digest(latest)})
            if iteration%val_every==0 and time.monotonic()<deadline-900:
                val_snapshot=output/'snapshots'/f'validation_{iteration:06}.pt';torch.save({'model':model.state_dict(),'state_version':state_version},val_snapshot)
                val_items=list(itertools.product(catalog.selected['val'],(50,120,200)));validation=[];before=model_hash(model)
                for j in range(0,len(val_items),2):
                    if time.monotonic()>deadline-900:break
                    validation+=collect(specs(val_items[j:j+2],'val',val_snapshot,f'val_update{iteration:06}_batch{j:03}'))
                if model_hash(model)!=before:raise ValueError('Validation changed model')
                report=validation_summary(validation,val_items,reward_config)
                score=report['macro_delta_quality']
                report.update(model_unchanged=True,rollout_diagnostics=rollout_summary(validation),
                              ppo_config=asdict(ppo.config),train_schedule=train_schedule,
                              policy_head_state_diagnostics=policy_state_diagnostics(model,load_batches(validation)) if validation else {})
                atomic_json(output/'validations'/f'{iteration:06}.json',report)
                if report['eligible_checkpoint'] and score>best:
                    best=score;save_checkpoint(output/'checkpoints/best_validation.pt',model,ppo,prices,iteration,episodes,bind,best,seed)
                save_checkpoint(latest,model,ppo,prices,iteration,episodes,bind,best,seed)
            atomic_json(output/'PROGRESS.json',{'iteration':iteration,'episodes':episodes,'checkpoint':'checkpoints/latest.pt','best_validation_gain':best,'checkpoint_sha256':digest(latest)})
            export_run(output,output.parent/f'VVC_RL_MINI_{output.name}_SAFE.zip')
            print('MINI_UPDATE_COMPLETE',iteration,episodes,stats,flush=True)
        return {'iteration':iteration,'episodes':episodes,'best_validation_gain':best}
    finally:export_run(output,output.parent/f'VVC_RL_MINI_{output.name}_SAFE.zip')
