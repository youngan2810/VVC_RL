from dataclasses import asdict
from pathlib import Path
import itertools,json,time,uuid
import numpy as np
import torch
from .config import Profile,PPOConfig,RATES,atomic_json,digest
from .locked.data import VideoCatalog
from .model import ActorCritic,model_hash
from .ppo import PPO
from .reward import RatePrices
from .collect import collect,load_batches
from .gates import binding,require_gate
from .artifacts import export_run

def save_checkpoint(path,model,ppo,prices,iteration,episodes,bind,best,seed):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    torch.save({'schema':'vvc-rl-checkpoint/1','model':model.state_dict(),'optimizer':ppo.optimizer.state_dict(),
      'prices':prices.state(),'iteration':iteration,'episodes':episodes,'binding':bind,'best_validation':best,'seed':seed,
      'torch_rng':torch.get_rng_state(),'model_sha256':model_hash(model)},tmp)
    tmp.replace(p);atomic_json(p.with_suffix('.identity.json'),{'sha256':digest(p),'model_sha256':model_hash(model),'iteration':iteration,'binding':bind})

def load_checkpoint(path,model,ppo,expected):
    p=Path(path);identity=json.loads(p.with_suffix('.identity.json').read_text())
    if digest(p)!=identity['sha256']:raise ValueError('Checkpoint digest differs')
    c=torch.load(p,map_location='cpu',weights_only=True)
    if c['schema']!='vvc-rl-checkpoint/1' or c['binding']!=expected:raise ValueError('Checkpoint source/profile/environment differs')
    model.load_state_dict(c['model']);ppo.optimizer.load_state_dict(c['optimizer'])
    if model_hash(model)!=c['model_sha256']:raise ValueError('Checkpoint weight identity differs')
    torch.set_rng_state(c['torch_rng']);return c

def mini_train(root,output,preflight,toolchain,input_root,tools,*,updates=10,val_every=5,profile=Profile(),seed=20261120,resume=None,authorized=False,hours=10.):
    if not authorized:raise RuntimeError('Training disabled: explicitly pass --start-mini-training after reviewing full preflight')
    root=Path(root);output=Path(output);bind=binding(root,toolchain,profile,2);require_gate(preflight,bind)
    if not 1<=updates or val_every<1:raise ValueError('Invalid update budget')
    output.mkdir(parents=True,exist_ok=True);protocol=output/'TRAIN_BINDING.json'
    if (output/'checkpoints/latest.pt').exists() and not resume:raise ValueError('Existing training checkpoint: explicitly resume instead of restarting')
    invocation=uuid.uuid4().hex[:12]
    if protocol.exists() and json.loads(protocol.read_text())!=bind:raise ValueError('Output contains a different experiment')
    atomic_json(protocol,bind);torch.set_num_threads(1);torch.manual_seed(seed)
    model=ActorCritic();ppo=PPO(model);prices=RatePrices();iteration=episodes=0;best=0.
    if resume:
        state=load_checkpoint(resume,model,ppo,bind);iteration=state['iteration'];episodes=state['episodes'];seed=state['seed'];best=state['best_validation']
        prices=RatePrices(**state['prices'])
    catalog=VideoCatalog(input_root,mode='mini',seed=20261111);catalog.check_available(('train','val'))
    schedule=list(itertools.product(catalog.selected['train'],RATES));deadline=time.monotonic()+hours*3600
    def specs(items,split,snapshot,tag):
        return [dict(input=str(input_root),output=str(output),tools=str(tools),encoder=str(toolchain['encoder']),decoder=str(toolchain['decoder']),
          snapshot=str(snapshot),model_sha256=model_hash(model),profile=asdict(profile),video=v,split=split,rate=r,price=prices.values[str(r)],
          seed=seed+episodes+i,device=f'cuda:{i}',episode_id=f'{invocation}_{tag}_{i}') for i,(v,r) in enumerate(items)]
    try:
        for _ in range(updates):
            if time.monotonic()>deadline-900:break
            # Collect complete clips with fixed weights/prices; no off-policy reuse.
            snapshot=output/'snapshots'/f'collector_{iteration:06}.pt';snapshot.parent.mkdir(exist_ok=True)
            torch.save({'model':model.state_dict()},snapshot)
            items=[schedule[(episodes+i)%len(schedule)] for i in range(2)]
            rows=collect(specs(items,'train',snapshot,f'train_update{iteration:06}'))
            stats=ppo.update(load_batches(rows));episodes+=2;iteration+=1
            prices.update([r['result'] for r in rows],training=True)
            atomic_json(output/'updates'/f'{iteration:06}.json',{'iteration':iteration,'episodes':episodes,'stats':stats,'prices':prices.state(),
              'completed':[{'video':r['video'],'rate':r['result']['rate'],'bits':r['result']['bits'],'budget':r['result']['budget'],'quality':r['result']['task_retention_f1_v2']} for r in rows]})
            latest=output/'checkpoints/latest.pt';save_checkpoint(latest,model,ppo,prices,iteration,episodes,bind,best,seed)
            atomic_json(output/'PROGRESS.json',{'iteration':iteration,'episodes':episodes,'checkpoint':'checkpoints/latest.pt','best_validation_gain':best,'checkpoint_sha256':digest(latest)})
            if iteration%val_every==0 and time.monotonic()<deadline-900:
                val_snapshot=output/'snapshots'/f'validation_{iteration:06}.pt';torch.save({'model':model.state_dict()},val_snapshot)
                val_items=list(itertools.product(catalog.selected['val'],(50,120,200)));validation=[];before=model_hash(model)
                for j in range(0,len(val_items),2):
                    if time.monotonic()>deadline-900:break
                    validation+=collect(specs(val_items[j:j+2],'val',val_snapshot,f'val_update{iteration:06}_batch{j:03}'))
                if model_hash(model)!=before:raise ValueError('Validation changed model')
                complete=len(validation)==len(val_items);quality=[];feasible=True;detail=[]
                for r in validation:
                    result=r['result'];anchor_quality=result['anchor_quality']
                    anchor_bits=result['anchor_bits'];no_extra=result['bits']<=anchor_bits*1.02;budget_ok=result['bits']<=result['budget']*1.02
                    delta=result['task_retention_f1_v2']-anchor_quality;quality.append(delta);feasible &= no_extra and budget_ok
                    detail.append({'video':r['video'],'rate':result['rate'],'delta_quality':delta,'actual_kbps':result['actual_kbps'],'no_extra_bits':no_extra,'budget_ok':budget_ok})
                score=float(np.mean(quality)) if quality else None
                atomic_json(output/'validations'/f'{iteration:06}.json',{'complete':complete,'feasible':feasible,'macro_delta_quality':score,'points':detail,'model_unchanged':True})
                if complete and feasible and score is not None and score>best:
                    best=score;save_checkpoint(output/'checkpoints/best_validation.pt',model,ppo,prices,iteration,episodes,bind,best,seed)
                save_checkpoint(latest,model,ppo,prices,iteration,episodes,bind,best,seed)
            atomic_json(output/'PROGRESS.json',{'iteration':iteration,'episodes':episodes,'checkpoint':'checkpoints/latest.pt','best_validation_gain':best,'checkpoint_sha256':digest(latest)})
            export_run(output,output.parent/'VVC_RL_MINI_TRAIN_KAGGLE_SAFE.zip')
            print('MINI_UPDATE_COMPLETE',iteration,episodes,stats,flush=True)
        return {'iteration':iteration,'episodes':episodes,'best_validation_gain':best}
    finally:export_run(output,output.parent/'VVC_RL_MINI_TRAIN_KAGGLE_SAFE.zip')
