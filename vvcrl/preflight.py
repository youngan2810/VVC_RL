from dataclasses import asdict
from pathlib import Path
import json,subprocess,sys,time
import numpy as np
import torch
from .config import Profile,PPOConfig,RewardConfig,atomic_json,digest
from .locked.data import VideoCatalog
from .model import ActorCritic,model_hash
from .ppo import PPO
from .collect import collect,load_batches
from .gates import MANDATORY,binding,write_gate
from .native_smoke import run_smoke
from .artifacts import export_run

def full_preflight(root,output,toolchain,input_root,tools,profile=Profile(),workers=2,seed=20261120,reward_config=RewardConfig()):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False);checks={k:False for k in MANDATORY}
    bind=binding(root,toolchain,profile,workers,reward_config);evidence=[]
    try:
        if workers!=2 or not torch.cuda.is_available() or torch.cuda.device_count()<2:raise RuntimeError('Full Kaggle preflight requires two real CUDA collectors; CPU fixture cannot authorize training')
        unit=output/'unit_contracts.log'
        with unit.open('wb') as f:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-p','test_*.py'],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
        log=unit.read_text(errors='replace')
        if 'Ran 0 tests' in log or 'OK' not in log:raise ValueError('Unit discovery did not run contracts')
        checks['unit_contracts']=True
        from .locked.pixels import verify_conversion
        atomic_json(output/'RGB_PROTOCOL.json',verify_conversion());checks['rgb_protocol']=True
        native=run_smoke(toolchain,output/'native_smoke');checks.update(native['checks'])
        torch.set_num_threads(1);torch.manual_seed(seed);model=ActorCritic();ppo=PPO(model)
        snapshot=output/'collector_initial.pt';torch.save({'model':model.state_dict()},snapshot)
        catalog=VideoCatalog(input_root,mode='mini',seed=20261111)
        catalog.check_available(('train','val'))
        def specs(videos,split,snapshot,model,tag):
            return [dict(input=str(input_root),output=str(output),tools=str(tools),encoder=str(toolchain['encoder']),decoder=str(toolchain['decoder']),
              snapshot=str(snapshot),model_sha256=model_hash(model),profile=asdict(profile),reward_config=asdict(reward_config),video=v,split=split,rate=120,price=1.,
              seed=seed+i,device=f'cuda:{i}',episode_id=f'{tag}_{i}') for i,v in enumerate(videos)]
        train=collect(specs(catalog.selected['train'][:2],'train',snapshot,model,'mechanical_train'))
        if {r['device'] for r in train}!={'cuda:0','cuda:1'} or len({r['pid'] for r in train})!=2:raise ValueError('Two-GPU collection contract failed')
        checks['two_gpu_collectors']=True;checks['real_train_rollout']=True
        batch=load_batches(train);before=model_hash(model);stats=ppo.update(batch)
        checks['logprob_value_replay']=stats['replay_logprob_error']<=2e-5 and stats['replay_value_error']<=2e-5
        # Entropy alone can change Actor weights: require a nonzero policy-loss
        # gradient as well. This is still a mechanism check, not useful learning.
        checks['actor_head_updated']=stats['actor_head_change']>0 and stats['policy_head_gradient_norm']>0
        checks['critic_head_updated']=stats['critic_head_change']>0
        checks['reward_identity']=all(r['result']['reward_identity_error']<1e-10 for r in train)
        atomic_json(output/'PPO_MECHANICAL_UPDATE.json',{'before':before,'after':model_hash(model),'stats':stats,'scope':'one train-only update; not evidence of useful policy'})
        after=output/'mechanical_checkpoint.pt';torch.save({'model':model.state_dict(),'optimizer':ppo.optimizer.state_dict(),'config':asdict(PPOConfig())},after)
        restored=ActorCritic();restored.load_state_dict(torch.load(after,map_location='cpu',weights_only=True)['model']);checks['checkpoint_reload']=model_hash(restored)==model_hash(model)
        val_before=model_hash(model)
        val=collect(specs(catalog.selected['val'][:1],'val',after,model,'validation_no_update'))
        checks['validation_no_update']=model_hash(model)==val_before and all(r['model_sha256']==val_before for r in val)
        atomic_json(output/'REAL_DATA_SUMMARY.json',{'train':[r['result'] for r in train],'validation':[r['result'] for r in val],
          'policy_changed_on_validation':False,'learning_quality_claim':False,'test_used':False})
        # Bind every stored proof except bulky source/reconstruction, not merely PASS flags.
        evidence=[p for p in output.rglob('*') if p.is_file() and p.suffix!='.yuv' and 'source_cache' not in p.relative_to(output).parts and 'recon' not in p.relative_to(output).parts]
        report=write_gate(output,bind,checks,evidence)
        if report['status']!='PASS_READY_FOR_MINI':raise RuntimeError('Preflight incomplete; training remains blocked')
        print('FULL_PREFLIGHT_PASS_READY_FOR_MINI — no mini training started',flush=True);return report
    except BaseException as e:
        atomic_json(output/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'training_blocked':True})
        write_gate(output,bind,checks,[p for p in evidence if p.exists()]);raise
    finally:export_run(output)
