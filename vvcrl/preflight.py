from dataclasses import asdict
from pathlib import Path
import json,subprocess,sys,time
import numpy as np
import torch
from .config import Profile,PPOConfig,RewardConfig,atomic_json,digest,LEGACY_STATE,resolve_ppo_config,resolve_reward_config,TRAINING_REWARD_PROTOCOL
from .locked.data import VideoCatalog
from .model import ActorCritic,model_hash
from .ppo import PPO,residual_mechanics_ok
from .collect import collect,load_batches
from .gates import required_checks,binding,write_gate
from .native_smoke import run_smoke
from .artifacts import export_run
from .train import save_checkpoint,load_checkpoint
from .reward import RatePrices

def full_preflight(root,output,toolchain,input_root,tools,profile=Profile(),workers=2,seed=20261120,reward_config=None,ppo_config=None,train_schedule='mixed',state_version=LEGACY_STATE):
    ppo_config=resolve_ppo_config(state_version,ppo_config)
    reward_config=resolve_reward_config(state_version,reward_config)
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    bind=binding(root,toolchain,profile,workers,reward_config,ppo_config,train_schedule,state_version);checks={k:False for k in required_checks(bind)};evidence=[]
    try:
        if 'validation_selection' in bind:
            from .data import validation_ground_truth
            val_gt=validation_ground_truth(input_root,profile.frames)
            atomic_json(output/'VALIDATION_GT.json',val_gt)
        else:val_gt={}
        if reward_config.gt_ap:
            from .data import training_ground_truth
            train_gt=training_ground_truth(input_root,profile.frames)
            atomic_json(output/'TRAINING_GT.json',train_gt)
        else:train_gt={}
        if workers!=2 or not torch.cuda.is_available() or torch.cuda.device_count()<2:raise RuntimeError('Full Kaggle preflight requires two real CUDA collectors; CPU fixture cannot authorize training')
        unit=output/'unit_contracts.log'
        with unit.open('wb') as f:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-p','test_*.py'],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
        log=unit.read_text(errors='replace')
        if 'Ran 0 tests' in log or 'OK' not in log:raise ValueError('Unit discovery did not run contracts')
        checks['unit_contracts']=True
        from .locked.pixels import verify_conversion
        atomic_json(output/'RGB_PROTOCOL.json',verify_conversion());checks['rgb_protocol']=True
        native=run_smoke(toolchain,output/'native_smoke');checks.update(native['checks'])
        torch.set_num_threads(1);torch.manual_seed(seed);model=ActorCritic(state_version);ppo=PPO(model,ppo_config)
        snapshot=output/'collector_initial.pt';torch.save({'model':model.state_dict(),'state_version':state_version},snapshot)
        catalog=VideoCatalog(input_root,mode='mini',seed=20261111)
        catalog.check_available(('train','val'))
        checked={}
        for split in ('train','val'):
            checked[split]=[{'video':v,'sha256':digest(catalog.source(v))} for v in catalog.selected[split]]
        atomic_json(output/'LOCKED_DATASET_VERIFIED.json',{'splits':checked,'test_used':False})
        def specs(videos,split,snapshot,model,tag):
            return [dict(input=str(input_root),output=str(output),tools=str(tools),encoder=str(toolchain['encoder']),decoder=str(toolchain['decoder']),
              snapshot=str(snapshot),model_sha256=model_hash(model),profile=asdict(profile),reward_config=asdict(reward_config),reward_scale=ppo_config.reward_scale,state_version=state_version,video=v,split=split,rate=120,price=1.,
              seed=seed+i,device=f'cuda:{i}',episode_id=f'{tag}_{i}',validation_gt=val_gt.get(v) if split=='val' else None,
              training_gt=train_gt.get(v) if split=='train' else None) for i,v in enumerate(videos)]
        train=collect(specs(catalog.selected['train'][:2],'train',snapshot,model,'mechanical_train'))
        if {r['device'] for r in train}!={'cuda:0','cuda:1'} or len({r['pid'] for r in train})!=2:raise ValueError('Two-GPU collection contract failed')
        checks['two_gpu_collectors']=True;checks['real_train_rollout']=True
        before=model_hash(model);batch=load_batches(train,expected_split='train',state_version=state_version,reward_config=reward_config,reward_scale=ppo_config.reward_scale,expected_model=before);stats=ppo.update(batch,diagnostics_path=output/'MECHANICAL_SIGNALS.npz')
        checks['logprob_value_replay']=stats['replay_logprob_error']<=2e-5 and stats['replay_value_error']<=2e-5
        # Entropy alone can change Actor weights: require a nonzero policy-loss
        # gradient as well. This is still a mechanism check, not useful learning.
        checks['actor_head_updated']=stats['actor_head_change']>0 and stats['policy_head_gradient_norm']>0 and residual_mechanics_ok(model,stats)
        checks['critic_head_updated']=stats['critic_head_change']>0
        checks['reward_identity']=all(r['result']['reward_identity_error']<1e-10 for r in train)
        if reward_config.gt_ap:
            from .metrics import coco_ap
            checks['training_gt_ap']=True
            for row in train:
                for key in ('training_ap','anchor_training_ap'):
                    proof=row['result'][key]
                    checks['training_gt_ap'] &= proof['protocol']==TRAINING_REWARD_PROTOCOL and coco_ap(proof['ground_truth'],proof['predictions'])==proof['metrics'] and proof['metrics']['ap50_95'] is not None
            checks['training_gt_ap']=bool(checks['training_gt_ap'])
        atomic_json(output/'PPO_MECHANICAL_UPDATE.json',{'before':before,'after':model_hash(model),'stats':stats,'ppo_config':asdict(ppo_config),'train_schedule':train_schedule,'scope':'one train-only update; not evidence of useful policy'})
        after=output/'mechanical_checkpoint.pt';prices=RatePrices();prices.update([r['result'] for r in train],config=reward_config)
        rng=torch.get_rng_state().clone()
        save_checkpoint(after,model,ppo,prices,1,len(train),bind,0.,seed)
        restored=ActorCritic(state_version);restored_ppo=PPO(restored,ppo_config)
        restored_state=load_checkpoint(after,restored,restored_ppo,bind)
        checks['checkpoint_reload']=model_hash(restored)==model_hash(model) and torch.equal(torch.get_rng_state(),rng) and restored_state['prices']==prices.state() and restored_state['iteration']==1 and restored_state['episodes']==len(train)
        if not checks['checkpoint_reload']:raise ValueError('Production checkpoint reload failed')
        val_before=model_hash(model)
        val=collect(specs(catalog.selected['val'][:1],'val',after,model,'validation_no_update'))
        load_batches(val,expected_split='val',state_version=state_version,reward_config=reward_config,reward_scale=ppo_config.reward_scale,expected_model=val_before)
        checks['validation_no_update']=model_hash(model)==val_before and all(r['model_sha256']==val_before for r in val) and all(r['result']['reward_identity_error']<1e-10 for r in val)
        if val_gt:
            from .metrics import coco_ap
            from .config import VALIDATION_SELECTION
            import importlib.metadata
            if importlib.metadata.version('pycocotools')!='2.0.11':raise ValueError('COCO AP engine version differs')
            result=val[0]['result'];proof=result['validation_ap'];base=result['anchor_validation_ap']
            metrics=coco_ap(proof['ground_truth'],proof['predictions'])
            anchor_metrics=coco_ap(base['ground_truth'],base['predictions'])
            checks['validation_gt_ap']=(proof['protocol']==VALIDATION_SELECTION and proof['ground_truth']==base['ground_truth']
                and metrics==proof['metrics'] and anchor_metrics==base['metrics']
                and metrics['ap50_95'] is not None and anchor_metrics['ap50_95'] is not None)
        atomic_json(output/'REAL_DATA_SUMMARY.json',{'train':[r['result'] for r in train],'validation':[r['result'] for r in val],
          'policy_changed_on_validation':False,'learning_quality_claim':False,'test_used':False})
        # Bind every stored proof except bulky source/reconstruction, not merely PASS flags.
        evidence=[p for p in output.rglob('*') if p.is_file() and p.suffix!='.yuv' and 'source_cache' not in p.relative_to(output).parts and 'recon' not in p.relative_to(output).parts]
        report=write_gate(output,bind,checks,evidence)
        if report['status']!='PASS_READY_FOR_MINI':raise RuntimeError('Preflight incomplete; training remains blocked')
        print('FULL_PREFLIGHT_PASS_READY_FOR_MINI â€” no mini training started',flush=True);return report
    except BaseException as e:
        atomic_json(output/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'training_blocked':True})
        write_gate(output,bind,checks,[p for p in evidence if p.exists()]);raise
    finally:export_run(output,output.parent/f'VVC_RL_PREFLIGHT_{output.name}_SAFE.zip')
