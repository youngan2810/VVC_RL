from dataclasses import asdict
from pathlib import Path
import hashlib,json,time,shutil,gzip
import numpy as np
from .config import FB,W,H,DELTAS,TASK,atomic_json,digest,canonical
from .engine import NativeProcess,verify_decode,audit_trace
from .state import StateBuilder
from .metrics import task,summary
from .reward import reward
from .model import model_hash

def cache_identity(source,profile,rate,encoder,detector_identity,boxes):
    teacher_sha=hashlib.sha256(canonical(boxes).encode()).hexdigest()
    return {'source_pixel_sha256':digest(source),'source_teacher_sha256':teacher_sha,'profile':asdict(profile),'rate':rate,'encoder_sha256':digest(encoder),'detector':detector_identity,'task':TASK}

def run_episode(*,encoder,decoder,source,boxes,detector,detector_identity,profile,rate,folder,model=None,anchor=None,price=1.,deterministic=False,fixed=None,split='train',verify=True):
    if split not in ('train','val'):raise ValueError('This research runner does not select from Test')
    if not verify:raise ValueError('Independent decoder verification cannot be disabled')
    identity=cache_identity(source,profile,rate,encoder,detector_identity,boxes)
    if anchor is not None and anchor['identity']!=identity:raise ValueError('Anchor cache identity mismatch')
    if len(boxes)!=profile.frames:raise ValueError('Teacher POC count mismatch')
    if split!='train' and not deterministic and fixed is None:raise ValueError('Validation policy must be deterministic')
    before=model_hash(model) if model is not None else None
    proc=NativeProcess(encoder,source,folder,profile,rate);state=StateBuilder(source,boxes,profile.frames,rate,price)
    batch={k:[] for k in ('x','g','actions','log_prob','value','reward','done','aux_target','aux_weight')};records=[];by_poc={};occupancies=[];timing={'state':0.,'decision':0.,'encode_handshake':0.,'decoded_detector':0.}
    try:
        for k in range(profile.frames):
            obs=proc.observation();started=time.perf_counter();x,g=state.build(obs);poc=obs['poc'];occupancies.append(x[0].ravel().copy());timing['state']+=time.perf_counter()-started;started=time.perf_counter()
            if fixed is not None:
                delta=np.asarray(fixed(obs),int).reshape(12);actions=delta+2;lp=value=0.
            elif model is None:actions=np.full(12,2,np.int64);delta=actions-2;lp=value=0.
            else:
                out=model.act(x,g,deterministic);actions=out['actions'];delta=actions-2;lp=float(out['log_prob']);value=float(out['value'])
            timing['decision']+=time.perf_counter()-started;started=time.perf_counter();outcome,recon=proc.step(delta);timing['encode_handshake']+=time.perf_counter()-started
            started=time.perf_counter();yuv8=np.minimum(255,(recon.astype(np.uint32)+2)>>2).astype(np.uint8);pred=detector(yuv8);timing['decoded_detector']+=time.perf_counter()-started;metrics=task(boxes[poc],pred)
            baseline=anchor['by_poc'][str(poc)] if anchor is not None else {'score':metrics['score'],'bits':outcome['bits']}
            rew=reward(metrics['score'],baseline['score'],outcome['bits'],baseline['bits'],profile.frames,state.budget,price)
            ref=state.raw[poc,:W*H].astype(np.float64)*4;mse=float(np.mean((recon[:W*H].astype(np.float64)-ref)**2))
            record={k:metrics[k] for k in ('score','legacy_f1','tp','fp','fn')}
            record.update(k=k,poc=poc,bits=outcome['bits'],bits_total=outcome['bits_total'],reward=rew,
              source_boxes=np.asarray(boxes[poc]).tolist(),decoded_boxes=np.asarray(pred).tolist(),requested_delta=delta.tolist(),actions=actions.tolist(),
              state_sha256=hashlib.sha256(x.tobytes()+g.tobytes()).hexdigest(),mse_y_10bit=mse,reconstruction_sha256=hashlib.sha256(recon.tobytes()).hexdigest())
            if model is not None:
                record.update(log_probability=lp,value=value,entropy_mean=float(out['entropy_mean']),logits=out['logits'].tolist())
            records.append(record);by_poc[str(poc)]={'score':metrics['score'],'bits':outcome['bits']}
            for key,val in [('x',x),('g',g),('actions',actions),('log_prob',lp),('value',value),('reward',rew['train']),('done',k==profile.frames-1),('aux_target',metrics['aux_target']),('aux_weight',metrics['aux_weight'])]:batch[key].append(val)
            state.advance(obs,delta,outcome)
            if (k+1)%16==0:print('PICTURE_PROGRESS',split,k+1,'/',profile.frames,'poc',poc,'bits',outcome['bits_total'],flush=True)
        result=proc.finish();result['decoder_native_parity']=verify_decode(folder,decoder,profile.frames)
        result['action_audit']=audit_trace(folder);result.update(summary(records))
        result['timing_ms_per_frame']={k:v*1000/profile.frames for k,v in timing.items()};result['timing_note']='Decision includes Actor for PPO or map construction for controls; encode_handshake includes native computation and IPC, excludes source preprocessing and independent decode'
        cats=np.concatenate([np.asarray(r['actions']) for r in records]);result['requested_action_distribution']={str(d):float(np.mean(cats==d+2)) for d in DELTAS}
        actions=np.asarray([r['requested_delta'] for r in records]);occ=np.asarray(occupancies)
        result['spatial_action_diagnostics']={'fraction_frames_nonuniform':float(np.mean(np.ptp(actions,axis=1)>0)),
          'mean_delta_object_ctus':float(actions[occ>0].mean()) if (occ>0).any() else None,
          'mean_delta_background_ctus':float(actions[occ==0].mean()) if (occ==0).any() else None,
          'per_ctu_distribution':[{str(d):float(np.mean(actions[:,i]==d)) for d in DELTAS} for i in range(12)]}
        result['mean_action_entropy']=float(np.mean([r['entropy_mean'] for r in records])) if model is not None else None
        result.update(identity=identity,split=split,rate=rate,budget=state.budget,by_poc=by_poc,model_sha256=before,deterministic=deterministic,
          anchor_bits=anchor['bits'] if anchor is not None else result['bits'],anchor_quality=anchor['task_retention_f1_v2'] if anchor is not None else result['task_retention_f1_v2'],
          psnr_y_10bit=10*np.log10(1023**2/max(np.mean([r['mse_y_10bit'] for r in records]),1e-30)))
        # Mechanical accounting identity, not a claim of learned usefulness.
        raw=sum(r['reward']['raw'] for r in records)
        expected=0. if anchor is None else result['task_retention_f1_v2']-anchor['task_retention_f1_v2']-price*(result['bits']-anchor['bits'])/state.budget
        if not np.isclose(raw,expected,atol=1e-10):raise ValueError('Reward sum does not equal clip objective')
        result['raw_return']=raw;result['reward_identity_error']=abs(raw-expected)
        if model is not None and model_hash(model)!=before:raise ValueError('Collection/validation changed model weights')
        folder=Path(folder);atomic_json(folder/'frames.json',records);atomic_json(folder/'result.json',result)
        packed={k:np.asarray(v) for k,v in batch.items()};np.savez_compressed(folder/'rollout.npz',**packed)
        files={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file() and 'recon' not in p.relative_to(folder).parts and p.name!='SEAL.json'}
        atomic_json(folder/'SEAL.json',files)
        # Every raw reconstructed picture was independently verified and hashed.
        # Keep streams/POC hashes/traces, release large raw images between episodes.
        shutil.rmtree(folder/'recon')
        print('EPISODE_VERIFIED',split,profile.frames,rate,result['actual_kbps'],result['task_retention_f1_v2'],flush=True)
        return result,packed
    finally:state.close();proc.close()

def verified_result(folder):
    folder=Path(folder);seal=folder/'SEAL.json'
    if not seal.exists():return None
    for n,h in json.loads(seal.read_text()).items():
        p=(folder/n).resolve()
        if not p.is_relative_to(folder.resolve()) or not p.is_file() or digest(p)!=h:raise ValueError('Episode evidence corrupted: '+str(p))
    return json.loads((folder/'result.json').read_text())
