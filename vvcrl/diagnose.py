"""Validation-only ablations; never optimize on validation or select Test videos."""
from pathlib import Path
from dataclasses import asdict
import json,torch,uuid,itertools,sys
from .config import Profile,RewardConfig,PPOConfig,atomic_json,LEGACY_STATE,D4_STATE,RATES,resolve_ppo_config
from .gates import binding,require_gate
from .model import ActorCritic,model_hash
from .train import load_checkpoint
from .ppo import PPO
from .locked.data import VideoCatalog
from .collect import collect
from .artifacts import export_run
from .validation import validation_summary,rd_validation_summary

def diagnose(root,output,gate,toolchain,input_root,tools,checkpoint,profile=Profile(),reward_config=RewardConfig(),ppo_config=None,train_schedule='mixed',state_version=LEGACY_STATE,mode='auto',max_encodes=0):
    """Explicitly bounded D4 GT-RD controls; never select a model or strategy.

    D4 uses the same four Val videos/five rates/labels/cap as checkpoint
    evaluation. Legacy teacher-only diagnostics remain available ONLY for
    historical states. All controls below are declared before metrics are read.
    """
    ppo_config=resolve_ppo_config(state_version,ppo_config)
    if mode=='auto':mode='rd' if state_version==D4_STATE else 'legacy'
    if mode not in ('legacy','compact','rd'):raise ValueError('Unknown diagnostic mode')
    rd=mode=='rd'
    if (state_version==D4_STATE)!=rd:raise ValueError('D4 requires GT RD diagnostics; legacy/compact modes are historical teacher-only protocols')
    if type(max_encodes) is not int or max_encodes<0:raise ValueError('Invalid diagnostic encode budget')
    catalog=VideoCatalog(input_root,mode='mini',seed=20261111)
    rates=RATES if rd else ((50,120,200) if mode=='compact' else (120,))
    controls=(('ppo','roi_0_bg_p1','roi_m1_bg_p1','roi_m2_bg_p1','roi_0_bg_m1') if rd else
        ('ppo','roi_0_bg_p1','roi_m1_bg_p1','roi_m2_bg_p1') if mode=='compact' else
        ('ppo','random','fixed_m2','fixed_p2','roi_protect','roi_redistribute','roi_0_bg_p1','roi_m1_bg_p1','roi_m2_bg_p1','roi_0_bg_m1'))
    items=list(itertools.product(catalog.selected['val'],rates))
    if rd and len(catalog.selected['val'])!=4:raise ValueError('D4 RD controls require exactly four locked Validation videos')
    planned=len(items)*(len(controls)+1) # one policy/control encode plus one uncached QPA anchor per pair
    if (rd or max_encodes) and max_encodes<planned:
        raise ValueError(f'Diagnostic encode budget too small: need up to {planned} including anchors; explicitly pass max_encodes/--max-diagnostic-encodes before any native collection')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);bind=binding(root,toolchain,profile,reward_config=reward_config,ppo_config=ppo_config,train_schedule=train_schedule,state_version=state_version);require_gate(gate,bind)
    if rd and 'validation_selection' not in bind:raise ValueError('D4 diagnostics require the GT RD gate, never a teacher-only gate')
    torch.set_num_threads(1);model=ActorCritic(state_version);state=load_checkpoint(checkpoint,model,PPO(model,ppo_config),bind);before=model_hash(model)
    rows={};token=uuid.uuid4().hex[:8];summaries={}
    try:
        from .data import validation_ground_truth
        val_gt=validation_ground_truth(input_root,profile.frames) if rd else {}
        if rd and set(val_gt)!=set(catalog.selected['val']):raise ValueError('Incomplete Validation GT for RD controls')
        atomic_json(output/'DIAGNOSTIC_PLAN.json',{'mode':mode,'rates':list(rates),'controls':list(controls),
            'pairs':items,'maximum_uncached_encodes':planned,'authorized_encode_budget':max_encodes,
            'checkpoint_selection_performed':False,'fixed_control_selection_performed':False,'test_used':False})
        for control in controls:
            results=[]
            for start in range(0,len(items),2):
                specs=[dict(input=str(input_root),output=str(output),tools=str(tools),encoder=str(toolchain['encoder']),decoder=str(toolchain['decoder']),
                    snapshot=str(checkpoint),model_sha256=before,profile=asdict(profile),reward_config=asdict(reward_config),reward_scale=ppo_config.reward_scale,state_version=state_version,video=v,split='val',rate=rate,price=state['prices']['values'][str(rate)],
                    seed=20261120+i+start,device=f'cuda:{i}',episode_id=f'{token}_{control}_{i+start}',control=control,
                    validation_gt=val_gt.get(v)) for i,(v,rate) in enumerate(items[start:start+2])]
                results+=collect(specs)
            rows[control]=[{'video':r['video'],**r['result']} for r in results]
            summaries[control]=(rd_validation_summary if rd else validation_summary)(results,items,reward_config)
            if rd:
                report=summaries[control]
                if not report['complete']:raise ValueError('Incomplete RD control grid; do not report a complete comparison')
                report['rd_condition_against_qpa']=report['eligible_checkpoint']
                report.update(eligible_checkpoint=False,eligible_utility_checkpoint=False,diagnostic_only=True,
                              evaluated_method=control,anchor_method='qpa')
            # Save each complete control before proceeding. Never present a
            # partial grid as a complete evaluation.
            atomic_json(output/'POLICY_CONTROLS_PARTIAL.json',{'status':'PARTIAL','mode':mode,'rates':rates,'completed_controls':list(rows),'summaries':summaries})
        if model_hash(model)!=before:raise ValueError('Diagnostics modified policy')
        report={'schema':'vvc-rl-policy-controls/2' if rd else 'vvc-rl-policy-controls/1',
          'binding':bind,'status':'VALIDATION_DIAGNOSTICS_COMPLETE','test_used':False,'checkpoint_model_sha256':before,'results':rows,'mode':mode,'rates':rates,'summaries':summaries,
          'checkpoint_selection_performed':False,'fixed_control_selection_performed':False,
          'interpretation':'Compare PPO to its paired no-op anchor, fixed -2/+2, random, historical semantic heuristics and the four native Validation controls using actual bits. Nonzero action or a changed bitstream alone is not useful learning. Rate-specific winners are development-set descriptions, not independent Test performance or BD-rate.'}
        if rd:
            from .metrics import bd_metrics
            comparisons={}
            for control in controls[1:]:
                comparisons[control]={
                    key:bd_metrics([(p['ppo']['actual_kbps'],p['ppo'][metric]) for p in summaries[control]['rd_curves']],
                                   [(p['ppo']['actual_kbps'],p['ppo'][metric]) for p in summaries['ppo']['rd_curves']])
                    for metric,key in (('ap50','bd_ap50'),('ap50_95','bd_ap50_95'))}
                comparisons[control].update(reference_method=control,candidate_method='ppo',
                    ppo_caps_pass=summaries['ppo']['feasible'],control_caps_pass=summaries[control]['feasible'])
            report.update(ppo_vs_fixed=comparisons,adaptation_proven=False,
                interpretation='Frozen PPO and preregistered fixed controls on the same four Val videos/five rates, real GT AP and original QPA cap. Negative PPO-vs-control BD-rate favors PPO only on the common domain; inspect BOTH cap flags. No control/model selection, Test access, training or generalization claim.')
        else:report['development_rate_conditioned_fixed']=rate_conditioned_fixed(rows,rates)
        atomic_json(output/'POLICY_CONTROLS.json',report);return report
    except BaseException as e:
        atomic_json(output/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'training_performed':False})
        raise
    finally:
        name=f'VVC_RL_POLICY_CONTROLS_FAILED_{token}_KAGGLE_SAFE.zip' if sys.exc_info()[0] is not None else 'VVC_RL_POLICY_CONTROLS_KAGGLE_SAFE.zip'
        export_run(output,output.parent/name)


def rate_conditioned_fixed(rows,rates):
    """Descriptive best feasible fixed control per rate on development only.
    This selection must be frozen before independent test, never used as a
    per-clip deployable oracle. No-op is included when its own cap permits it.
    """
    result={}
    fixed=[k for k in rows if k.startswith('roi_') and '_bg_' in k]
    for rate in rates:
        candidates=[]
        for name in fixed:
            rr=[r for r in rows[name] if r['rate']==rate]
            if rr and all(r['clip_objective']['within_cap'] for r in rr):
                candidates.append((sum(r['task_retention_f1_v2']-r['anchor_quality'] for r in rr)/len(rr),name))
        rr=[r for r in rows.get('ppo',[]) if r['rate']==rate]
        if rr and all(r['anchor_bits']<=r['clip_objective']['budget_cap_bits'] for r in rr):candidates.append((0.,'noop'))
        result[str(rate)]=dict(gain=None,control=None) if not candidates else dict(zip(('gain','control'),max(candidates,key=lambda t:(t[0],t[1]))))
    return result
