# D-series controlled QP research

## Current experiment: D4 with real-label RD checkpoint selection

D35 controls established useful clip-level alternatives in the existing action space. The matched-prefix diagnostic completed 4 baseline replays and 24 single-step interventions on 8 Train states: all prefixes matched and all alternatives remained inside the original cap. Only one state had a tested utility improvement above the declared 0.001 threshold. Immediate quality could improve while final quality deteriorated, and matched-IoU utility could improve while teacher F1 deteriorated. This does not establish a unique implementation bug, a need to replace PPO, or an optimal action duration.

D4 isolates one supported intervention: remove the known budget potential from the neural critic's learning burden. The value decomposition is retained. This revision separately replaces D4 checkpoint selection with real-label Validation RD metrics; native code, Actor, action, state information, PPO and reward formula are unchanged. No Test outcome was used to tune it.

| Variant | State/model convention | Reward |
|---|---|---|
| D0 | encoder-feedback-v1 | task-budget-v3-progressive-soft-iou |
| D1 | d1-decoupled-v1 | task-budget-v3-progressive-soft-iou |
| D1i | d1-isolated-v1 | task-budget-v3-progressive-soft-iou |
| D2 | d2-region-v1 | task-budget-v3-progressive-soft-iou |
| D2-neutral | d2-region-neutral-v1 | task-budget-v3-progressive-soft-iou |
| D3 | d3-online-budget-v1 | task-budget-v4-online-soft-iou |
| **D4** | **d4-budget-value-v1** | **task-budget-v4-online-soft-iou (unchanged)** |

D1 separates Actor and Critic/Aux representations; D1i and later variants also use independent clipping budgets on disjoint Adam parameter groups. D2 adds nonlinear cross-region residual logits; D3 adds causal budget pressure/headroom and online target-linear intermediate caps. All previous identifiers and numerical behavior are retained.

## Value convention and why it is valid

Before coding step t, let E_t=max(0, used_bits_t-online_cap_t). During a clip the price is fixed. Let scale=100 and Phi_t=-scale*price*E_t/target_bits; Phi_T=0 at termination. The existing progressive reward equals task reward plus terminal excess cost plus Phi_(t+1)-Phi_t, including the final anchor-dependent cap correction. Gamma remains 1.

The D4 critic outputs:

```
V_shaped(s_t) = V_residual(s_t) - Phi_t
               = V_residual(s_t) + 100 * price * positive_budget_pressure_t
```

State indices 14 and 18 already provide price/10 and signed pre-action pressure. D4 uses those existing causal features; no future anchor total, current reconstruction, action or final clip outcome enters the offset. The offset has no parameters and receives no gradient. D3/D4 have identical state shapes, weight initialization, RNG and Actor logits at the same seed; the value convention alone changes.

GAE and PPO replay use the full shaped value, exactly as the collector does. The critic MSE is written against returns_shaped-offset. Terminal/done masks suppress the next value and offset, including between concatenated clips. For a matching residual value, GAE on shaped rewards is equivalent to GAE on terminalized rewards for both supported lambda values, up to float32 error. This is a reparameterization, not a claim that lambda=1, a different reward, or a different optimal policy has been proven better. The reward scale is explicitly locked to 100 for D4 at binding, collection and optimizer entry.

Theory: [GAE](https://arxiv.org/abs/1506.02438), [potential shaping and value initialization](https://arxiv.org/abs/1106.5267). These support the estimator/value-convention reasoning, not any VVC performance guarantee.

## Evidence and limits

CPU replay used the first two verified historical D3 Train clips (512 transitions), with identical recorded Actor actions and probabilities. Values were reevaluated under each convention, followed by one optimizer update (24 steps) per model. This is a frozen-trajectory mechanism probe, not fresh D4 native training or validation.

| Probe | D3 | D4 |
|---|---:|---:|
| Neural value-target standard deviation | 1.737173 | 0.209521 |
| Initial neural value RMSE | 1.758101 | 0.222405 |
| Mean value loss in this update | 1.517920 | 0.022457 |
| Actor updated / Critic updated | yes / yes | yes / yes |

Actor log-probability difference before the update was exactly zero; maximum offset error against recorded native budget context was 5.73e-7; GAE equivalence error was 1.61e-6. A smaller neural target is useful conditioning evidence, not evidence of higher F1, mAP, lower BD-rate, or generalization. D4 could still learn a nearly constant argmax or fail to improve detection.

PPO saves budget offset, residual value/returns and Monte Carlo advantage alongside original GAE, reward, TD error and factor probabilities. Every D4 episode also checks its pre-action offset against the actual causal reward-budget context and records the maximum identity error. Monte Carlo is diagnostic only and never replaces the configured GAE. Both shaped and residual explained variance are reported so known-offset prediction is not mistaken for learned task value. Module gradients and state-dependence probes remain available; perturbation probes are not native causal evidence.

### Checkpoint selection (D4, current source only)

Use the same locked four Validation videos at **50/80/120/160/200 kbps**, 256 frames, every five updates. Authenticate `mot_labels.parquet` against the original dataset lock. Label frameIndex maps to normalized POC by multiplying by six, offset zero, using the reviewed display coordinates. Exclude unknown/all-null frames; explicitly annotated frames with other classes but no cars remain negative examples. Ground truth is never passed into Train, state, PPO or reward. Test is never used for checkpoint selection.

The frozen YOLOv5n AP inference uses confidence 0.001, NMS 0.45, maxDets 300, FP32, car only. It runs separately on annotated decoded frames; the state/reward teacher remains confidence 0.25. AP is COCO bbox 101-recall AP50 and AP50:95, pooled across the four videos at each rate, not an average of per-frame AP. Rates are total bitstream bits divided by total clip duration, not requested kbps. The original independent decode, bit accounting, action audit and seals still apply to every episode and QPA anchor.

`best_validation.pt` minimizes pooled **BD-rate AP50:95 vs QPA**. An eligible checkpoint must have complete 20-pair coverage, every pair inside `(1+0.02)*min(target_bits,QPA_bits)`, negative BD-rate AP50:95 and positive BD-AP50:95. Exact BD-rate ties (1e-9 numerical tolerance) prefer larger BD-AP50:95, then the earlier checkpoint. AP50 metrics are supplemental. BD-AP is in percentage points, not relative percent. PCHIP integrates only common domains of the logged Pareto frontiers; dominated points and overlap ranges remain auditable. Undefined overlap or insufficient points are N/A, never zero or a teacher-F1 fallback.

Reports retain all five pooled QPA/PPO RD points, both BD metrics, all per-pair cap decisions and teacher F1/utility diagnostics. If nothing qualifies, only latest and weight snapshots are available; no fabricated best or secondary utility checkpoint is produced for this D4 protocol. Progress uses `best_validation_rd`, not the historical `best_validation_gain`. Resume uses latest and the identical bound source, GT, selection protocol and metric engine versions. Historical variants retain their explicitly labelled teacher-F1 selection.

This aligns model selection with the research endpoint; it does not prove D4 learns a useful allocation or guarantees a -10% BD-rate/+10% BD-mAP gain. The reward remains the existing dense teacher matched-IoU/budget surrogate. Any surrogate/GT gap must be measured, not hidden. Final frozen Test evaluation compares VVC original, VVC+RL and, when compatible artifacts are available, H264+RL; the paper comparator never selects the VVC policy.

## Kaggle workflow and compatibility

1. [Stage 1: full preflight](../notebooks/VVC_RL_D_Series_01_Preflight_Kaggle.ipynb). Input: Driving Video with Object Tracking, including locked MOV videos **and `mot_labels.parquet`**. Exports a new source-bound gate and exact native toolchain bundle.
2. [Stage 2: fresh 10 updates](../notebooks/VVC_RL_D_Series_02_Train10_Kaggle.ipynb). Inputs: the same MOV/labels dataset + NEW Stage 1 gate + NEW Stage 1 toolchain. Review learning and validation before continuing.
3. [Stage 3: resume to 40](../notebooks/VVC_RL_D_Series_03_Resume30_Kaggle.ipynb). Also attach completed D4 Stage 2 SAFE. Resume latest only; never resume best_validation as a training continuation.

All launchers pin the same immutable functional commit and fingerprint and default to D4. Changing VARIANT to D3 provides a same-source control but requires its own fresh preflight and training. Do not reuse D3/B/Combined checkpoints or gates for D4: weight shapes are not a semantic compatibility test. Start from fresh weights. The earlier D4 teacher-F1 preflight is also incompatible with this source-bound RD selection revision; rerun Stage 1. An interrupted same-version, same-source stage can resume from its non-FAILED SAFE via PARTIAL_STAGE_INPUT. Completed exports are preserved on failure.

The mini protocol stays 16 Train / 4 Validation, 256 frames, all 5 Train rates 50/80/120/160/200, seed 20261120 and the mixed 40-update/80-clip schedule. D4 Validation now uses four videos at all five rates every five updates; 20 anchor/policy pairs rather than 12. A preflight checks all four GT coverages and the real AP path on one Val clip, not the full expensive RD matrix. It cannot establish a useful learned policy. No Test selection. PPO LR 1e-4, 3 epochs, gamma 1, lambda .95, auxiliary .1 and entropy .01 remain unchanged. Partition/MTS, action duration and joint-action redesign are deferred until there is evidence for a separate intervention.

SAFE can be an intact ZIP or Kaggle-expanded payload.bin plus its hash sidecar; toolchain can be ZIP or an expanded manifest/executable directory. Do not expand the inner payload.bin. The legacy B/V2 launchers remain pinned historical controls. The all-in-one D-series notebook is retained, defaulting to D4 with training disabled.
