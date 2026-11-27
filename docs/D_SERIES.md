# D-series controlled QP research

## Current experiment: D4 budget-value decomposition

D35 controls established useful clip-level alternatives in the existing action space. The matched-prefix diagnostic completed 4 baseline replays and 24 single-step interventions on 8 Train states: all prefixes matched and all alternatives remained inside the original cap. Only one state had a tested utility improvement above the declared 0.001 threshold. Immediate quality could improve while final quality deteriorated, and matched-IoU utility could improve while teacher F1 deteriorated. This does not establish a unique implementation bug, a need to replace PPO, or an optimal action duration.

D4 isolates one supported intervention: remove the known budget potential from the neural critic's learning burden. No native, Actor, action, state information, reward formula, checkpoint selection metric or bitrate-cap change is bundled into this experiment. No Test outcome was used to tune it.

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

Validation additionally counts utility-gain/F1-loss pairs. Primary best_validation.pt still requires complete validation, positive macro teacher F1 and every pair inside cap=(1+tolerance)*min(target_bits,QPA_bits), with tolerance 0.02. best_utility.pt is secondary diagnostic evidence. Teacher retention F1/IoU is not ground-truth AP. Independent GT mAP/BD-rate evaluation is still required after training; no success claim follows from PPO mechanics or utility alone.

## Kaggle workflow and compatibility

1. [Stage 1: full preflight](../notebooks/VVC_RL_D_Series_01_Preflight_Kaggle.ipynb). Input: locked BDD100K MOV videos. Exports a new source-bound gate and exact native toolchain bundle.
2. [Stage 2: fresh 10 updates](../notebooks/VVC_RL_D_Series_02_Train10_Kaggle.ipynb). Inputs: MOV + Stage 1 gate + Stage 1 toolchain. Review learning and validation before continuing.
3. [Stage 3: resume to 40](../notebooks/VVC_RL_D_Series_03_Resume30_Kaggle.ipynb). Also attach completed D4 Stage 2 SAFE. Resume latest only; never resume best_validation as a training continuation.

All launchers pin the same immutable functional commit and fingerprint and default to D4. Changing VARIANT to D3 provides a same-source control but requires its own fresh preflight and training. Do not reuse D3/B/Combined checkpoints or gates for D4: weight shapes are not a semantic compatibility test. Start from fresh weights. An interrupted same-version, same-source stage can resume from its non-FAILED SAFE via PARTIAL_STAGE_INPUT. Completed exports are preserved on failure.

The mini protocol stays 16 Train / 4 Validation, 256 frames, all 5 Train rates 50/80/120/160/200, seed 20261120 and the mixed 40-update/80-clip schedule. Validation remains four videos at 50/120/200 every five updates. No Test selection. PPO LR 1e-4, 3 epochs, gamma 1, lambda .95, auxiliary .1 and entropy .01 remain unchanged. Partition/MTS, action duration and joint-action redesign are deferred until there is evidence for a separate intervention.

SAFE can be an intact ZIP or Kaggle-expanded payload.bin plus its hash sidecar; toolchain can be ZIP or an expanded manifest/executable directory. Do not expand the inner payload.bin. The legacy B/V2 launchers remain pinned historical controls. The all-in-one D-series notebook is retained, defaulting to D4 with training disabled.
