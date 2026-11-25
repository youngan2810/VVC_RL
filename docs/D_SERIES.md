# D-series controlled research implementation

Base: Combined d135a8a4c57b0d96c356f36647109125c08b1932. Fixed policy is observed, not a proven architectural failure. Native counterfactual is incomplete. No useful-learning claim or Test10 tuning.

| Variant | --state-version | --reward-version |
|---|---|---|
| D0 | encoder-feedback-v1 | task-budget-v3-progressive-soft-iou |
| D1 | d1-decoupled-v1 | task-budget-v3-progressive-soft-iou |
| D1i | d1-isolated-v1 | task-budget-v3-progressive-soft-iou |
| D2 | d2-region-v1 | task-budget-v3-progressive-soft-iou |
| D2 neutral initialization control | d2-region-neutral-v1 | task-budget-v3-progressive-soft-iou |
| D3 | d3-online-budget-v1 | task-budget-v4-online-soft-iou |

D1 deep-copies the original CNN/MLP for value and auxiliary. Same initialization and joint gradient clipping; D1i isolates clipping budgets, using two disjoint Adam parameter groups (per-parameter moments, not two independent optimizer objects). D2 adds a zero-output-initialized nonlinear residual to the original regional logits; the context includes both area-weighted regional embeddings, raw features, global features, area fractions and presence masks. D2-neutral removes the original no-op bias only. D3 adds two causal budget features (signed pressure and next-step headroom) and target-linear intermediate cap with anchor-dependent terminal correction. No future anchor information enters policy. Terminal correction preserves undiscounted objective, but may affect GAE and remains partially observed. Existing reward/architecture identifiers are preserved. New anchor result JSON has explicit coding_order; D3 requires it. Historical C2 anchor fallback remains for compatibility only.

Use the same locked dataset, profile, seed, mixed schedule, rates, PPO, 40 updates, validation cadence 5. First run full real-data preflight **for each exact variant/source/toolchain**; no resume from D0/B20. D0 and D1 are the first comparison, then D1i, D2, D3 if warranted. Primary checkpoint selection remains feasible positive macro teacher F1. Secondary best_utility.pt is diagnostic, not a post-hoc alternative primary. Compare against VVC+QPA, B20, C15 and fixed (0,+1), (-1,+1), (-2,+1) on identical native toolchain. Multiple seeds and independent GT AP/RD evaluation required before claims. No Test10 tuning.

Example (set paths to actual Kaggle inputs/toolchain; use an immutable commit checkout):
```bash
python -m vvcrl preflight --output /kaggle/working/d1_gate --toolchain /kaggle/working/toolchain.json --input /kaggle/input --state-version d1-decoupled-v1 --reward-version task-budget-v3-progressive-soft-iou --rate-tolerance .02 --frames 256 --seed 20261120 --train-schedule mixed
python -m vvcrl train-mini --output /kaggle/working/d1_run --preflight /kaggle/working/d1_gate --toolchain /kaggle/working/toolchain.json --input /kaggle/input --state-version d1-decoupled-v1 --reward-version task-budget-v3-progressive-soft-iou --rate-tolerance .02 --frames 256 --seed 20261120 --train-schedule mixed --updates 40 --val-every 5 --start-mini-training
```
Check actual historical rate_tolerance from TRAIN_BINDING.json and use it consistently; do not assume defaults. D4 counterfactual advantage, expanded action range, and native modifications are intentionally deferred. CPU tests and native fixture smoke are mechanics only; real GPU preflight and matched training are required.

## Verification and training entrypoint

D-series validation now includes per-episode deterministic feature sensitivity, residual/base centered logit contributions, logit margin, and observed same-video/POC cross-rate comparisons. Probes are model sensitivity, not causal codec evidence. The budget probe recomputes derived features but holds other history fixed. PPO reports module-specific policy, weighted entropy and total preclip gradient norms and parameter changes. Preflight checks active residual heads. Online reward phase statistics are saved per temporal layer. The compact diagnostic mode (`--diagnostic-mode compact` with `diagnose`) runs PPO and three prespecified strong fixed controls on all 12 development video/rate points (48 policy/fixed encodes plus anchors); no-op is inferred from the matched anchor. Rate-specific fixed selection is development-only. SAFE import is streamed and staged; export cadence remains unchanged for safe recovery. Real GPU preflight remains mandatory.
