# C1–C3 experimental source branch

Base: main 161adf96456e36c20340885737a13489bc88d45f.
Frozen historical B20: ed8f5d96bef821d08c1e5ee9329345210733f58a. Keep its notebook,
checkpoint and toolchain at that commit. New fingerprints intentionally prohibit
resuming old optimizer/checkpoint under a changed experiment. Legacy state and
reward still use 8x3x4 / 15 and the original PPO architecture. Versioned checkpoints use /3; historical /2
can only be read under their exact old binding, never resumed into this branch.
Common C1 weights and RNG are initialized identically to C0, new input weights
start at zero and are trainable. New training must
start in a separate output directory; do not overwrite B20 or use Test10.

| Ablation | --state-version | --reward-version |
|---|---|---|
| C0 | legacy-v1 | task-budget-v2 |
| C1 | encoder-feedback-v1 | task-budget-v2 |
| C2 | legacy-v1 | task-budget-v3-progressive |
| C3 | legacy-v1 | task-budget-v2-soft-iou |
| Combined (not selected) | encoder-feedback-v1 | task-budget-v3-progressive-soft-iou |

C1: three extra spatial channels (previous area-weighted winning CU QP /63,
winning QP minus previous *pre-action CTU QPA* /63, previous ROI mask), three
extra global features (previous area-weighted ROI minus BG QP /63 and two
region-present flags). Existing history flag marks initial zeros. The response
is not the effective candidate QG delta and not region bit allocation.
Previous means previous coding step, not POC-1. No future RL outcome enters
the state. Only native change: opt-in trace flush in reconstruction callback,
before outcome, without altering encoder decisions. A rebuilt instrumented
binary has a new hash; run parity checks before interpreting results.

C2: cap unchanged (1.02 * min(target, anchor)), same price within clip and
gamma=1. Use actual anchor AU distribution in coding order as a predetermined
schedule. Excess increments telescope; temporary excess can be refunded.
No extra reward below cap, no double terminal penalty. No-op reference runs
without a known anchor use linear progress with terminal correction.
C3: soft F1 = 2 * sum IoU over the original matched detections / (Nref+Npred);
same confidence-ordered one-to-one matching and IoU >= .5, empty/empty = 1.
Unmatched detections penalized. Confidence/size weighting deferred.
Both anchor and policy use the same utility. Legacy F1 remains in every
report and is still the checkpoint selection criterion, subject to all caps.
GT-mAP on labeled validation frames is required before selecting a new
utility or claiming machine-vision gains. No Test use.

## Running (Kaggle, locked data and detector)
From an exact commit of this branch, rebuild using the repository setup script:
```sh
python scripts/setup_kaggle.py --tools /kaggle/working/vvc_rl_tools --jobs 2
python scripts/verify_source.py
python -m unittest discover -s tests -v
python -m vvcrl native-smoke --output /kaggle/working/smoke_c --toolchain /kaggle/working/vvc_rl_tools/TOOLCHAIN.json
python -m vvcrl preflight --output /kaggle/working/preflight_c1 --toolchain /kaggle/working/vvc_rl_tools/TOOLCHAIN.json --input /kaggle/input --tools /kaggle/working/vvc_rl_tools --state-version encoder-feedback-v1 --reward-version task-budget-v2
```
Repeat preflight separately for each configuration. The full real-data gate
requires two CUDA collectors and includes exactly one mechanical train-only PPO
update, not full training. Only after reviewing the gate, run train-mini with
the identical state/reward flags, same seed and predeclared budget, fresh output,
and --start-mini-training. Resume only from the same binding; use --resume
with matching latest checkpoint and count additional updates. The original
historical notebook remains pinned to B20; do not replace its hash.
C4/C5 architecture/optimizer tuning intentionally deferred until ablation.
The validation-only diagnose command includes all four exact ROI/BG controls
from the 72-case native validation. It is a one-rate diagnostic, not BD-rate.
The real-data gate remains mandatory; the native smoke also exercises a single
synthetic combined PPO update and never authorizes mini training.

Acceptance: source and native parity, causality and replay, exact raw/scaled
reward accounting, independent decode, checkpoint/resume, real-data gate, then
paired validation F1, actual bitrate and all caps against C0/B20/fixed controls,
GT-mAP and multiple seeds. Unit/synthetic smoke evidence is not useful learning.

## Pre-training audit hardening
Resume requires the latest checkpoint inside the same imported run, matching
RUN_PROTOCOL.json (seed and validation cadence) and PROGRESS.json (checkpoint
hash, completed updates and pending validation). Incomplete validation must
finish before further training; a crash between checkpoint and journal writes
fails closed; recover the last intact SAFE export. Do not resume historical
artifacts into this branch. Full preflight exercises the production checkpoint
save/load and verifies RNG and dual prices. Cache keys bind video/split and
preprocessing or decoder as appropriate. All changes are mechanical safeguards,
not optimizer or policy changes. Real-data preflight still requires two GPUs.

At the end of the 40-update schedule, resume with `--updates 0` only if a
validation is still pending; this performs no PPO updates. Failure exports
have a separate `_FAILED_<invocation>_SAFE.zip` name; the last consistent SAFE
is preserved. Preflight checks SHA-256 of every locked mini Train/Val MOV.
