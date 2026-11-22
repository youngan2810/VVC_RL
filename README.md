# VVC_RL — task-aware residual QP control over native VVenC QPA

This is a new research implementation, not a resume of the earlier Gaussian/PPO or D3QN experiments. QPA and rate control remain active. The learned policy selects an integer residual per CTU to preserve object-detection information under a bitrate budget. Only QP control is implemented; Partition, MTS and other machine tasks are future work.

## Research design

- PPO-Clip + GAE; two learned categorical factors per coded picture (source-ROI / background), each in `{-2,-1,0,1,2}`, mapped to the existing 12 integer CTU residuals. QPA still supplies the detailed base QPs. Neither region is forced to receive a particular sign or nonzero action.
- Native 480×320, 30 fps, CTU128, QPA enabled, medium preset, GOP32, one-pass RC, serial coded pictures. Two independent GPU detector/encoder collectors; PPO itself runs on CPU.
- State: 8 spatial channels and 15 global features. Frozen source-frame YOLOv5n semantic cues complement native QPA information. Current decoded detections are reward/auxiliary targets, never current-state inputs.
- Reward V2 (default): `J = (F1 - F1_QPA) - price * max(0, (bits - cap)/target_bits)`, with `cap = (1 + tolerance) * min(target_bits, QPA_bits)`. Default tolerance is 2%, shared by reward, rate-price adaptation and validation checkpoint selection; use `--rate-tolerance 0` for no slack. There is no bonus for spare bits below the cap. Task gain is dense; the overshoot penalty is applied exactly once at the final **coding step**, with actual Annex-B accounting verified after encoder completion. This remains a soft training penalty; only validation-feasible checkpoints qualify as best.
- Auxiliary action-conditioned source-object retention prediction is retained. It is not described as a causal CTU reward.
- Native QG residual and lambda changes, quantization candidates, winning CU QP pixel coverage, actual bitstream size, and independent VVdeC reconstructions are audited separately.

Current semantic-region policy, unchanged V2 objective and historical definitions: [docs/DESIGN.txt](docs/DESIGN.txt). Code uses task-retention F1 v2 (empty teacher + empty prediction = 1) and also records legacy F1. These teacher-retention scores are **not ground-truth mAP**.

## Run on Kaggle first

Import [notebooks/VVC_RL_QPA_Preflight_Kaggle.ipynb](notebooks/VVC_RL_QPA_Preflight_Kaggle.ipynb), enable Internet and **two GPUs**, then add **one dataset containing the original locked BDD100K `.mov` videos** (the Driving Video dataset is usable if its files match the lock). Metadata-only `bdd100k.zip`, snapshot, old training checkpoints and previous evaluation exports are not required. Video identities, bytes and mini splits are checked from the lock included in this repository: 16 train / 4 validation; 30 Test videos remain reserved.

The notebook defaults to **preflight only**. It builds vanilla and instrumented VVenC from the same pinned source, builds pinned VVdeC, checks the frozen YOLO source/weights, runs native action tests, collects real train rollouts on both GPUs, performs one mechanical PPO update, and verifies validation without updating weights. This single update is a mechanism test, not a trained model.

**Schedule-mix-only experiment (2026-11-22):** The two Train collectors now receive different source videos in every update via deterministic, seed-based interleaving. All 16 videos × 5 rates are covered exactly once across 40 updates. PPO/reward/model/native, validation pairs, bitrate cap, and mini budget are unchanged. The Kaggle preflight launcher pins the revised source; use its separate `vvc_rl_mini_regions_mixed` output, run a fresh preflight, and **start from new weights** rather than resuming historical ROI/background checkpoints. This is an ablation, not a claim of improved detection.

The current preflight notebook pins the semantic-region runtime and offers **40 additional mini updates = 80 train clips = 20,480 transitions**. State, reward, bitrate tolerance, GAE lambda, LR, losses, seed and dataset stay fixed; the controlled change for this ablation is video interleaving; the existing semantic-region policy stays fixed. The other mini launchers are historical V1/V2 independent-CTU notebooks pinned to their original commits. They cannot launch or authorize this policy. Use the current preflight entry point for a fresh experiment; no old preflight/checkpoint is silently migrated.

**Three controlled experiments on the same source and locked Train/Validation split:**

| Experiment | Train schedule | Gamma | GAE lambda | Kaggle output |
| --- | --- | --- | --- | --- |
| A | blocked (original order) | 1.0 | 0.95 | `vvc_rl_mini_A` |
| B | mixed (distinct videos per update) | 1.0 | 0.95 | `vvc_rl_mini_B` |
| C | mixed (distinct videos per update) | 1.0 | 1.0 | `vvc_rl_mini_C` |

The existing Kaggle notebook defaults to B with training disabled. Change `EXPERIMENT` before the preflight to select A or C. Each needs a fresh configuration-bound preflight; outputs and SAFE archives use distinct names. A here means a **fresh blocked-order control run**, not the historical already-trained checkpoint. Preserve old outputs. `--gamma 1 --gae-lambda {0.95|1.0} --train-schedule {blocked|mixed}` must match for preflight, training, checkpoint loading, and diagnosis. A checkpoint from B is rejected under C, and vice versa. These are ablations, not claims of a better F1; lambda=1 may increase variance/value gradients.

Mini training requires both an intact `PASS_READY_FOR_MINI` report bound to the same source/profile/binaries/environment and an explicit `RUN_MINI_TRAINING=True`. No long training is started automatically. A passing preflight does not claim useful learning.

## Commands

```bash
python -m unittest discover -s tests -v
python scripts/setup_kaggle.py --tools /kaggle/working/vvc_rl_tools
python -m vvcrl preflight --output /kaggle/working/vvc_rl_preflight --toolchain /kaggle/working/vvc_rl_tools/TOOLCHAIN.json
# Only after reviewing the real-data preflight:
python -m vvcrl train-mini --output /kaggle/working/vvc_rl_mini --toolchain /kaggle/working/vvc_rl_tools/TOOLCHAIN.json --preflight /kaggle/working/vvc_rl_preflight --updates 10 --start-mini-training
```

For a checkpoint resume, add `--resume /path/checkpoints/latest.pt` and keep its adjacent `.identity.json`. Source, profile and environment changes invalidate compatibility. The semantic-region policy requires a fresh model and preflight: old V1/V2 checkpoints remain frozen controls, not optimizer/weight resume inputs. Checkpoint format `/2` is unchanged; experiment schema `/3` and source fingerprint distinguish this policy. A `paired-linear-v1` run here changes only the formula, not the policy, so it is not an exact V1 replication. Historical V1 remains at `4d185279e0ba6596ff31bbf6359ab88a33f36288`, independent-CTU V2 at `968d9de9792b790d65fde25424f80e602537a549`.

Additional `--updates` means additional complete two-clip PPO updates, not episodes or epochs. The schedule rotates all 16 train videos × 5 rates; a short run is not a complete schedule. Windows encoder builds use `scripts/build_native.py` with MinGW; the complete real-data preflight requires Linux/CUDA.

`python -m vvcrl diagnose` provides validation-only paired PPO/random/fixed−2/fixed+2/no-op/ROI-protect/ROI-redistribute comparisons at 120 kbps. Pass the same toolchain/preflight and trained checkpoint using `--resume`. It does not optimize on validation and is not a BD-rate study. Mini training records validation at 50/120/200 kbps and retains a best checkpoint only for a positive macro task gain, complete unique validation pairs and every clip satisfying the same bitrate cap used by reward. It logs task/bitrate reward components, requested/effective actions, entropy, budget feasibility and per-clip quality regressions. No Test data changes prices or policy. If no checkpoint qualifies, there is no claimed best model.

SAFE exports preserve the payload as `.bin` inside a ZIP and contain checkpoints, policy logits/actions, rollout arrays, bitstreams, QP/lambda traces, source/decoded detector boxes, metrics and hashes. Raw source/reconstructed video is excluded after decode parity is verified. Keep the whole SAFE archive. `python -m vvcrl import --package ... --output ...` validates it before extracting into an empty directory. Export commands use `--output` as the existing run folder.

## Verification and limits

Independent-CTU V2 completed 40 updates but all 96 validation clips were no-op. In 20,480 sampled Train frames, no map was coherent within both ROI/background regions; the expected number of exact ROI(-2)/background(+1) maps under recorded logits was only 0.0000629. Joint PPO log-probability was correct, but exploration rarely tested coordinated allocations previously observed to help on some matched controls. This motivates a smaller learned action space; it does not prove a unique cause or guarantee better F1. Global F1 still cannot provide causal per-CTU credit.

The two factors are pooled by actual CTU area from separate ROI/background logits. ROI comes only from the existing source-occupancy state. Inactive factors are marginalized. The rollout still stores the exact 12 integer CTU categories; replay recovers their unique active factors and rejects incoherent old maps. Log probability and joint entropy count each active factor once, never once per CTU. Both factors may select any sign or no-op. Auxiliary remains action-conditioned on the executed CTU map. At the same host/seed, both actor heads clone the old initial head, while shared/Critic/Aux tensors and CPU RNG state remain identical to the original initialization; the heads then train independently.

GAE stays at 0.95: a recorded-data lambda=1 probe removed direct terminal attenuation but enlarged value gradients sharply on overshooting clips (e.g. 1.22 to 15.97), so it was not combined with this change. Biases retain the same mild no-op prior. StateBuilder, reward, PPO optimizer, Auxiliary, native QPA/RC code, metric, bitrate cap and dataset lock are unchanged. Validation/Test data never train the policy. Real BDD100K learning and useful F1/bitrate improvement remain unverified for this policy.

See [docs/VERIFICATION.json](docs/VERIFICATION.json) for actual local checks. Local CPU/native smoke fixtures cannot authorize mini training and do not establish detection gains. Real BDD100K + YOLO + two-GPU preflight remains mandatory on Kaggle. Neither better accuracy nor lower BD-rate is guaranteed by this implementation. Final held-out Test/ground-truth mAP and BD analyses belong after the learning check; they are not silently run by this notebook.

## Source and licensing

`vendor/vvenc` includes the full VVenC source at `1a4f83bd9feeb91c6792fb8e0676b1c8f3683d4f` with the narrow native integration. `native/originals` and `native/online_manifest.json` preserve the original/modified hashes. Vanilla baseline is restored from those originals; old CTU32 or unrelated binary results must not be substituted. External decoder/detector versions and weights are pinned separately. See [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and preserved upstream licenses. No models, credentials, videos or historical results are published.
