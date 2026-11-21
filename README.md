# VVC_RL — task-aware residual QP control over native VVenC QPA

This is a new research implementation, not a resume of the earlier Gaussian/PPO or D3QN experiments. QPA and rate control remain active. The learned policy selects an integer residual per CTU to preserve object-detection information under a bitrate budget. Only QP control is implemented; Partition, MTS and other machine tasks are future work.

## Research design

- PPO-Clip + GAE; 12 categorical decisions per coded picture, each in `{-2,-1,0,1,2}`. No Gaussian rounding or forced nonzero decisions.
- Native 480×320, 30 fps, CTU128, QPA enabled, medium preset, GOP32, one-pass RC, serial coded pictures. Two independent GPU detector/encoder collectors; PPO itself runs on CPU.
- State: 8 spatial channels and 15 global features. Frozen source-frame YOLOv5n semantic cues complement native QPA information. Current decoded detections are reward/auxiliary targets, never current-state inputs.
- Reward V2 (default): `J = (F1 - F1_QPA) - price * max(0, (bits - cap)/target_bits)`, with `cap = (1 + tolerance) * min(target_bits, QPA_bits)`. Default tolerance is 2%, shared by reward, rate-price adaptation and validation checkpoint selection; use `--rate-tolerance 0` for no slack. There is no bonus for spare bits below the cap. Task gain is dense; the overshoot penalty is applied exactly once at the final **coding step**, with actual Annex-B accounting verified after encoder completion. This remains a soft training penalty; only validation-feasible checkpoints qualify as best.
- Auxiliary action-conditioned source-object retention prediction is retained. It is not described as a causal CTU reward.
- Native QG residual and lambda changes, quantization candidates, winning CU QP pixel coverage, actual bitstream size, and independent VVdeC reconstructions are audited separately.

Current V2 objective and historical V1 state/action definitions, rationale and limitations: [docs/DESIGN.txt](docs/DESIGN.txt). Code uses task-retention F1 v2 (empty teacher + empty prediction = 1) and also records legacy F1. These teacher-retention scores are **not ground-truth mAP**.

## Run on Kaggle first

Import [notebooks/VVC_RL_QPA_Preflight_Kaggle.ipynb](notebooks/VVC_RL_QPA_Preflight_Kaggle.ipynb), enable Internet and **two GPUs**, then add **one dataset containing the original locked BDD100K `.mov` videos** (the Driving Video dataset is usable if its files match the lock). Metadata-only `bdd100k.zip`, snapshot, old training checkpoints and previous evaluation exports are not required. Video identities, bytes and mini splits are checked from the lock included in this repository: 16 train / 4 validation; 30 Test videos remain reserved.

The notebook defaults to **preflight only**. It builds vanilla and instrumented VVenC from the same pinned source, builds pinned VVdeC, checks the frozen YOLO source/weights, runs native action tests, collects real train rollouts on both GPUs, performs one mechanical PPO update, and verifies validation without updating weights. This single update is a mechanism test, not a trained model.

The current notebook pins an exact V2 runtime commit, defaults to preflight only, and offers **40 additional mini updates = 80 train clips = 20,480 transitions**, with the same seed and PPO/state/action settings as the reviewed V1 mini. The old `VVC_RL_Mini_Train_From_Preflight_Kaggle.ipynb` is a **historical V1 notebook pinned to its old commit**, not the V2 entry point.

Mini training requires both an intact `PASS_READY_FOR_MINI` report bound to the same source/profile/binaries/environment and an explicit `RUN_MINI_TRAINING=True`. No long training is started automatically. A passing preflight does not claim useful learning.

## Commands

```bash
python -m unittest discover -s tests -v
python scripts/setup_kaggle.py --tools /kaggle/working/vvc_rl_tools
python -m vvcrl preflight --output /kaggle/working/vvc_rl_preflight --toolchain /kaggle/working/vvc_rl_tools/TOOLCHAIN.json
# Only after reviewing the real-data preflight:
python -m vvcrl train-mini --output /kaggle/working/vvc_rl_mini --toolchain /kaggle/working/vvc_rl_tools/TOOLCHAIN.json --preflight /kaggle/working/vvc_rl_preflight --updates 10 --start-mini-training
```

For a checkpoint resume, add `--resume /path/checkpoints/latest.pt` and keep its adjacent `.identity.json`. Source, profile and environment changes invalidate preflight/checkpoint compatibility by design. V2 must start fresh: old V1 checkpoints remain frozen controls and cannot resume V2 weights/optimizer/prices. Checkpoint schema `/2` and binding include the reward version/tolerance. For a fresh V1-formula A/B run on this implementation, pass `--reward-version paired-linear-v1` consistently to both preflight and train-mini; do not use a V2 preflight for V1. Existing historical V1 results remain available at commit `4d185279e0ba6596ff31bbf6359ab88a33f36288`.

Additional `--updates` means additional complete two-clip PPO updates, not episodes or epochs. The schedule rotates all 16 train videos × 5 rates; a short run is not a complete schedule. Windows encoder builds use `scripts/build_native.py` with MinGW; the complete real-data preflight requires Linux/CUDA.

`python -m vvcrl diagnose` provides validation-only paired PPO/random/fixed−2/fixed+2/no-op/ROI-protect/ROI-redistribute comparisons at 120 kbps. Pass the same toolchain/preflight and trained checkpoint using `--resume`. It does not optimize on validation and is not a BD-rate study. Mini training records validation at 50/120/200 kbps and retains a best checkpoint only for a positive macro task gain, complete unique validation pairs and every clip satisfying the same bitrate cap used by reward. It logs task/bitrate reward components, requested/effective actions, entropy, budget feasibility and per-clip quality regressions. No Test data changes prices or policy. If no checkpoint qualifies, there is no claimed best model.

SAFE exports preserve the payload as `.bin` inside a ZIP and contain checkpoints, policy logits/actions, rollout arrays, bitstreams, QP/lambda traces, source/decoded detector boxes, metrics and hashes. Raw source/reconstructed video is excluded after decode parity is verified. Keep the whole SAFE archive. `python -m vvcrl import --package ... --output ...` validates it before extracting into an empty directory. Export commands use `--output` as the existing run folder.

## Verification and limits

The V1 frozen check completed 40 clips with unchanged weights: PPO argmax saved 2.10% bitrate but lost 2.53 F1 percentage points; semantic ROI controls improved F1 on average while using some extra bits. That diagnoses a useful-learning problem, not proof of broken injection or a guarantee that V2 will improve. The V2 source change deliberately keeps StateBuilder, ActorCritic, PPO/GAE, Auxiliary, native QPA/RC integration and dataset lock byte-identical. Only the objective, its propagation, checks, diagnostics and run packaging are revised.

See [docs/VERIFICATION.json](docs/VERIFICATION.json) for actual local checks. Local CPU/native smoke fixtures cannot authorize mini training and do not establish detection gains. Real BDD100K + YOLO + two-GPU preflight remains mandatory on Kaggle. Neither better accuracy nor lower BD-rate is guaranteed by this implementation. Final held-out Test/ground-truth mAP and BD analyses belong after the learning check; they are not silently run by this notebook.

## Source and licensing

`vendor/vvenc` includes the full VVenC source at `1a4f83bd9feeb91c6792fb8e0676b1c8f3683d4f` with the narrow native integration. `native/originals` and `native/online_manifest.json` preserve the original/modified hashes. Vanilla baseline is restored from those originals; old CTU32 or unrelated binary results must not be substituted. External decoder/detector versions and weights are pinned separately. See [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and preserved upstream licenses. No models, credentials, videos or historical results are published.
