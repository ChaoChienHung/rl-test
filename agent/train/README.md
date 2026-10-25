# Train-only Jev controller pilot

This pilot uses exactly 5,500 FEVER train IDs from `verifier/data/train/sft/fever-oracle-direct-5500/labeled_prompts.jsonl`. The frozen 900 dev claims and their three-draw source are evaluation only. These scripts have not yet been run on a SoC GPU.

`agent/train.sbatch` has two phases:

1. `PHASE=rollouts` retrieves evidence and saves three stochastic Qwen outputs for every fixed 10-sentence chunk. It does not choose a controller action. Output: `agent/data/train_rollouts/<model>/<condition>/<run>/`, Git-ignored.
2. `PHASE=rl` runs `CONTROLLER_VARIANT=citation` (STOP/RETRY only) or `selection` (STOP/RETRY plus immediate KEEP/DROP on each current draw's cited sentences). The selection policy sees only the current verifier output and cited sentence texts. Its KEEP decisions accumulate in a set per chunk, including on RETRY. The chunk sets are pooled for three new stochastic Qwen verifier draws. The controller can stop after draw 1 or 2; if it reaches draw 3, the same majority-verdict/confidence rule as validation picks the final output. Training output: `agent/experiments/jev-rl-<variant>-from-rsi-v2p1-5500-samples/runs/<run>/`.

Only the RSI-Jev option scorer is updated; the language tower is frozen. For the first pilot, a pooled verifier draw is **good** when its verdict matches FEVER gold and its evidence F1 is strictly above 2/3; a correct NEI verdict is good without evidence F1. On the first two pooled draws, good STOP and bad RETRY receive +1; bad STOP and good RETRY receive -1. The mean pooled-action feedback is added to +1/-1 for the selected pooled output, with 0.05 subtracted per extra chunk or pooled draw. The forced third draw has no STOP/RETRY action reward. Chunk STOP/RETRY and KEEP/DROP actions receive only this downstream episode signal, not a chunk-level gold label. The objective has an EMA baseline, optional KL to the frozen initial scorer, and entropy regularization. `replay_env.py` defines the good-output and action-feedback rules, `jev_head.py` loads and saves the scorer, and `train_jev_rl.py` runs train-only episodes.

From the repository root on SoC, after checking `sinfo` and installing RSI-Jev in `.venv`:

```bash
PHASE=rollouts sbatch --export=ALL agent/train.sbatch
# Wait for a completed metrics.json in agent/data/train_rollouts/... before training.
PHASE=rl CONTROLLER_VARIANT=citation RUN_NAME=run1 sbatch --export=ALL agent/train.sbatch
PHASE=rl CONTROLLER_VARIANT=selection RUN_NAME=run1 sbatch --export=ALL agent/train.sbatch
```

Each training run writes `manifest.json`, `episodes.jsonl`, and `artifacts/checkpoints/scorer.pt`. Checkpoints are Git-ignored. Use a new run name for every attempt. Evaluate a trained head on frozen dev only after the policy and reward weights are fixed, by setting `JEV_TRAINED_HEAD` for `agent/eval.sbatch`.

Both training and validation use the same pooled-stage action budget and final-selection rule. Training samples actions from the trainable head; validation chooses the head's top action. Training generates all three pooled candidates for each episode but charges its **simulated** extra draws in the reward. The full 5,500-claim rollout and training run can be long; measure a small train-only pilot before scheduling the complete run.
