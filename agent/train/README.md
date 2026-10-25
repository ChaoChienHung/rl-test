# Train-only Jev controller pilot

This pilot uses exactly 5,500 FEVER train IDs from `verifier/data/train/sft/fever-oracle-direct-5500/labeled_prompts.jsonl`. The frozen 900 dev claims and their three-draw source are evaluation only. These scripts have not yet been run on a SoC GPU.

`agent/train.sbatch` has two phases:

1. `PHASE=rollouts` retrieves evidence and saves three stochastic Qwen outputs for every fixed 10-sentence chunk. It does not choose a controller action. Output: `agent/data/train_rollouts/<model>/<condition>/<run>/`, Git-ignored.
2. `PHASE=rl` runs `CONTROLLER_VARIANT=citation` (learn STOP/RETRY, automatically keep every valid cited sentence from each observed draw) or `selection` (learn STOP/RETRY plus immediate KEEP/DROP on each current draw's cited sentences). Both see only the current verifier output and cited sentence texts; both accumulate saved citations in a set per chunk, including on RETRY. The chunk sets are pooled for three new stochastic Qwen verifier draws. The controller can stop after draw 1 or 2; if it reaches draw 3, the same majority-verdict/confidence rule as validation picks the final output. Training output: `agent/experiments/jev-rl-<variant>-from-rsi-v2p1-5500-samples/runs/<run>/`.

Only the RSI-Jev option scorer is updated; the language tower is frozen. The selected pooled verdict contributes +1 when correct and -1 when wrong. Citation reward is half of the best FEVER gold-set overlap fraction, plus a 0.25 complete-set bonus and minus 0.25 times the fraction of off-gold citations; a correct NEI prediction earns a 0.25 empty-citation bonus. The first two pooled STOP/RETRY actions receive mean feedback: correct-verdict STOP +1, wrong-verdict STOP -1, correct-verdict RETRY -1.5, and wrong-verdict RETRY +0.25 only if the selected final verdict becomes correct (otherwise -1). Each extra chunk or pooled draw costs 0.05. The forced third draw has no action feedback. Chunk STOP/RETRY and KEEP/DROP receive only this downstream episode signal, not a chunk-level gold label. FEVER citations are incomplete annotations, so off-gold citations receive a modest penalty rather than being declared false. The objective has an EMA baseline, optional KL to the frozen initial scorer, and entropy regularization. `replay_env.py` defines reward helpers; `train_jev_rl.py` runs train-only episodes.

From the repository root on SoC, after checking `sinfo` and installing RSI-Jev in `.venv`:

```bash
PHASE=rollouts sbatch --export=ALL agent/train.sbatch
# Wait for a completed metrics.json in agent/data/train_rollouts/... before training.
PHASE=rl CONTROLLER_VARIANT=citation RUN_NAME=run1 sbatch --export=ALL agent/train.sbatch
PHASE=rl CONTROLLER_VARIANT=selection RUN_NAME=run1 sbatch --export=ALL agent/train.sbatch
```

Each training run writes `manifest.json`, `episodes.jsonl`, and `artifacts/checkpoints/scorer.pt`. Checkpoints are Git-ignored. Use a new run name for every attempt. Evaluate a trained head on frozen dev only after the policy and reward weights are fixed, by setting `JEV_TRAINED_HEAD` for `agent/eval.sbatch`.

Both training and validation use the same pooled-stage action budget and final-selection rule. Training samples actions from the trainable head; validation chooses the head's top action. Training generates all three pooled candidates for each episode but charges its **simulated** extra draws in the reward. The full 5,500-claim rollout and training run can be long; measure a small train-only pilot before scheduling the complete run.
