# Train-only Jev controller pilot

This pilot uses exactly 5,500 FEVER train IDs from `verifier/data/train/sft/fever-oracle-direct-5500/labeled_prompts.jsonl`. The frozen 900 dev claims and their three-draw source are evaluation only. These scripts have not yet been run on a SoC GPU.

`agent/train.sbatch` has two phases:

1. `PHASE=rollouts` retrieves evidence and saves three stochastic Qwen outputs for every fixed 10-sentence chunk. It does not choose a controller action. Output: `agent/data/train_rollouts/<model>/<condition>/<run>/`, Git-ignored.
2. `PHASE=rl` runs `CONTROLLER_VARIANT=citation` (learn STOP/RETRY, automatically keep valid citations from the STOP draw) or `selection` (learn STOP/RETRY, then KEEP/DROP for citations from the STOP draw). RETRY discards the current draw and its citations; both variants see only the current verifier output and cited sentence texts. Saved citations from each chunk's STOP draw are pooled for three new stochastic Qwen verifier draws. The controller can stop after draw 1 or 2; if it reaches draw 3, the same majority-verdict/confidence rule as validation picks the final output. Training output: `agent/experiments/jev-rl-<variant>-from-rsi-v2p1-5500-samples/runs/<run>/`.

Only the RSI-Jev option scorer is updated; the language tower is frozen. The selected pooled verdict contributes +1 when correct and -1 when wrong. Citation reward is half of the best FEVER gold-set overlap fraction, plus a 0.25 complete-set bonus and minus 0.25 times the fraction of off-gold citations; a correct NEI prediction earns a 0.25 empty-citation bonus. The first two pooled STOP/RETRY actions receive mean feedback: correct-verdict STOP +1, wrong-verdict STOP -1, correct-verdict RETRY -1.5, and wrong-verdict RETRY +0.25 only if the selected final verdict becomes correct (otherwise -1). The `selection` variant also gets a small KEEP/DROP term **only for the STOP draw**: KEEP annotated gold +1, DROP annotated gold -0.5, KEEP unannotated -0.25, DROP unannotated +0.1; average these decisions and multiply by 0.25. Skip this term for NEI or claims with no annotated gold set. The small DROP reward avoids encouraging an all-DROP policy. Each extra chunk or pooled draw costs 0.05. The forced third draw has no STOP/RETRY action feedback. Chunk STOP/RETRY receives downstream episode reward; KEEP/DROP also receives the local annotation-based term. FEVER annotations may omit valid evidence, so an unannotated citation is not declared false. The objective has an EMA baseline, optional KL to the frozen initial scorer, and entropy regularization. `replay_env.py` defines reward helpers; `train_jev_rl.py` runs train-only episodes.

From the repository root on SoC, after checking `sinfo` and installing RSI-Jev in `.venv`:

```bash
PHASE=rollouts sbatch --export=ALL agent/train.sbatch
# Wait for a completed metrics.json in agent/data/train_rollouts/... before training.
PHASE=rl CONTROLLER_VARIANT=citation RUN_NAME=run1 sbatch --export=ALL agent/train.sbatch
PHASE=rl CONTROLLER_VARIANT=selection RUN_NAME=run1 sbatch --export=ALL agent/train.sbatch
```

Each training run writes `manifest.json`, `episodes.jsonl`, and `artifacts/checkpoints/scorer.pt`. Checkpoints are Git-ignored. Use a new run name for every attempt. Evaluate a trained head on frozen dev only after the policy and reward weights are fixed, by setting `JEV_TRAINED_HEAD` for `agent/eval.sbatch`.

Both training and validation use the same pooled-stage action budget and final-selection rule. Training samples actions from the trainable head; validation chooses the head's top action. Training generates all three pooled candidates for each episode but charges its **simulated** extra draws in the reward. The full 5,500-claim rollout and training run can be long; measure a small train-only pilot before scheduling the complete run.
