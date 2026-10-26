# Train-only Jev controller pilot

This pilot uses exactly 5,500 FEVER train IDs from `verifier/data/train/sft/fever-oracle-direct-5500/labeled_prompts.jsonl`. The frozen 900 dev claims and their three-draw source are evaluation only. These scripts have not yet been run on a SoC GPU.

`agent/train.sbatch` has two phases:

1. `PHASE=rollouts` retrieves evidence and saves three stochastic Qwen outputs for every fixed 10-sentence chunk. It does not choose a controller action. Output: `agent/data/train_rollouts/<model>/<condition>/<run>/`, Git-ignored.
2. `PHASE=rl` runs `CONTROLLER_VARIANT=citation` (learn STOP/RETRY, discard RETRY draws, automatically keep valid citations from the STOP draw) or `selection` (learn STOP/RETRY and KEEP/DROP on every visited draw). The selection policy sees the current verifier output and its own previously kept citations from this chunk; DROP can remove a previous KEEP. It does not see uncited retrieval sentences or previous verdicts. Each chunk's final saved set is pooled for three new stochastic Qwen verifier draws. The controller can stop after draw 1 or 2; if it reaches draw 3, the same majority-verdict/confidence rule as validation picks the final output. Training output: `agent/experiments/jev-rl-<variant>-from-rsi-v2p1-5500-samples/runs/<run>/`.

Only the RSI-Jev option scorer is updated; the language tower is frozen. The selected pooled verdict contributes +1 when correct and -1 when wrong. Pooled-verifier citations receive half of the best FEVER gold-set overlap score, including a 0.25 complete-set bonus and a modest off-gold penalty. The **final unique saved set** receives the same type of gold-set score, weighted by 0.25, exactly once per claim. A KEEP during RETRY gives no immediate reward; if the ID is dropped later, it contributes nothing to the final set score. Missing annotated gold evidence lowers coverage; saving off-gold citations incurs a smaller penalty. The first two pooled STOP/RETRY actions receive mean feedback: correct-verdict STOP +1, wrong-verdict STOP -1, correct-verdict RETRY -1.5, and wrong-verdict RETRY +0.25 only if the selected final verdict becomes correct (otherwise -1). Each extra chunk or pooled draw costs 0.05. The forced third draw has no STOP/RETRY action feedback. FEVER annotations may omit valid evidence, so an off-gold citation is not declared false. The objective has an EMA baseline, optional KL to the frozen initial scorer, and entropy regularization. `replay_env.py` defines reward helpers; `train_jev_rl.py` runs train-only episodes.

From the repository root on SoC, after checking `sinfo` and installing RSI-Jev in `.venv` with `python -m pip install --no-cache-dir -r agent/requirements.txt` (set `TMPDIR="$HOME/.pip-tmp"` first; see the root README):

```bash
PHASE=rollouts sbatch --export=ALL agent/train.sbatch
# Wait for a completed metrics.json in agent/data/train_rollouts/... before training.
PHASE=rl CONTROLLER_VARIANT=citation RUN_NAME=run1 sbatch --export=ALL agent/train.sbatch
PHASE=rl CONTROLLER_VARIANT=selection RUN_NAME=run1 sbatch --export=ALL agent/train.sbatch
```

Each training run writes `manifest.json`, `episodes.jsonl`, and `artifacts/checkpoints/scorer.pt` plus `scorer.json` metadata. Checkpoint weights are Git-ignored. Use a new run name for every attempt. Evaluate a trained head on frozen dev only after the policy and reward weights are fixed, by setting `JEV_TRAINED_HEAD` for `agent/eval.sbatch`. With `EVAL_METHOD=rsi_jev_evidence_selection_baseline`, read `metrics.json`'s `primary_trained_policy` and `primary_trained_quality`: the selection head is evaluated as `jev` (STOP/RETRY plus KEEP/DROP), while the citation head is evaluated as `jev_citation` (Jev STOP/RETRY at chunk and pooled stages, automatic retention of valid STOP citations). Do not use the full `jev` arm as the citation head's primary result; that head did not train KEEP/DROP.

Both training and validation use the same pooled-stage action budget and final-selection rule. Training samples actions from the trainable head; validation chooses the head's top action. Training generates all three pooled candidates for each episode but charges its **simulated** extra draws in the reward. The full 5,500-claim rollout and training run can be long; measure a small train-only pilot before scheduling the complete run.
