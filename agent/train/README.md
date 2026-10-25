# Train-only Jev controller pilot

This pilot uses exactly 5,500 FEVER train IDs from `verifier/data/train/sft/fever-oracle-direct-5500/labeled_prompts.jsonl`. The frozen 900 dev claims and their three-draw source are evaluation only. These scripts have not yet been run on a SoC GPU.

`agent/train.sbatch` has two phases:

1. `PHASE=rollouts` retrieves evidence and saves three stochastic Qwen outputs for every fixed 10-sentence chunk. It does not choose a controller action. Output: `agent/data/train_rollouts/<model>/<condition>/<run>/`, Git-ignored.
2. `PHASE=rl` runs `CONTROLLER_VARIANT=citation` (STOP/RETRY only) or `selection` (STOP/RETRY plus immediate KEEP/DROP on each current draw's cited sentences). The selection policy sees only the current verifier output and cited sentence texts. Its KEEP decisions accumulate in a set per chunk, including on RETRY. The chunk sets are pooled for one final Qwen verification. Training output: `agent/experiments/jev-rl-<variant>-from-rsi-v2p1-5500-samples/runs/<run>/`.

Only the RSI-Jev option scorer is updated; the language tower is frozen. The policy-gradient objective uses a terminal reward from final Qwen FEVER score, with penalties for extra chunk draws, missing reachable gold evidence, and excessive saved evidence. It has an EMA baseline, optional KL to the frozen initial scorer, and entropy regularization. It does not require hand-labeled STOP/RETRY choices. `replay_env.py` defines the reward, `jev_head.py` loads and saves the scorer, and `train_jev_rl.py` runs train-only episodes.

From the repository root on SoC, after checking `sinfo` and installing RSI-Jev in `.venv`:

```bash
PHASE=rollouts sbatch --export=ALL agent/train.sbatch
# Wait for a completed metrics.json in agent/data/train_rollouts/... before training.
PHASE=rl CONTROLLER_VARIANT=citation RUN_NAME=run1 sbatch --export=ALL agent/train.sbatch
PHASE=rl CONTROLLER_VARIANT=selection RUN_NAME=run1 sbatch --export=ALL agent/train.sbatch
```

Each training run writes `manifest.json`, `episodes.jsonl`, and `artifacts/checkpoints/scorer.pt`. Checkpoints are Git-ignored. Use a new run name for every attempt. Evaluate a trained head on frozen dev only after the policy and reward weights are fixed, by setting `JEV_TRAINED_HEAD` for `agent/eval.sbatch`.

**Current train/eval distinction:** evaluation can sample the pooled final verifier up to three times and let the controller choose STOP/RETRY. This first training pilot currently uses one final Qwen call for its terminal reward. Treat that as a different training environment when interpreting results; extend the train episode before claiming the policy learned pooled-stage retry behavior. The full 5,500-claim rollout and training run can be long; measure a small train-only pilot before scheduling the complete run.
