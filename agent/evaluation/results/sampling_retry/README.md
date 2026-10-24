# Sampling retry evaluation

`eval_sampling_retry.py` runs three stochastic verifier draws on identical retrieved evidence and compares the first draw with a simulated confidence/NEI stopping rule. Submit from the repository root with `sbatch agent/evaluation/eval.sbatch`.

New outputs go under `runs/<eval-model>/<prompt>-cap<tokens>/<run>/` and contain `manifest.json`, `predictions.jsonl`, and `metrics.json`. Use a fresh `RUN_NAME`; the evaluator refuses to overwrite a run. Slurm writes `slurm-agent-sampling-retry-<job-id>.out` in the directory where `sbatch` was run after the job starts. The script changes to the repository root before running the evaluator.
