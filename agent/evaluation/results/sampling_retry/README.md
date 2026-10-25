# Sampling retry evaluation

`eval_sampling_retry.py` runs three stochastic verifier draws on identical retrieved evidence and compares the first draw with a simulated confidence/NEI stopping rule. Submit from the repository root with `EVAL_METHOD=sampling_retry sbatch --export=ALL agent/evaluation/eval.sbatch` (or omit `EVAL_METHOD`, which defaults to `sampling_retry`).

New outputs go under `runs/<eval-model>/<prompt>-cap<tokens>/<run>/` and contain `manifest.json`, `predictions.jsonl`, and `metrics.json`. Use a fresh `RUN_NAME`; the evaluator refuses to overwrite a run. New jobs write `slurm-agent-eval-<job-id>.out` in the directory where `sbatch` was run after the job starts. Historical jobs keep their original log names. The script changes to the repository root before running the evaluator.
