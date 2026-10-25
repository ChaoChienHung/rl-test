# Sampling retry evaluation

`eval_sampling_retry.py` runs three stochastic verifier draws on identical retrieved evidence and compares the first draw with a simulated confidence/NEI stopping rule. Submit from the repository root with `EVAL_METHOD=sampling_retry sbatch --export=ALL agent/eval.sbatch`.

New outputs go under `runs/<eval-model>/<prompt>-cap<tokens>/<run>/` and contain `manifest.json`, `predictions.jsonl`, and `metrics.json`. Use a fresh `RUN_NAME`; the evaluator refuses to overwrite a run. New jobs write `slurm-agent-eval-<job-id>.out` in the directory where `sbatch` was run after the job starts. Historical jobs keep their original log names. The script changes to the repository root before running the evaluator.

`metrics.json` keeps selected-output accuracy and FEVER score. New runs also include `controller_diagnostics` in each group: selected correct, exhausted budget with all three outputs wrong, exhausted budget despite an earlier correct output, early wrong stop with a later correct output, and early wrong stop without a later correct output. `controller_decision_accuracy_hindsight` credits selected-correct plus exhausted/all-wrong cases. It uses gold and future draws, so it is a post-hoc attribution only. For archived runs, generate `controller_diagnostics.json` without changing their original predictions or metrics:

```bash
python -m agent.evaluation.results.sampling_retry.reanalyse <completed-run-directory>
```
