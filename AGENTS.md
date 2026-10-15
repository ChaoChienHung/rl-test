# Repository conventions

This is the team RL branch for a fact verifier. Read `README.md` for the directory and document map, then `verifier/TODO.md` for current priorities.

## Directory roles

- `core/` has shared reward definitions and interfaces. `retrieval/` builds and queries the FEVER index. `agent/` has the search/verify loop and controller scaffold. `utils/` has data preparation and shared experiment helpers. Do not recreate a `src/` wrapper.
- `verifier/model.py` contains reusable verifier inference, prompting and scoring logic. `verifier/evaluation/` contains the shared evaluator, single-condition Slurm entry point and `results/`. The root `run.sbatch` launches the six-condition Base/SFT × direct/CoT 1024-output-token sweep; every array task uses `verifier/evaluation/run_eval.sh`. Archived claim-level predictions are in `verifier/evaluation/results/archive/`; new outputs go under `results/<condition>/<run>/`.
- `verifier/data/train/` contains committed SFT/GRPO examples; `verifier/data/eval/` has frozen claim IDs. Raw FEVER and the BM25/SQLite index must be downloaded or built locally; follow `verifier/docs/data.md`.
- Each training configuration lives under `verifier/experiments/<config>/` and each attempt under `runs/<run>/`. Submit `verifier/experiments/run_train.sbatch` for SoC training; `TRAIN_SCRIPT` selects a configuration. `OUTPUT_ROOT` can move the experiments root to another shared filesystem. Keep metadata, checkpoint files and logs under `runs/<run>/artifacts/`. Model weights and terminal `.log` files are ignored. Use a new run name; never overwrite an existing run.
- The fixed 4,695/1,200 FEVER split is documented by `verifier/data/train/sft/fever-oracle-disjoint-4695/split_manifest.json`; verify it with `python -m utils.build_disjoint_sft_grpo_split --check`. On SoC, `verifier/experiments/submit_split_comparison.sh` submits four training jobs and makes the last GRPO job depend on the disjoint SFT job. Keep the fixed 1,200-row second-stage set identical in both comparison arms.
- `verifier/docs/` records methods and historical results. `verifier/TODO.md` tracks unfinished work. `verifier/INSIGHT.md` is local and Git-ignored; do not commit it or treat it as a shared source.

## Editing and running

- Run Python modules from the repository root with `python -m ...` where a module is available. Keep Slurm submission commands rooted here. `EVAL_CONDITION` names an evaluation condition and its results folder; `RESULTS_ROOT` moves the evaluation results root while retaining condition/run subdirectories; `RESULT_DIR` selects one exact new run directory. Set only one of the two path overrides. Inspect the current SoC partition and resource limits before submitting.
- Preserve historical `meta.json` values and prediction rows as records of the original run. Update current code and documentation paths rather than rewriting their provenance.
- Name new evaluation result conditions with lowercase hyphens: `<model>-<prompt>-cap<output-tokens>` for cap-controlled runs. Use `cot-long` and `cot-brief` in result paths; the historical/internal prompt-style IDs remain `cot_chain` and `cot_brief`. Do not rename older 160/512-token runs merely to make their names uniform.
- Compare experiments with the same claim IDs, model, retrieval condition and evaluator. Record seed, revision, resources, predictions, metrics and logs for each run.
- Update `README.md`, `verifier/TODO.md` and relevant `verifier/docs/` pages when commands, layout or experiment assumptions change.
- Do not commit raw datasets, indexes, checkpoints, credentials or terminal logs. Do not push or submit cluster jobs unless the user requests it.
