# Repository conventions

This is the team RL branch for a fact verifier. Read `README.md` for the directory and document map, then `verifier/TODO.md` for current priorities.

## Directory roles

- `core/` has shared reward definitions and interfaces. `retrieval/` builds and queries the FEVER index. `agent/` has the search/verify loop and controller scaffold. `utils/` has data preparation and shared experiment helpers. Do not recreate a `src/` wrapper.
- `verifier/model.py` contains reusable verifier inference, prompting and scoring logic. `verifier/evaluation/` contains the command-line evaluators, untrained prompt baselines and `results/`. Archived claim-level predictions are in `verifier/evaluation/results/archive/`; new outputs go under `results/<configuration>/<run>/`.
- `verifier/data/train/` contains committed SFT/GRPO examples; `verifier/data/eval/` has frozen claim IDs. Raw FEVER and the BM25/SQLite index must be downloaded or built locally; follow `verifier/docs/data.md`.
- Each training configuration lives under `verifier/experiments/<config>/` and each attempt under `runs/<run>/`. Keep its metadata, checkpoint files and logs under `runs/<run>/artifacts/`. Model weights and terminal `.log` files are ignored. Use a new run name; never overwrite an existing run.
- `verifier/docs/` records methods and historical results. `verifier/TODO.md` tracks unfinished work. `verifier/INSIGHT.md` is local and Git-ignored; do not commit it or treat it as a shared source.

## Editing and running

- Run Python modules from the repository root with `python -m ...` where a module is available. Keep Slurm submission commands rooted here. Inspect the current SoC partition and resource limits before submitting.
- Preserve historical `meta.json` values and prediction rows as records of the original run. Update current code and documentation paths rather than rewriting their provenance.
- Compare experiments with the same claim IDs, model, retrieval condition and evaluator. Record seed, revision, resources, predictions, metrics and logs for each run.
- Update `README.md`, `verifier/TODO.md` and relevant `verifier/docs/` pages when commands, layout or experiment assumptions change.
- Do not commit raw datasets, indexes, checkpoints, credentials or terminal logs. Do not push or submit cluster jobs unless the user requests it.
