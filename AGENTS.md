# Repository conventions

This is the team RL branch for the fact verifier. Read `README.md` for the document map and `TODO.md` for current priorities before editing. The root contains top-level project documents; detailed historical results and methods are in `docs/`.

## Files and runs

- `data/train/sft/` contains committed SFT prompt/completion pairs. `data/train/grpo/` contains prepared GRPO prompts and disagreement samples. `data/eval/` contains frozen evaluation IDs. Raw FEVER, SciFact, AVeriTeC and retrieval indexes must be downloaded or built locally; follow `docs/data.md`.
- `data/stage0/` and `data/traces/` contain archived predictions and agent traces. Keep them as historical records. New evaluation outputs go to `results/eval/<version>/<run>/`.
- Keep each training configuration under `experiments/<config>/` and each attempt under `runs/<run>/`. Run-specific metadata, checkpoint files and training logs belong under `runs/<run>/artifacts/`; terminal logs go in `artifacts/logs/`. The model weights are ignored by Git, while small metadata and structured `train_log.json` may be committed.
- The two prompt controls are `experiments/direct-baseline/` and `experiments/cot-baseline/`. `experiments/run_eval.sh` is shared execution logic; each experiment's `run.sbatch` is its Slurm entry point. Use a new `RUN_NAME` for each attempt. Never overwrite an existing run.
- Training entry points are the configuration-specific `experiments/<config>/train.py` files. Shared trainers are in `experiments/_shared/`; reusable helpers are in `src/utils/`. Keep resource requests in `.sbatch` files aligned with the target cluster.

## Working practices

- Do not rewrite historical `meta.json` paths or prediction JSONL just to match the new tree; those values record the original runs. Update present-day code and documentation paths instead.
- Do not claim a historical adapter is available because an `adapter_config.json` exists; the weight file is absent from Git. Check data availability before scheduling training or evaluation.
- Keep comparisons on the same claim IDs, model, retrieval condition and evaluator. Preserve run manifests, seeds, and logs so repeated runs are traceable.
- When changing layout or commands, update `README.md` and relevant `docs/` pages. Track outstanding work in `TODO.md`. `INSIGHT.md` is a local, gitignored private note and is not a shared project document.
- Do not commit raw datasets, indexes, model weights, credentials or generated terminal `.log` files. Do not push or submit jobs unless the user asks.
