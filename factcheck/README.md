# Verifier workspace

A claim and retrieved FEVER sentences go to a Qwen3.5-2B verifier, which returns a verdict, cited sentence IDs and confidence. The team-level overview and ownership are in the root [README](../README.md) and [plan](docs/plan.md); pending work is in the root [TODO](../TODO.md).

## Doc map

| Document | Content |
|---|---|
| [docs/plan.md](docs/plan.md) | Architecture, roles, decisions and evaluation rules |
| [docs/data.md](docs/data.md) | Raw dataset download and index build |
| [docs/stage0-results.md](docs/stage0-results.md) | Prompted and retrieval baselines |
| [docs/sft-probe.md](docs/sft-probe.md) | SFT setup and results |
| [docs/grpo-results.md](docs/grpo-results.md) | Single/multi-turn GRPO runs and results |
| [docs/trace-format.md](docs/trace-format.md) | Search/verify trace schema |
| [docs/related-work.md](docs/related-work.md) | Literature notes requiring source checks |

## Data and result map

| Path under `factcheck/` | Role | Git |
|---|---|---|
| `data/train/manifest.json` | Entry point to existing SFT/GRPO training examples | Yes |
| `data/eval/fever-dev-900-ids.json` | Frozen archived FEVER dev claim IDs for matched comparisons | Yes |
| `data/sft_probe/`, `data/grpo_data/` | Existing training sets and run metadata at their historical paths | Partly |
| `data/grpo/` | GRPO logs and run metadata | Partly |
| `data/stage0/` | Archived predictions; replay only, not fresh runs | Yes |
| `data/traces/` | Agent-loop traces | Yes |
| `data/fever/`, `data/scifact/`, `data/averitec/` | Downloaded raw corpora and FEVER index | No |
| `results/eval/direct/job-<id>/` | New direct JSON predictions, metrics, manifest, log | No |
| `results/eval/cot-chain/job-<id>/` | New evidence-chain predictions, metrics, manifest, log | No |
| `experiments/direct/`, `experiments/cot-chain/` | Version-isolated Slurm jobs | Yes |

The `data/train/` folder is a manifest over existing training files. Those files stay at their historical paths so published run metadata and scripts remain reproducible. `data/eval/` freezes the 900 archived IDs; later prompt-development, calibration and holdout splits are still to be chosen. The scripts use `stage0.py`'s deterministic seed-0 FEVER sampling and the same IDs for the two 900-claim runs. Fresh outputs are ignored by Git so large logs and predictions do not enter a push accidentally.

## Existing results

On the same 900 FEVER dev claims:

| Run | Retrieval | Accuracy | FEVER | Stated Brier / ECE | Token Brier / ECE |
|---|---|---:|---:|---:|---:|
| Base 2B | titles | 0.459 | 0.340 | 0.515 / 0.513 | 0.329 / 0.308 |
| SFT 2B | oracle | 0.779 | 0.702 | 0.221 / 0.221 | 0.146 / 0.028 |
| SFT 2B | titles | 0.759 | 0.667 | 0.241 / 0.241 | 0.158 / 0.034 |

These calibration numbers are binary correctness scores: confidence in the selected verdict being right. The base run has stated confidence on 93.8% of rows and token confidence on 94.4%; SFT has both on 100%. They are replayed from saved predictions, not fresh inference. Recompute without a GPU:

```bash
python3 factcheck/src/eval_saved_runs.py factcheck/data/stage0/local_n900_titles.jsonl
python3 factcheck/src/eval_saved_runs.py factcheck/data/stage0/local+Qwen_Qwen3.5-2B_n6000_n900_titles.jsonl
```

The 5,895 distinct SFT training IDs and matched 1,200-row subset do not overlap the archived 900 dev IDs. This does not yet establish separate prompt-development, calibration and holdout sets. The SFT target wrote confidence `1.0`; its stated confidence is therefore a poor calibration target. The LoRA weights are excluded from Git, so saved JSONL alone cannot reproduce SFT inference.

## Fresh direct versus evidence-chain baseline on SoC

From the repo root, create `.venv` as described in the root README and download FEVER using [docs/data.md](docs/data.md). Inspect `sinfo` and adjust the scripts' GPU, RAM and time requests for the available partition. Build the index once; wait for success before running GPU jobs:

```bash
sbatch factcheck/slurm/build_fever_index.sbatch
squeue -u "$USER"
# After the index finishes successfully:
sbatch --export=ALL,N_CLAIMS=30 factcheck/experiments/direct/run.sbatch
sbatch --export=ALL,N_CLAIMS=30 factcheck/experiments/cot-chain/run.sbatch
# Then repeat each with N_CLAIMS=900 after checking the smoke-test logs.
```

Each job writes `predictions.jsonl`, `metrics.json`, `manifest.json` and `run.log` in its own `results/eval/<version>/job-<id>/`. Slurm also writes a small `slurm-<name>-<id>.out` at the root; both locations are ignored. For a 900-claim job, compare its ID set with `data/eval/fever-dev-900-ids.json` before using the result. Both prompts use the base `Qwen/Qwen3.5-2B`, title retrieval and greedy decoding. Direct outputs only the verdict JSON; `cot_chain` asks for a concise numbered evidence chain before the verdict in JSON. Its generation cap is larger, so compare tokens and cost as well as accuracy. These are prompt baselines, not SFT or GRPO training. A cited E-ID is only syntax; inspect whether it genuinely supports the verdict.

Use `sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,ExitCode` and the per-run `run.log` to diagnose jobs. The scripts expect submission from the repo root, raw FEVER, built index and a CUDA-ready `.venv`; they do not submit anything themselves.

## Code map

- `src/stage0.py`: retrieval, prompts, local/API inference and per-claim output (`--output` avoids resuming old archived runs).
- `src/eval_saved_runs.py`: offline evaluator with FEVER, evidence, recall, binary Brier/ECE, risk–coverage and evidence-in-context strata.
- `src/sft_probe.py`, `src/grpo_verifier.py`, `src/reward.py`: verifier training and reward.
- `src/grpo_multiturn.py`: multi-turn tool-calling experiment.
- `src/build_bm25_index.py`, `src/fever_retrieval.py`: FEVER index and search.
- `src/interfaces.py`, `src/verification_env.py`, `src/run_scaffold.py`: shared contracts and toy controller scaffold.
