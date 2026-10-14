# Fact verification with RL

This repository studies evidence-grounded claim verification on FEVER. Given a claim and retrieved sentences, the verifier returns a JSON verdict (`SUPPORTED`, `REFUTED`, or `NOT_ENOUGH_INFO`), cited sentence IDs, and confidence. The `rl` branch records SFT and GRPO experiments plus an agent search loop. Raw FEVER files, the search index, and LoRA weights are not committed.

## Status (2026-10-14)

- FEVER title retrieval has 0.875 page recall@5. An untrained Qwen3.5-2B reached 0.459 accuracy / 0.340 FEVER on 900 dev claims with title retrieval.
- The 2B SFT LoRA trained on about 6,000 claims reached 0.759 accuracy / 0.667 FEVER with title retrieval. Its **training examples and saved predictions are committed; adapter weights are not**.
- Archived GRPO runs have not exceeded that SFT accuracy, and verdict-token calibration worsened. See [GRPO results](docs/grpo-results.md).
- The next comparison is fresh direct JSON versus zero-shot evidence-chain prompting on identical claims and retrieval. Neither run has been executed in the new layout.

## Document map

| File | Purpose |
|---|---|
| [TODO.md](TODO.md) | Next steps and experiment priorities |
| [AGENTS.md](AGENTS.md) | Conventions for collaborators and coding agents |
| [docs/plan.md](docs/plan.md) | Architecture, team roles and experiment decisions |
| [docs/data.md](docs/data.md) | Data inventory, raw downloads and index build |
| [docs/stage0-results.md](docs/stage0-results.md) | Prompted baselines and retrieval results |
| [docs/sft-probe.md](docs/sft-probe.md) | SFT design and historical results |
| [docs/grpo-results.md](docs/grpo-results.md) | GRPO runs and limitations |
| [docs/trace-format.md](docs/trace-format.md) | Agent trace schema |
| [docs/models.md](docs/models.md) | Model choice and hardware notes |
| [docs/related-work.md](docs/related-work.md) | Literature notes |

## Layout and data availability

```text
data/train/sft/             committed prompt/completion JSONL for historical SFT
 data/train/grpo/            committed GRPO prompt sets and disagreement samples
 data/eval/                  frozen FEVER dev IDs
 data/stage0/, data/traces/  archived prediction and agent traces
 data/fever/                 raw FEVER and BM25/SQLite index; download/build locally
experiments/direct-baseline/, experiments/cot-baseline/
                            Slurm scripts and run-specific logs
experiments/<train-config>/train.py
experiments/<train-config>/runs/<run>/artifacts/
                            metadata, structured training logs, checkpoint directory
results/eval/<version>/<run>/
                            fresh predictions, metrics and run manifest
src/                        evaluation, retrieval, reward and agent code
src/utils/                  shared path and GRPO utilities
```

`train.jsonl` has a `prompt` containing the claim and numbered evidence plus a `completion` JSON answer. That completion is the SFT target; it does not teach search steps or a calibrated confidence label. The archived training metadata and adapter configs live with each experiment, but `adapter_model.safetensors` files are absent. Historical `meta.json` paths name the original machine and are retained as provenance. [Data guide](docs/data.md) lists download commands and expected sizes.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
# On SoC, install a PyTorch build compatible with the allocated GPU.
python -m pip install -r requirements.txt
```

Download FEVER and build the retrieval index using [docs/data.md](docs/data.md). The baseline requires `data/fever/shared_task_dev.jsonl`, `pages.sqlite`, and `bm25_index/` in addition to the committed dev ID manifest. Run from the repo root. Optional API keys belong in the ignored `.env` file.

## Run direct and evidence-chain baselines on SoC

First inspect available GPU partitions and limits with `sinfo`. Submit from the repo root; adjust the `#SBATCH` resource lines in each script to the available partition if needed. Use distinct run names to keep every attempt.

```bash
N_CLAIMS=30 RUN_NAME=smoke1 sbatch --export=ALL,N_CLAIMS,RUN_NAME experiments/direct-baseline/run.sbatch
N_CLAIMS=30 RUN_NAME=smoke1 sbatch --export=ALL,N_CLAIMS,RUN_NAME experiments/cot-baseline/run.sbatch
# After inspecting both smoke results:
N_CLAIMS=900 RUN_NAME=run1 sbatch --export=ALL,N_CLAIMS,RUN_NAME experiments/direct-baseline/run.sbatch
N_CLAIMS=900 RUN_NAME=run1 sbatch --export=ALL,N_CLAIMS,RUN_NAME experiments/cot-baseline/run.sbatch
```

Each job uses the same model, seed and title retrieval; only the prompt style changes. The script refuses to overwrite an existing run. Logs go to `experiments/<version>/runs/<run>/artifacts/logs/run.log`; outputs go to `results/eval/<version>/<run>/`. Slurm's root `slurm-*.out` is a temporary scheduler fallback and is ignored. Check `squeue -u "$USER"` for status. These runs require a GPU and the locally built FEVER index.

Archived predictions can be scored without a GPU or raw FEVER:

```bash
python3 src/eval_saved_runs.py data/stage0/local_n900_titles.jsonl
python3 src/test_eval_saved_runs.py
```

## Repeat training experiments

Each historical SFT or GRPO configuration has `experiments/<config>/train.py`, reconstructed from its recorded `runs/run1/artifacts/meta.json`. For example:

```bash
python experiments/sft_qwen_3.5-2b_n6000/train.py --run-name run2 --seed 1
python experiments/grpo_qwen_3.5-2b_full_n1200_kl/train.py --run-name run3 --seed 2
```

The SFT run needs raw FEVER and the index. GRPO also needs the SFT adapter weights, which must be restored or retrained. Training writes model weights under that run's `artifacts/checkpoints/` and a structured `artifacts/logs/train_log.json`; large weights are ignored by Git. Use new run names and record the actual dataset, seed, resources and evaluation in the run metadata before comparing results.
