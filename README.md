# Fact verifier experiments

This repository studies evidence-grounded claim verification on FEVER. A verifier reads a claim and retrieved sentences, then returns a JSON verdict and cited sentence IDs; some configurations also return a stated confidence. The `rl` branch contains historical SFT and GRPO runs plus direct, CoT, and brief reasoning baselines.

- **Current priorities & experiment checklist**: [`verifier/TODO.md`](verifier/TODO.md)
- **Research journal, experimental insights & error analyses**: `verifier/INSIGHT.md` (local / Git-ignored) and [`verifier/docs/`](verifier/docs/)

## Where to look

| Path | What it contains |
|---|---|
| [`run.sbatch`](run.sbatch) | One-command Slurm array for the base & SFT `cot_brief` retest |
| [`verifier/TODO.md`](verifier/TODO.md) | Current priorities and experiment checklist |
| [`verifier/docs/`](verifier/docs/) | Data guide, experiment results, architecture and background |
| [`verifier/data/train/`](verifier/data/train/) | Committed SFT and GRPO training examples |
| [`verifier/data/eval/`](verifier/data/eval/) | Frozen FEVER dev claim IDs |
| [`verifier/experiments/`](verifier/experiments/) | SFT and GRPO training configurations |
| [`verifier/evaluation/`](verifier/evaluation/) | Shared evaluator, Slurm launcher and `results/` |
| [`retrieval/`](retrieval/) | FEVER index and search code |
| [`core/`](core/) | Shared reward and interfaces |
| [`agent/`](agent/) | Search/verify loop and controller scaffold |
| [`utils/`](utils/) | Data preparation and experiment helpers |
| [`AGENTS.md`](AGENTS.md) | Conventions for collaborators and coding agents |

Detailed document map: [data and downloads](verifier/docs/data.md), [system plan](verifier/docs/plan.md), [prompted baseline](verifier/docs/prompted-baseline-results.md), [SFT](verifier/docs/sft-probe.md), [GRPO](verifier/docs/grpo-results.md), [agent traces](verifier/docs/trace-format.md), [models](verifier/docs/models.md), and [related work](verifier/docs/related-work.md).

## Current status

- **Retrieval**: Title retrieval reached 0.875 FEVER page recall@5.
- **Base model (untrained Qwen3.5-2B)**: On 900 dev claims with title retrieval, direct prompting scored 45.9% accuracy / 33.8% FEVER score; zero-shot evidence-chain (`cot_chain`) scored 60.8% accuracy / 51.1% FEVER score.
- **SFT (2B LoRA, 5,895 claims)**: Direct prompting scored 74.1% accuracy / 65.9% FEVER score. Zero-shot `cot_chain` scored 71.4% accuracy / 62.4% FEVER score, suffering from 512-token truncations and NEI recall drops.
- **Next experiment**: Run [`run.sbatch`](run.sbatch) on SoC to test `cot_brief` (concise <=50 words evidence reasoning) on both base and SFT models. See [`verifier/TODO.md`](verifier/TODO.md) for execution details.

Detailed metric comparisons, per-claim error cases, and calibration analyses are tracked in `verifier/INSIGHT.md`.

## SoC cluster setup & resources

Submit jobs from an SoC Slurm login node (`xlogin.comp.nus.edu.sg`), not `stu2`. Run commands from the repository root after syncing the branch:

```bash
cd "$HOME/rl-test"
git fetch origin
git switch rl
git pull --ff-only origin rl
```

### Partitions & GPU guidelines

| Partition | Max wall time | Usage |
|---|---:|---|
| `gpu` | 3 hours | Evaluation (~90 min requested) |
| `gpu-long` | 3 days | Training (~3–6 hours requested) |
| `normal` | 3 hours | CPU index build (~3 hours requested) |

- **Single GPU**: Evaluation and training scripts each run **one Python process on one GPU**. Requesting multiple GPUs does not speed up execution.
- **GPU types**: Use `a100-40:1` (default), `a100-80:1`, `h100-47:1`, or `h100-96:1` which support `bfloat16`. Avoid `nv` (which may allocate older cards like Titan RTX on `xgpe2`).
- **Disk quota**: `/tmp` on SoC login nodes has a strict write quota. Always export `TMPDIR="$HOME/.pip-tmp"` before installing packages or submitting jobs.

## Data preparation

The raw FEVER corpus and indexes require roughly 21 GB and are not committed to Git.

```bash
# 1. Prepare Python environment
mkdir -p "$HOME/.pip-tmp"
export TMPDIR="$HOME/.pip-tmp"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -r requirements.txt

# 2. Download FEVER raw data
mkdir -p verifier/data/fever
cd verifier/data/fever
curl -fLO https://fever.ai/download/fever/train.jsonl
curl -fLO https://fever.ai/download/fever/shared_task_dev.jsonl
curl -fLO https://fever.ai/download/fever/wiki-pages.zip
unzip -q wiki-pages.zip && rm wiki-pages.zip
cd ../../..
```

## Index preparation

Evaluation and SFT data preparation require `pages.sqlite` (Wikipedia sentences) and `bm25_index/` (BM25 search index).

1. Verify raw data:
```bash
for file in verifier/data/fever/train.jsonl verifier/data/fever/shared_task_dev.jsonl .venv/bin/python; do
  test -f "$file" && echo "OK $file" || echo "MISSING $file"
done
test -d verifier/data/fever/wiki-pages && echo 'OK wiki-pages' || echo 'MISSING wiki-pages'
```

2. Submit CPU indexing job (takes ~13 minutes on SoC `normal` partition):
```bash
export TMPDIR="$HOME/.pip-tmp"
job_id=$(sbatch --parsable --export=ALL retrieval/build_fever_index.sbatch)
echo "Index job: $job_id"
squeue -j "$job_id"
```

3. Confirm outputs after completion:
```bash
test -s verifier/data/fever/pages.sqlite && echo 'SQLite: OK' || echo 'SQLite: MISSING'
test -f verifier/data/fever/bm25_index/doc_ids.json && echo 'BM25: OK' || echo 'BM25: MISSING'
```

## Training on SoC

Submit training jobs from the repository root using `verifier/experiments/run_train.sbatch`. Artifacts (`checkpoints/`, `meta.json`, `logs/train_log.json`) are written under `verifier/experiments/<config>/runs/<run-name>/artifacts/`.

### 1. Standard no-CoT SFT (Qwen3.5-2B)

Trains on the committed 5,895 FEVER examples (seed 0, ~37 minutes on A100):

```bash
cd "$HOME/rl-test"
job_id=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 --time=03:00:00 --export=ALL,RUN_NAME=run2,SEED=0 verifier/experiments/run_train.sbatch)
echo "no-CoT SFT job: $job_id"
squeue -j "$job_id"
# Checkpoint: verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints/
```

### 2. CoT SFT Pilot (with SoCLaaS)

`utils/build_cot_sft_data.py` queries SoCLaaS (`qwen3.8:27b`) or Gemini on FEVER **train-only** oracle prompts to generate short verified reasoning chains:

```bash
# Generate teacher data (interactive API key entry, no GPU needed)
python -m utils.build_cot_sft_data prepare --provider soclaas --model qwen3.8:27b
python -m utils.build_cot_sft_data run --out verifier/data/train/sft/fever-oracle-cot-soclaas/pilot1
python -m utils.build_cot_sft_data collect --out verifier/data/train/sft/fever-oracle-cot-soclaas/pilot1

# Train CoT SFT pilot adapter on SoC
export TRAIN_SCRIPT=verifier/experiments/cot_sft_qwen_3.5-2b_oracle_pilot300/train.py
cot_sft_job=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 --time=03:00:00 --export=ALL,RUN_NAME=pilot1,SEED=0 verifier/experiments/run_train.sbatch)
unset TRAIN_SCRIPT
```

### 3. GRPO training

Requires an existing SFT adapter checkpoint:

```bash
export TRAIN_SCRIPT=verifier/experiments/grpo_qwen_3.5-2b_full_n1200/train.py
export SFT_CHECKPOINTS=verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints
sbatch --export=ALL,RUN_NAME=run2,SEED=1 verifier/experiments/run_train.sbatch
unset TRAIN_SCRIPT SFT_CHECKPOINTS
```

## Evaluation on SoC

Evaluation jobs load dev claims, query the BM25 index and SQLite cache, generate predictions, and score accuracy and FEVER metrics.

### 1. One-command array retest (`run.sbatch`)

Launches a two-task Slurm array to evaluate both the base model (task 0) and the SFT `run2` adapter (task 1) using the concise `cot_brief` prompt on the frozen 900 dev claims:

```bash
cd "$HOME/rl-test"
job=$(sbatch --parsable run.sbatch)
echo "Array job: $job (task 0: base, task 1: SFT)"
squeue -j "$job"
```
Results write to `verifier/evaluation/results/base-cot-brief-eval/prompt-retest1/` and `sft-cot-brief-eval/prompt-retest1/`.

### 2. Single evaluation run (`verifier/evaluation/run.sbatch`)

Submit single evaluation conditions via `verifier/evaluation/run.sbatch`:

```text
[RESULTS_ROOT=ROOT_PATH | RESULT_DIR=RUN_PATH] sbatch [--partition=PARTITION] [--gpus=GPU_TYPE:1] [--time=HH:MM:SS] \
  --export=ALL[,N_CLAIMS=COUNT][,RUN_NAME=NAME][,THINKING=on|off | ,PROMPT_STYLE=STYLE][,CHECKPOINTS=ADAPTER_PATH][,EVAL_CONDITION=NAME] \
  verifier/evaluation/run.sbatch
```

**Key options:**
- `N_CLAIMS`: Number of dev claims (default: 900; smoke test: 30).
- `PROMPT_STYLE`: `direct` (short JSON), `cot_chain` (explicit chain), `cot_brief` (<=50 words reasoning), or `cot_sft` (confidence-free CoT adapter).
- `CHECKPOINTS`: Path to adapter checkpoint directory (e.g. `verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints`). Omit to evaluate the untrained base model.
- `RUN_NAME`: Unique run identifier (e.g. `run1`, `run2`).
- `EVAL_CONDITION`: Result subdirectory name under `verifier/evaluation/results/`.

**Examples:**

```bash
# 30-claim smoke test
sbatch --export=ALL,N_CLAIMS=30,RUN_NAME=smoke1,THINKING=off verifier/evaluation/run.sbatch

# Base model: direct vs cot_chain (900 claims)
sbatch --partition=gpu --gpus=a100-40:1 --time=01:30:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run1,THINKING=off verifier/evaluation/run.sbatch
sbatch --partition=gpu --gpus=a100-40:1 --time=03:00:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run1,THINKING=on verifier/evaluation/run.sbatch

# SFT checkpoint: direct vs cot_chain (900 claims)
adapter=verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints
sbatch --partition=gpu --gpus=a100-40:1 --time=01:30:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run2,THINKING=off,CHECKPOINTS="$adapter",EVAL_CONDITION=sft-direct-eval verifier/evaluation/run.sbatch
sbatch --partition=gpu --gpus=a100-40:1 --time=03:00:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run2,THINKING=on,CHECKPOINTS="$adapter",EVAL_CONDITION=sft-cot-eval verifier/evaluation/run.sbatch
```

### 3. Inspecting results & offline scoring

Each run directory contains `predictions.jsonl`, `metrics.json`, and `manifest.json`:

```bash
cat verifier/evaluation/results/direct-baseline/run1/metrics.json
head -n 1 verifier/evaluation/results/direct-baseline/run1/predictions.jsonl
```

Re-score archived predictions offline without GPU:
```bash
python -m verifier.evaluation.score_predictions verifier/evaluation/results/archive/local_n900_titles.jsonl
python -m unittest verifier.tests.test_score_predictions verifier.tests.test_eval_verifier_reasoning
```

## Experiment catalogue

| Configuration in `verifier/experiments/` | Description |
|---|---|
| `sft_qwen_3.5-2b_n60` | 2B no-CoT SFT, 60-claim smoke run |
| `sft_qwen_3.5-2b_n6000` | 2B no-CoT SFT, 5,895 usable FEVER claims (main SFT baseline) |
| `cot_sft_qwen_3.5-2b_oracle_pilot300` | 2B CoT SFT on 227 accepted SoCLaaS explanations |
| `sft_qwen_3.5-2b_n1200_from_qwen_3.5-2b_n6000` | 2B SFT on the 1,200-claim subset matched to GRPO |
| `sft_qwen_3.5-4b_n60_qlora` | 4B QLoRA SFT, 60-claim smoke run |
| `sft_qwen_3.5-4b_n6000_qlora` | 4B QLoRA SFT, same 5,895 examples as 2B |
| `grpo_qwen_3.5-2b_full_n16_smoke` | 2B single-turn GRPO, 16-claim smoke run |
| `grpo_qwen_3.5-2b_full_n1200` | 2B single-turn GRPO from SFT, full reward, 1,200 claims, no KL |
| `grpo_qwen_3.5-2b_base_full_n1200` | 2B single-turn GRPO from base, full reward, 1,200 claims, KL beta 0.04 |
| `grpo_qwen_3.5-2b_full_n1200_kl` | 2B GRPO from SFT, 1,200-claim pool with reward-spread filtering, KL beta 0.04 |
| `grpo_qwen_3.5-2b_full_n1200_kl05` | Same filtered setup with KL beta 0.5 |
| `grpo_mt_qwen_3.5-2b_mt_base_full_n16` | Multi-turn GRPO from base with search/verify actions, 16-claim trial |
