# Fact verifier experiments

This repository studies evidence-grounded claim verification on FEVER. A verifier reads a claim and retrieved sentences, then returns a JSON verdict and cited sentence IDs; some configurations also return a stated confidence. The `rl` branch contains historical SFT and GRPO runs plus direct, CoT, and brief reasoning baselines.

- **Current priorities & experiment checklist**: [`verifier/TODO.md`](verifier/TODO.md)
- **Research journal, experimental insights & error analyses**: `verifier/INSIGHT.md` (local / Git-ignored) and [`verifier/docs/`](verifier/docs/)

## Where to look

| Path | What it contains |
|---|---|
| [`run.sbatch`](run.sbatch) | Two-job GRPO array for the current fixed 1,200-claim split comparison |
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
- **Base model (untrained Qwen3.5-2B)**: On the same 900 dev claims with title retrieval, direct prompting scored 45.9% accuracy / 33.8% FEVER; `cot_chain` scored 60.8% / 51.1%; `cot_brief` scored 61.8% / 48.6%. The direct run used a 160-token output cap, while CoT used 512; 46 direct outputs hit the cap and were invalid. Retest with matched caps before attributing the full gap to reasoning.
- **SFT (2B LoRA, 5,895 claims)**: Direct prompting scored 74.1% accuracy / 65.9% FEVER; `cot_chain` scored 71.4% / 62.4%; `cot_brief` scored 74.6% / 63.9%. Brief reasoning largely removed truncation, but still reduced NEI recall and FEVER score relative to direct prompting.
- **Matched 1024-token evaluation**: On 900 dev claims, Base `direct` / long CoT / brief CoT reached 47.6% / 62.8% / 61.8% accuracy. SFT `run2` reached 74.6% / 71.4% / 74.6%; its direct prompt had the highest FEVER score (66.3%). See the comparison below and the local, Git-ignored `verifier/INSIGHT.md`.
- **Next experiments**: For the split comparison submitted on 2026-10-15, the two SFT jobs were already running on SoC. Submit only the two GRPO jobs with [`run.sbatch`](run.sbatch), then evaluate the resulting adapters on an untouched holdout. See [`verifier/TODO.md`](verifier/TODO.md) for execution details.

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

Submit training jobs from the repository root using `verifier/experiments/run_train.sbatch`. Artifacts (`checkpoints/`, `meta.json`, `logs/train_log.json`) are written under `verifier/experiments/<config>/runs/<run-name>/artifacts/`. Configuration names follow `<method>-<output>-from-<start>-<model>-<data>[-<setting>]`; for example, `sft-direct-from-base-qwen3p5-2b-train5895`. `qwen3p5` means Qwen3.5; `train5895` counts actual training rows. `pilot300` means 300 teacher requests, of which 227 were accepted. `filtered1200` means reward-spread filtering of a 1,200-claim pool. `kl004` and `kl05` mean KL beta 0.04 and 0.5. Each `train.py` writes to its own configuration folder, and `RUN_NAME` chooses the repeat.

If your SoC checkout predates the training-directory rename, wait for running training jobs to finish, pull this branch, then move any Git-ignored run files (including adapter weights and new job logs) into their new paths:

```bash
python -m utils.migrate_training_runs          # preview
python -m utils.migrate_training_runs --apply  # move without overwriting
test -s verifier/experiments/sft-direct-from-base-qwen3p5-2b-train5895/runs/run2/artifacts/checkpoints/adapter_model.safetensors
```

The script skips files already present at the destination, reports any different-content conflicts, and never overwrites them. The current `run_train.sbatch` also translates former `TRAIN_SCRIPT`, `SFT_CHECKPOINTS` and `INIT_CHECKPOINTS` paths to their renamed configuration directories. This applies to newly submitted jobs after pulling the change; a job submitted before the launcher changed keeps its old Slurm script and must be resubmitted if it failed. Saved evaluation manifests retain their original checkpoint path in `original_checkpoints` and use the current path in `checkpoints`.

### 1. Standard no-CoT SFT (Qwen3.5-2B)

Trains on the committed 5,895 FEVER examples (seed 0; the recorded `run2` took 112 minutes):

```bash
cd "$HOME/rl-test"
job_id=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 --time=03:00:00 --export=ALL,RUN_NAME=run3,SEED=0 verifier/experiments/run_train.sbatch)
echo "no-CoT SFT job: $job_id"
squeue -j "$job_id"
# Checkpoint: verifier/experiments/sft-direct-from-base-qwen3p5-2b-train5895/runs/run3/artifacts/checkpoints/
```

### 2. CoT SFT Pilot (with SoCLaaS)

`utils/build_cot_sft_data.py` queries SoCLaaS (`qwen3.8:27b`) or Gemini on FEVER **train-only** oracle prompts to generate short verified reasoning chains:

```bash
# Generate teacher data (interactive API key entry, no GPU needed)
python -m utils.build_cot_sft_data prepare --provider soclaas --model qwen3.8:27b
python -m utils.build_cot_sft_data run --out verifier/data/train/sft/fever-oracle-cot-soclaas/pilot1
python -m utils.build_cot_sft_data collect --out verifier/data/train/sft/fever-oracle-cot-soclaas/pilot1

# Train the 227 accepted CoT SFT pilot examples on a SoC login node.
# This produces a CoT SFT adapter; it is not CoT GRPO.
test -f verifier/data/train/sft/fever-oracle-cot-soclaas/pilot1/cot_sft_labeled_prompts.jsonl
test -x .venv/bin/python
export TRAIN_SCRIPT=verifier/experiments/sft-cot-from-base-qwen3p5-2b-pilot300/train.py
cot_sft_job=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 --exclude=xgpe2 --time=03:00:00 --export=ALL,RUN_NAME=pilot1,SEED=0 verifier/experiments/run_train.sbatch)
unset TRAIN_SCRIPT
echo "CoT SFT pilot job: $cot_sft_job"
squeue -j "$cot_sft_job"
```

The committed `pilot1` data has 227 accepted examples. The separate `full1` teacher batch has 5,895 requested rows but, in the current local snapshot, only 2,132 responses and no collected training file. No CoT SFT or CoT GRPO checkpoint is recorded in this repository yet. Existing `grpo-direct-*` scripts train short direct JSON completions with a 96-token cap.

### 3. Fixed 4,695/1,200 split comparison

The fixed 1,200 claims in `verifier/data/train/sft/fever-oracle-matched-grpo-1200/` are excluded from the new 4,695-claim SFT set. The two sets have no claim-ID overlap and together contain all 5,895 prepared oracle prompts. Verify the committed split with `python -m utils.build_disjoint_sft_grpo_split --check`; its hashes and label counts are in `verifier/data/train/sft/fever-oracle-disjoint-4695/split_manifest.json`.

The two SFT arms have completed on SoC. The root `run.sbatch` now submits two independent tasks: evaluate the 1,200-row SFT continuation, and train GRPO from the disjoint 4,695-row SFT checkpoint. After pulling this branch, confirm both adapter directories contain `adapter_config.json` and `adapter_model.safetensors`, then submit from a SoC login node:

```bash
cd "$HOME/rl-test"
continued=verifier/experiments/sft-direct-from-sft-qwen3p5-2b-train1200/runs/split1-seed0-seen-sft/artifacts/checkpoints
disjoint=verifier/experiments/sft-direct-from-base-qwen3p5-2b-disjoint4695/runs/split1-seed0-unseen-sft/artifacts/checkpoints
if [[ -f "$continued/adapter_config.json" && -s "$continued/adapter_model.safetensors" && \
      -f "$disjoint/adapter_config.json" && -s "$disjoint/adapter_model.safetensors" ]]; then
  job=$(sbatch --parsable --export=ALL,RUN_TAG=split1-seed0 run.sbatch)
  echo "Follow-up array: $job"
  squeue -j "$job"
else
  echo "One or both renamed SFT checkpoint directories are incomplete"
fi
```

Task 0 evaluates the continued SFT adapter on the frozen 900 dev claims with title retrieval, a direct prompt and a 1024-token output cap. Results go to `verifier/evaluation/results/sft-continued1200-direct-cap1024/<RUN_TAG>/`. Task 1 starts GRPO from the disjoint 4,695-row SFT adapter on the same fixed 1,200 training rows, writing to `verifier/experiments/grpo-direct-from-sft-qwen3p5-2b-train1200-kl0/runs/<RUN_TAG>-unseen-grpo/`. Both tasks require the **new configuration paths** shown above. `RUN_TAG` defaults to `split1-seed0`; use `CONTINUED_SFT_RUN_NAME` or `DISJOINT_SFT_RUN_NAME` if either source run differs. Use a new `RUN_TAG` for a repeat so results are not overwritten. Use `verifier/experiments/submit_split_comparison.sh` only for a fresh four-job comparison.

| Job | Starting checkpoint | Training rows | Output run |
|---|---|---:|---|
| Seen → SFT | existing 5,895-row SFT `run2` | fixed 1,200 | `sft-direct-from-sft-qwen3p5-2b-train1200/runs/<tag>-seen-sft/` |
| Seen → GRPO | existing 5,895-row SFT `run2` | fixed 1,200 | `grpo-direct-from-sft-qwen3p5-2b-train1200-kl0/runs/<tag>-seen-grpo/` |
| Unseen → SFT | Qwen3.5-2B base | complementary 4,695 | `sft-direct-from-base-qwen3p5-2b-disjoint4695/runs/<tag>-unseen-sft/` |
| Unseen → GRPO | adapter from the preceding job | fixed 1,200 | `grpo-direct-from-sft-qwen3p5-2b-train1200-kl0/runs/<tag>-unseen-grpo/` |

The GRPO run saves its LoRA adapter to `artifacts/checkpoints/`, `meta.json` and `logs/train_log.json`; the SFT evaluation saves predictions, metrics, manifest and logs in its results folder. This array writes `slurm-split-followup-<job-id>_<task-id>.out` at the repo root. Check `squeue -j "$job"` and `sacct -j "$job" --format=JobID,State,ExitCode,Elapsed,MaxRSS`. The GRPO configuration here is **direct JSON** with a 96-token training completion cap, not CoT GRPO.

The disjoint path has 5,895 unique claim exposures across its two stages. The seen path deliberately trains on 1,200 claims twice. Compare each child against its own starting checkpoint on the same dev IDs; compare the two seen-claim continuations to distinguish GRPO from additional SFT. The seen and unseen paths have different first-stage SFT data sizes, so their raw scores alone do not isolate the effect of overlap.

## Evaluation on SoC

Evaluation jobs load dev claims, query the BM25 index and SQLite cache, generate predictions, and score accuracy and FEVER metrics.

### Completed 1024-token evaluation sweep

The six completed runs used the same frozen 900 dev claims, title retrieval, evaluator, and `MAX_NEW_TOKENS=1024`. Their results remain under `verifier/evaluation/results/`; the root `run.sbatch` now runs the split follow-up evaluation and GRPO training. For a new standalone evaluation, use `verifier/evaluation/run.sbatch` below.

| Task | Model | Prompt | Result condition |
|---:|---|---|---|
| 0 | Base | `direct` | `baseline-direct-cap1024` |
| 1 | Base | `cot_chain` (long CoT) | `baseline-cot-long-cap1024` |
| 2 | Base | `cot_brief` (brief CoT) | `baseline-cot-brief-cap1024` |
| 3 | SFT `run2` | `direct` | `sft-direct-cap1024` |
| 4 | SFT `run2` | `cot_chain` (long CoT) | `sft-cot-long-cap1024` |
| 5 | SFT `run2` | `cot_brief` (brief CoT) | `sft-cot-brief-cap1024` |

| Model | Direct Acc / FEVER | Long CoT Acc / FEVER | Brief CoT Acc / FEVER |
|---|---:|---:|---:|
| Base | 47.6% / 34.6% | **62.8% / 52.8%** | 61.8% / 48.6% |
| SFT `run2` | **74.6% / 66.3%** | 71.4% / 62.4% | 74.6% / 63.9% |

Result conditions use lowercase hyphens: `<model>-<prompt>-cap<output-tokens>`. The untrained model uses `baseline`; `cot-long` is the result label for `PROMPT_STYLE=cot_chain`, and `cot-brief` maps to `cot_brief`. Historical results also use `cap160` or `cap512`: `baseline-direct-cap160`, `baseline-cot-long-cap512`, `baseline-cot-brief-cap512`, `sft-direct-cap160`, `sft-cot-long-cap512`, and `sft-cot-brief-cap512`. Their manifests retain original condition/run names and original run paths; prediction rows and logs are unchanged.

Each completed run has predictions, metrics, manifest, and log under `verifier/evaluation/results/<result-condition>/<run-name>/`. At 1024 tokens, SFT long CoT had 44 capped outputs and 48 format failures, while Base direct had 130 format failures with only 3 capped outputs. These are different failure modes. Base direct/long and brief CoT were generated from different code revisions; brief CoT was also developed after inspecting these dev claims. Treat this as a development-set comparison and validate on an untouched holdout.

### Single evaluation run (`verifier/evaluation/run.sbatch`)

Submit single evaluation conditions via `verifier/evaluation/run.sbatch`:

```text
[RESULTS_ROOT=ROOT_PATH | RESULT_DIR=RUN_PATH] sbatch [--partition=PARTITION] [--gpus=GPU_TYPE:1] [--time=HH:MM:SS] \
  --export=ALL[,N_CLAIMS=COUNT][,RUN_NAME=NAME][,THINKING=on|off | ,PROMPT_STYLE=STYLE][,MAX_NEW_TOKENS=COUNT][,CHECKPOINTS=ADAPTER_PATH][,EVAL_CONDITION=NAME] \
  verifier/evaluation/run.sbatch
```

**Key options:**
- `N_CLAIMS`: Number of dev claims (default: 900; smoke test: 30).
- `PROMPT_STYLE`: `direct` (short JSON), `cot_chain` (explicit chain), `cot_brief` (<=50 words reasoning), or `cot_sft` (confidence-free CoT adapter).
- `MAX_NEW_TOKENS`: Optional positive output-token cap. Defaults: `direct=160`, `cot_chain=512`, `cot_brief=512`, `cot_sft=320`. This is not the model's total context window; set a distinct result directory for each cap.
- `CHECKPOINTS`: Path to adapter checkpoint directory (e.g. `verifier/experiments/sft-direct-from-base-qwen3p5-2b-train5895/runs/run2/artifacts/checkpoints`). Omit to evaluate the untrained base model.
- `RUN_NAME`: Unique run identifier (e.g. `run1`, `run2`).
- `EVAL_CONDITION`: Result subdirectory name under `verifier/evaluation/results/`.
- Without `EVAL_CONDITION`, the result name is `baseline-<prompt>-cap<tokens>` for the untrained model or `checkpoint-<prompt>-cap<tokens>` for an adapter. Set an explicit `sft-*` condition when the adapter is known to be SFT.

**Examples:**

```bash
# 30-claim smoke test
sbatch --export=ALL,N_CLAIMS=30,RUN_NAME=smoke2,THINKING=off verifier/evaluation/run.sbatch

# Base model: direct vs cot_chain (900 claims)
sbatch --partition=gpu --gpus=a100-40:1 --time=01:30:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run2,THINKING=off verifier/evaluation/run.sbatch
sbatch --partition=gpu --gpus=a100-40:1 --time=03:00:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run2,THINKING=on verifier/evaluation/run.sbatch

# SFT checkpoint: direct vs cot_chain (900 claims)
adapter=verifier/experiments/sft-direct-from-base-qwen3p5-2b-train5895/runs/run2/artifacts/checkpoints
sbatch --partition=gpu --gpus=a100-40:1 --time=01:30:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run2,THINKING=off,CHECKPOINTS="$adapter",EVAL_CONDITION=sft-direct-cap160 verifier/evaluation/run.sbatch
sbatch --partition=gpu --gpus=a100-40:1 --time=03:00:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run2,THINKING=on,CHECKPOINTS="$adapter",EVAL_CONDITION=sft-cot-long-cap512 verifier/evaluation/run.sbatch

```

### Inspecting results & offline scoring

Each run directory contains `predictions.jsonl`, `metrics.json`, and `manifest.json`:

```bash
cat verifier/evaluation/results/baseline-direct-cap160/run1/metrics.json
head -n 1 verifier/evaluation/results/baseline-direct-cap160/run1/predictions.jsonl
```

Re-score archived predictions offline without GPU:
```bash
python -m verifier.evaluation.score_predictions verifier/evaluation/results/archive/local_n900_titles.jsonl
python -m unittest verifier.tests.test_score_predictions verifier.tests.test_eval_verifier_reasoning
```

## Experiment catalogue

| Configuration in `verifier/experiments/` | Description |
|---|---|
| `sft-direct-from-base-qwen3p5-2b-smoke60` | 2B no-CoT SFT, 60-claim smoke run |
| `sft-direct-from-base-qwen3p5-2b-train5895` | 2B no-CoT SFT, 5,895 usable FEVER claims (main SFT baseline) |
| `sft-direct-from-base-qwen3p5-2b-disjoint4695` | 2B no-CoT SFT on the 4,695 claims outside the fixed GRPO set |
| `sft-direct-from-sft-qwen3p5-2b-train1200` | Continue the existing 5,895-claim SFT adapter on the fixed 1,200 claims |
| `sft-cot-from-base-qwen3p5-2b-pilot300` | 2B CoT SFT on 227 accepted SoCLaaS explanations |
| `sft-direct-from-base-qwen3p5-2b-matched1200` | 2B SFT on the 1,200-claim subset matched to GRPO |
| `sft-direct-from-base-qwen3p5-4b-smoke60-qlora` | 4B QLoRA SFT, 60-claim smoke run |
| `sft-direct-from-base-qwen3p5-4b-train5895-qlora` | 4B QLoRA SFT, same 5,895 examples as 2B |
| `grpo-direct-from-sft-qwen3p5-2b-smoke16-kl0` | 2B single-turn GRPO, 16-claim smoke run |
| `grpo-direct-from-sft-qwen3p5-2b-train1200-kl0` | 2B single-turn GRPO from SFT, full reward, 1,200 claims, no KL |
| `grpo-direct-from-base-qwen3p5-2b-train1200-kl004` | 2B single-turn GRPO from base, full reward, 1,200 claims, KL beta 0.04 |
| `grpo-direct-from-sft-qwen3p5-2b-filtered1200-kl004` | 2B GRPO from SFT, 1,200-claim pool with reward-spread filtering, KL beta 0.04 |
| `grpo-direct-from-sft-qwen3p5-2b-filtered1200-kl05` | Same filtered setup with KL beta 0.5 |
| `grpo-multiturn-from-base-qwen3p5-2b-smoke16-kl004` | Multi-turn GRPO from base with search/verify actions, 16-claim trial |
