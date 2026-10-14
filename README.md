# Fact verifier experiments

This repository studies evidence-grounded claim verification on FEVER. A verifier reads a claim and retrieved sentences, then returns a JSON verdict, cited sentence IDs, and confidence. The `rl` branch contains historical SFT and GRPO runs plus the next direct and evidence-chain baseline experiments. The raw FEVER corpus, retrieval index, and model weights are not committed.

## Where to look

| Path | What it contains |
|---|---|
| [`verifier/TODO.md`](verifier/TODO.md) | Current priorities and experiment checklist |
| [`verifier/docs/`](verifier/docs/) | Data guide, experiment results, architecture and background |
| [`verifier/data/train/`](verifier/data/train/) | Committed SFT and GRPO training examples |
| [`verifier/data/eval/`](verifier/data/eval/) | Frozen FEVER dev claim IDs |
| [`verifier/experiments/`](verifier/experiments/) | Historical SFT and GRPO training configurations |
| [`verifier/evaluation/`](verifier/evaluation/) | Shared evaluator, one Slurm launcher and `results/` |
| [`retrieval/`](retrieval/) | FEVER index and search code |
| [`core/`](core/) | Shared reward and interfaces |
| [`agent/`](agent/) | Search/verify loop and controller scaffold |
| [`utils/`](utils/) | Data preparation and experiment helpers |
| [`AGENTS.md`](AGENTS.md) | Conventions for collaborators and coding agents |

Detailed document map: [data and downloads](verifier/docs/data.md), [system plan](verifier/docs/plan.md), [prompted baseline](verifier/docs/prompted-baseline-results.md), [SFT](verifier/docs/sft-probe.md), [GRPO](verifier/docs/grpo-results.md), [agent traces](verifier/docs/trace-format.md), [models](verifier/docs/models.md), and [related work](verifier/docs/related-work.md).

## Current status (2026-10-14)

- Title retrieval reached 0.875 FEVER page recall@5. The untrained Qwen3.5-2B verifier reached 0.459 accuracy / 0.340 FEVER score on 900 dev claims with title retrieval.
- A 2B SFT LoRA trained on about 6,000 claims reached 0.759 accuracy / 0.667 FEVER score with title retrieval. Its training examples and saved predictions are committed; its LoRA weights are not.
- Archived GRPO runs have not surpassed that SFT accuracy, and verdict-token calibration worsened. The next comparison is direct JSON versus zero-shot evidence-chain prompting on identical claims and retrieval. These new jobs have not yet been run.

A committed SFT `labeled_prompts.jsonl` row has a claim/evidence `prompt` and a gold-answer JSON `completion`. The completion is the training target; it does not teach search steps or give a calibrated confidence target. Historical `meta.json` paths preserve the original machine's layout, while current scripts use this repository's paths.

## Set up on the SoC cluster

Submit jobs from an SoC Slurm login node (`xlogin.comp.nus.edu.sg`), not `stu2`. Run commands from the repository root after pushing this branch and pulling it on SoC. Check the [SoC Slurm guide](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm) and [cluster information](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm-info) on the cluster for current partitions, GPU types and limits; resource lines in the supplied batch files may need adjustment.

Run the shell commands below **after SSHing into SoC**, where `$HOME` refers to your cluster home (for this account, `/home/c/chaoch`). On the Mac, `$HOME` refers to the Mac home (for example, `/Users/ludwigchao`); its `.venv`, downloaded data and temporary directory are separate. The Slurm job must be able to see the cluster paths used for the repo, data and `TMPDIR`.

```bash
cd "$HOME/rl-test"
git fetch origin
git switch rl
git pull --ff-only origin rl
sinfo
squeue -u "$USER"
```

## Data preparation

The repository commits prepared SFT/GRPO examples and archived predictions, but not raw FEVER, its retrieval index, model weights or the Python environment. Prepare these once on a filesystem visible to both the login and compute nodes. The raw FEVER files plus index need roughly 21 GB; model downloads and Python packages need additional space. Check your quota before starting. Creating `.venv` inside a Slurm job at the same repo path does not avoid a home-directory quota; a node-local temporary environment would disappear after the job.

From the repository root, create the environment and install dependencies. The batch files activate this existing `.venv` but do not create it. On SoC `xlogin1`, a direct write test found that `/tmp` hit a write quota after about 10 MiB even though `df -h /tmp` showed free filesystem space. Set `TMPDIR` to a writable directory in the cluster home before installing; keep it exported when submitting Slurm jobs. If that directory also reports `Disk quota exceeded`, check its quota instead of deleting data based on `df` alone.

```bash
mkdir -p "$HOME/.pip-tmp"
export TMPDIR="$HOME/.pip-tmp"
python3 -m venv .venv
source .venv/bin/activate
# Install a CUDA-compatible PyTorch build for the cluster GPU, then:
python -m pip install --no-cache-dir -r requirements.txt
```

Download only FEVER for the baseline; [the data guide](verifier/docs/data.md) has SciFact and AVeriTeC instructions for later experiments. The zip and extracted Wikipedia pages coexist until the successful `unzip` finishes, then the command removes the zip to recover space.

```bash
mkdir -p verifier/data/fever
cd verifier/data/fever
curl -fLO https://fever.ai/download/fever/train.jsonl
curl -fLO https://fever.ai/download/fever/shared_task_dev.jsonl
curl -fLO https://fever.ai/download/fever/wiki-pages.zip
unzip -q wiki-pages.zip && rm wiki-pages.zip
cd ../../..
```

If `wiki-pages/` was already extracted by an earlier attempt but the 1.6 GB zip is still present, confirm the expected 109 `wiki-*.jsonl` files, then remove the redundant archive from the repository root:

```bash
find verifier/data/fever/wiki-pages -maxdepth 1 -name 'wiki-*.jsonl' -type f | wc -l
rm verifier/data/fever/wiki-pages.zip
```

## Index preparation

Evaluation and SFT data preparation require two local retrieval files that Git does not contain: `pages.sqlite` (Wikipedia sentences) and `bm25_index/` (the BM25 search index). Build them once from FEVER's `wiki-pages/` on SoC. Before submitting, check the raw inputs and Python environment from the cluster repo root:

```bash
cd "$HOME/rl-test"
for file in verifier/data/fever/train.jsonl verifier/data/fever/shared_task_dev.jsonl .venv/bin/python; do
  if test -f "$file"; then echo "OK $file"; else echo "MISSING $file"; fi
done
test -d verifier/data/fever/wiki-pages && echo 'OK wiki-pages' || echo 'MISSING wiki-pages'
```

When all four checks say `OK`, submit the CPU indexing job. It requests eight CPUs, 32 GB RAM and up to six hours; check the current SoC limits before submission. The job writes `verifier/data/fever/pages.sqlite` and `verifier/data/fever/bm25_index/` and refuses to overwrite either existing output.

```bash
mkdir -p "$HOME/.pip-tmp"
export TMPDIR="$HOME/.pip-tmp"
job_id=$(sbatch --parsable --export=ALL retrieval/build_fever_index.sbatch)
echo "Index job: $job_id"
squeue -j "$job_id"
```

After it leaves the queue, check `sacct -j "$job_id" --format=JobID,State,ExitCode,Elapsed,MaxRSS` for `COMPLETED` and exit code `0:0`, then check both outputs:

```bash
test -s verifier/data/fever/pages.sqlite && echo 'SQLite: OK' || echo 'SQLite: MISSING'
test -f verifier/data/fever/bm25_index/doc_ids.json && echo 'BM25: OK' || echo 'BM25: MISSING'
```

The job log is `slurm-fever-index-<job-id>.out` in the repo root; read it if the job fails. A failed build can leave partial files, so inspect those and the log before retrying. Once both outputs exist and the job completed successfully, proceed to training or evaluation.

## Train the verifier on SoC

The first reproducible training run is the 2B SFT configuration. It trains a LoRA adapter from FEVER examples; it does not run evaluation. GRPO configurations need an existing SFT adapter, including base-start GRPO (which reads the SFT LoRA shape). See the [experiment catalogue](#experiment-catalogue) below for the configuration names.

Submit from the SoC repo root with `sbatch`, not by running `train.py` on the login node. `run_train.sbatch` uses one GPU, four CPUs, 32 GB RAM and six hours by default; check `sinfo` and adjust these Slurm requests for the cluster and configuration. Set `TRAIN_SCRIPT` to a configuration's `train.py`; the default is `sft_qwen_3.5-2b_n6000`. `RUN_NAME` identifies a fresh attempt and `SEED` defaults to 0.

```bash
cd "$HOME/rl-test"
sbatch --export=ALL,RUN_NAME=run2,SEED=1 verifier/experiments/run_train.sbatch
# For a specific GRPO configuration after an SFT adapter exists:
export TRAIN_SCRIPT=verifier/experiments/grpo_qwen_3.5-2b_full_n1200/train.py
export SFT_CHECKPOINTS=verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints
sbatch --export=ALL,RUN_NAME=run2,SEED=1 verifier/experiments/run_train.sbatch
unset TRAIN_SCRIPT SFT_CHECKPOINTS
```

By default, a run writes `meta.json`, `logs/train_log.json`, and `checkpoints/` under `verifier/experiments/<configuration>/runs/<run-name>/artifacts/`. `checkpoints/` contains the LoRA adapter for later evaluation; the Slurm terminal log is `slurm-train-verifier-<job-id>.out` in the submission directory. The script refuses to reuse an existing artifacts directory. To put all training runs on a different shared filesystem, set `OUTPUT_ROOT` to the **experiments root** before submission; the configuration and run subdirectories are still added automatically. For example:

```bash
export OUTPUT_ROOT="$HOME/rl-training"
sbatch --export=ALL,RUN_NAME=run2,SEED=1 verifier/experiments/run_train.sbatch
# Then inspect: $HOME/rl-training/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/
unset OUTPUT_ROOT
```

For GRPO using an SFT adapter outside the repo, set `SFT_CHECKPOINTS` to its checkpoint directory; single-turn GRPO can also set `TRAIN_DATA` to a directory containing `labeled_prompts.jsonl`. `OUTPUT_ROOT` changes run artifacts, not the FEVER source data. Inspect a completed training run with `sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS` and, for the default 2B SFT example, `cat verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/meta.json` and `cat verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/logs/train_log.json`. If it fails before those files appear, read `slurm-train-verifier-<job-id>.out`.

## Evaluate the verifier on SoC

Submit `verifier/evaluation/run.sbatch` with `sbatch` from the SoC repo root. It requests one GPU, four CPUs, 24 GB RAM and 90 minutes, then invokes `run_eval.sh` inside the job. Evaluation needs `shared_task_dev.jsonl`, `pages.sqlite`, a completed `bm25_index/` and `.venv`. Run these checks first; plain `test -f` is silent even on success:

```bash
cd "$HOME/rl-test"
for file in verifier/data/fever/shared_task_dev.jsonl verifier/data/fever/pages.sqlite .venv/bin/python; do
  if test -f "$file"; then echo "OK $file"; else echo "MISSING $file"; fi
done
test -f verifier/data/fever/bm25_index/doc_ids.json && echo 'OK BM25 index' || echo 'MISSING BM25 index'
```

The job calls the shared evaluator, scores predictions and records the run. Both prompt settings use the same FEVER dev IDs and title retrieval; neither trains a model. The job sets `TMPDIR` to a directory in cluster home if none is exported, avoiding the observed `/tmp` write quota.

`CHECKPOINTS` is optional: without it the evaluator loads the untrained `Qwen/Qwen3.5-2B`; with it, the evaluator loads the LoRA adapter and its base model from `adapter_config.json`. Adapter weights must be present on the cluster, since Git does not contain them. `THINKING=off` (default) uses direct JSON; `THINKING=on` asks for a concise evidence chain in the JSON `reasoning` field. This switch changes the prompt; Qwen's internal `enable_thinking` remains disabled in both modes. Choose a new `RUN_NAME` for every attempt.

First run a 30-claim smoke test, then the 900-claim comparison. Change the run name if it already exists:

```bash
sbatch --export=ALL,N_CLAIMS=30,RUN_NAME=smoke1 verifier/evaluation/run.sbatch
sbatch --export=ALL,N_CLAIMS=30,RUN_NAME=smoke1,THINKING=on verifier/evaluation/run.sbatch
# After both smoke jobs pass:
sbatch --export=ALL,N_CLAIMS=900,RUN_NAME=run1 verifier/evaluation/run.sbatch
sbatch --export=ALL,N_CLAIMS=900,RUN_NAME=run1,THINKING=on verifier/evaluation/run.sbatch

# Example: evaluate a restored or newly trained SFT adapter with CoT prompting.
sbatch --export=ALL,N_CLAIMS=30,RUN_NAME=sft-smoke1,THINKING=on,CHECKPOINTS=verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints verifier/evaluation/run.sbatch
```

Each job writes `predictions.jsonl` (one claim per line), `metrics.json` (aggregate scores), `manifest.json` (settings and paths), and `logs/run.log` under `verifier/evaluation/results/<condition>/<run>/`. Conditions are `direct-baseline`, `cot-baseline`, `checkpoint-direct` and `checkpoint-cot`. For example, read a completed direct run with:

```bash
cat verifier/evaluation/results/direct-baseline/run1/metrics.json
head -n 1 verifier/evaluation/results/direct-baseline/run1/predictions.jsonl
cat verifier/evaluation/results/direct-baseline/run1/manifest.json
```

To save evaluation runs elsewhere, export `RESULTS_ROOT` as the **results root** before `sbatch`; the condition and run subdirectories are added automatically. The path must be visible and writable on the compute node. For example:

```bash
export RESULTS_ROOT="$HOME/rl-evaluation"
sbatch --export=ALL,N_CLAIMS=900,RUN_NAME=run2,THINKING=off verifier/evaluation/run.sbatch
# Then inspect: $HOME/rl-evaluation/direct-baseline/run2/metrics.json
unset RESULTS_ROOT
```

The script refuses to overwrite an existing run. Use `squeue -j <job-id>`, then `sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS` and the run log to inspect status. Slurm also writes `slurm-eval-verifier-<job-id>.out` in the submission directory; read it first if a job fails before creating its result directory. No cluster jobs have been submitted from this workspace.

Archived predictions can be evaluated without a GPU or raw FEVER:

```bash
python -m verifier.evaluation.score_predictions verifier/evaluation/results/archive/local_n900_titles.jsonl
python -m unittest verifier.tests.test_score_predictions verifier.tests.test_eval_verifier_reasoning
```

## Experiment catalogue

The training configuration names are historical. `n16`, `n60` and `n1200` are requested training claim counts; `n6000` produced 5,895 usable rows. `full` means the complete reward formula, not a full dataset. `base` means GRPO starts from the base model; otherwise it starts from an SFT model or adapter. `kl` uses KL beta 0.04, while `kl05` uses 0.5. `qlora` means 4-bit QLoRA, `mt` means multi-turn search/verification, and `smoke` marks a small trial.

| Configuration in `verifier/experiments/` | What it ran |
|---|---|
| `sft_qwen_3.5-2b_n60` | 2B SFT, 60-claim smoke run |
| `sft_qwen_3.5-2b_n6000` | 2B SFT, 5,895 usable FEVER claims |
| `sft_qwen_3.5-2b_n1200_from_qwen_3.5-2b_n6000` | 2B SFT on the 1,200-claim subset matched to GRPO |
| `sft_qwen_3.5-4b_n60_qlora` | 4B QLoRA SFT, 60-claim smoke run |
| `sft_qwen_3.5-4b_n6000_qlora` | 4B QLoRA SFT, same 5,895 examples as 2B |
| `grpo_qwen_3.5-2b_full_n16_smoke` | 2B single-turn GRPO, 16-claim smoke run |
| `grpo_qwen_3.5-2b_full_n1200` | 2B single-turn GRPO from SFT, full reward, 1,200 claims, no KL penalty |
| `grpo_qwen_3.5-2b_base_full_n1200` | 2B single-turn GRPO from base, full reward, 1,200 claims, KL beta 0.04 |
| `grpo_qwen_3.5-2b_full_n1200_kl` | 2B GRPO from SFT, 1,200-claim pool with reward-spread filtering, KL beta 0.04; `run2` is seed 1 |
| `grpo_qwen_3.5-2b_full_n1200_kl05` | Same filtered setup with KL beta 0.5 |
| `grpo_mt_qwen_3.5-2b_mt_base_full_n16` | Multi-turn GRPO from base with search/verify actions, 16-claim trial |

Each SFT or GRPO configuration has a `train.py` launcher reconstructed from its saved `runs/run1/artifacts/meta.json`. Its checkpoint weights are ignored by Git. SFT needs raw FEVER and the index; GRPO additionally needs the SFT adapter weights restored or retrained. See [Train the verifier on SoC](#train-the-verifier-on-soc) for commands and output locations.

The historical JSONL predictions are in `verifier/evaluation/results/archive/`; new results belong in a named evaluation run. Preserve seeds, input IDs, retrieval condition and resource use when comparing runs.
