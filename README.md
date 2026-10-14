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

## SoC GPU and time limits

Check the live cluster configuration before submitting; the following is a snapshot from 2026-10-15, not a reservation or a promise of free GPUs:

```bash
sinfo -o "%P %l %a"                  # partition, maximum wall time, availability
sinfo -N -p gpu,gpu-long -o "%N %P %t %G"  # node, partition, state, configured GPUs
```

| Partition | Maximum wall time on 2026-10-15 | Used here |
|---|---:|---|
| `gpu` | 3 hours | Evaluation: 90 minutes requested |
| `gpu-long` | 3 days | Training: 6 hours requested |
| `normal` | 3 hours | CPU index build: 3 hours requested |
| `long` | 3 days | Optional for CPU jobs requiring more than 3 hours |

The wall-time limit belongs to the **partition**, not to a GPU model. A job's `--time` must fit the selected partition; requesting a shorter time does not imply it will actually run that long. `gpu-long` contains fewer eligible nodes than `gpu`, so a job may wait longer. The `test` partition also reported a 3-day limit but is not selected by these scripts.

The cluster advertised these GPU resource types in `sinfo`; the number shown is the **configured cards per node**, not the number currently free or available to this account:

| Slurm GPU type | Configured cards per node seen | For this repository |
|---|---:|---|
| `a100-40` | 1–2 | Default for evaluation and training; supports `bfloat16` |
| `a100-80` | 1 | Alternative when a run needs more GPU memory |
| `h100-47` | 4 | Compatible alternative |
| `h100-96` | 2 | Compatible alternative |
| `h200-141` | 4 | Compatible alternative; verify partition and access first |
| `nv` | 1–2 | Unspecified model; includes the `xgpe2` Titan RTX, so avoid for the current `bfloat16` code |

The current eval and train launchers each start **one Python process that uses one GPU**. Set the GPU count to `1`; requesting two cards will not make these scripts run twice as fast. Use a typed request such as `--gpus=a100-40:1` or `--gpus=h100-47:1` to avoid an incompatible node. A command-line `sbatch` option overrides its `#SBATCH` default; for example, `sbatch --partition=gpu --gpus=h100-47:1 --time=01:30:00 --export=ALL,N_CLAIMS=30,RUN_NAME=h100-smoke verifier/evaluation/run.sbatch`. `--mem=24G` in the eval script is **CPU memory**, separate from the GPU's 40/47/80 GB VRAM. Check pending jobs with `squeue -u "$USER"`; inspect the job's log and `sacct` after it finishes. [Slurm `sbatch` options](https://slurm.schedmd.com/sbatch.html) and [NVIDIA `bfloat16` support](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html) explain the resource syntax and hardware requirement.

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

When all four checks say `OK`, submit the CPU indexing job. It requests the `normal` partition, eight CPUs, 32 GB RAM and up to three hours. The job writes `verifier/data/fever/pages.sqlite` and `verifier/data/fever/bm25_index/` and refuses to overwrite either existing output.

```bash
mkdir -p "$HOME/.pip-tmp"
export TMPDIR="$HOME/.pip-tmp"
job_id=$(sbatch --parsable --export=ALL retrieval/build_fever_index.sbatch)
echo "Index job: $job_id"
squeue -j "$job_id"
```

After it leaves the queue, check `sacct -j "$job_id" --format=JobID,State,ExitCode,Elapsed,MaxRSS` for `COMPLETED` and exit code `0:0`, then check both outputs. If you opened a new shell and no longer have `job_id`, use `sacct -u "$USER" --starttime today --format=JobID,JobName,State,ExitCode,Elapsed` to find the `fever-index` job. Run these commands from the repo root:

```bash
test -s verifier/data/fever/pages.sqlite && echo 'SQLite: OK' || echo 'SQLite: MISSING'
test -f verifier/data/fever/bm25_index/doc_ids.json && echo 'BM25: OK' || echo 'BM25: MISSING'
```

The job log is `slurm-fever-index-<job-id>.out` in the repo root. The angle-bracket text is a placeholder, not something to type literally in a shell. To read the most recent index log without knowing its job ID:

```bash
ls -t slurm-fever-index-*.out | head -n 1 | xargs tail -n 5
```

The successful SoC run on 2026-10-15 processed 5,396,106 pages and 3,114,367 terms in 13 minutes 3 seconds (job 946444, `COMPLETED`, exit code `0:0`). This is an observed runtime, not a guarantee for another node or queue; the `done:` line in the log confirms the indexer reached its end. A failed build can leave partial files, so inspect those and the log before retrying. Once both outputs exist and the job completed successfully, proceed to training or evaluation.

## Train the verifier on SoC

The first reproducible training run is **no-CoT SFT** for Qwen3.5-2B. Its committed FEVER examples contain a prompt and a target JSON with `verdict`, `evidence`, and `confidence`, but no `reasoning`; 5,895 usable rows were saved for the 6,000-claim configuration. It trains a LoRA adapter and does not run evaluation. The historical seed-0 run took 2,190 seconds (about 37 minutes) for training and reported 5.18 GB peak GPU memory; setup, queueing, and hardware can change the total time. A separate **CoT SFT data preparation** workflow below can request short reasoning from SoCLaaS or Gemini; a CoT trainer is not yet implemented. Running `THINKING=on` with the no-CoT SFT adapter tests a *prompt change on the same checkpoint*, not CoT SFT. GRPO configurations need an existing SFT adapter, including base-start GRPO (which reads the SFT LoRA shape). See the [experiment catalogue](#experiment-catalogue) below for the configuration names.

### Prepare CoT SFT data with SoCLaaS

`utils/build_cot_sft_data.py` uses the committed **FEVER train only** oracle prompts. It sends the shown E-numbered evidence and fixed gold answer to the teacher, asks only for a short explanation, then checks the returned relation and citations. The accepted completion contains `reasoning`, the original gold `verdict`, and the original gold `evidence`; it omits the old uncalibrated `confidence: 1.0`. The generated file is named `cot_sft_labeled_prompts.jsonl` under a `fever-oracle-cot-<provider>/<run>/` directory so it cannot be mistaken for the no-CoT `labeled_prompts.jsonl`. This is a separate dataset, and the current no-CoT trainer/evaluator still needs an explicit CoT configuration before training or scoring it.

SoCLaaS is SoC's free, OpenAI-compatible gateway. Its [access guide](https://dochub.comp.nus.edu.sg/cf/guides/soclaas/access) and [quickstart](https://dochub.comp.nus.edu.sg/cf/guides/soclaas/quickstart) specify `https://soclaas-api.comp.nus.edu.sg/v1` and `GET /models`; the model list is specific to your key. Use a local computer connected to the NUS network if it can reach the gateway. The source prompts are already in Git, so no SoC GPU or server copy is needed for teacher generation. `models` and `run` privately prompt for the key in an interactive terminal; the script never puts it in command arguments, environment variables, the repo or logs. It only sends the key to the verified SoCLaaS host. Keep it private and rotate it through the SoCLaaS portal if exposed. The default pilot selects 100 claims per class, seed 0.

```bash
python -m utils.build_cot_sft_data models
python -m utils.build_cot_sft_data prepare --provider soclaas --model qwen3.8:27b
python -m utils.build_cot_sft_data run --out verifier/data/train/sft/fever-oracle-cot-soclaas/pilot1 --max-requests 10
python -m utils.build_cot_sft_data run --out verifier/data/train/sft/fever-oracle-cot-soclaas/pilot1
python -m utils.build_cot_sft_data collect --out verifier/data/train/sft/fever-oracle-cot-soclaas/pilot1
```

`qwen3.8:27b` is the current recommended reasoning teacher in the SoCLaaS [model catalogue](https://dochub.comp.nus.edu.sg/cf/guides/soclaas/models); use it only if `models` lists it for your key. `prepare` and `collect` work offline. `run` resumes from saved successful responses and stops on HTTP 429 or another API error. The default delay is one second between SoCLaaS requests (10 seconds for Gemini), and `--delay` overrides it. The published [default limits](https://dochub.comp.nus.edu.sg/cf/guides/soclaas/usage-limits) are 90 requests/minute and virtual US$80/day or US$800/month; these are accounting limits, **not real charges**, and your key can have different limits. A 10-request run gives actual token usage and latency before expanding. After all 300 pilot responses arrive, inspect `summary.json` and validate the explanations before preparing all 5,895 rows with `prepare --provider soclaas --model YOUR_MODEL_ID --per-class 0 --out verifier/data/train/sft/fever-oracle-cot-soclaas/full1`, then `run` and `collect` with that directory. Replace `YOUR_MODEL_ID` with an actual allowed ID. `requests.jsonl`, API responses and rejected rows are locally ignored; `manifest.json`, `summary.json` and accepted `cot_sft_labeled_prompts.jsonl` can be reviewed for inclusion in Git. The validator catches wrong E IDs and label relations, but cannot prove every explanation is semantically faithful. Only sync an accepted dataset to SoC for training.

The earlier Gemini route remains available with `prepare --provider gemini --model gemini-3.8-flash --out verifier/data/train/sft/fever-oracle-cot-gemini/pilot2`, but SoCLaaS is the preferred teacher for this SoC experiment. Run either provider on an interactive machine that can reach its endpoint; do not put an API key into an `sbatch --export` argument.

Submit from the SoC repo root with `sbatch`, not by running `train.py` on the login node. `run_train.sbatch` uses `gpu-long`, one A100 40 GB GPU, four CPUs, 32 GB CPU RAM and a six-hour limit by default. If a training run fits within three hours, you can select `--partition=gpu --time=03:00:00` at submission; if it needs more GPU memory, select a different supported GPU type. See [SoC GPU and time limits](#soc-gpu-and-time-limits). Set `TRAIN_SCRIPT` to a configuration's `train.py`; the default is `sft_qwen_3.5-2b_n6000`. `RUN_NAME` identifies a fresh attempt and `SEED` defaults to 0.

```bash
cd "$HOME/rl-test"
job_id=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 --time=03:00:00 --export=ALL,RUN_NAME=run2,SEED=0 verifier/experiments/run_train.sbatch)
echo "no-CoT SFT job: $job_id"
squeue -j "$job_id"
# Adapter after successful completion:
# verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints/

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

For GRPO using an SFT adapter outside the repo, set `SFT_CHECKPOINTS` to its checkpoint directory; single-turn GRPO can also set `TRAIN_DATA` to a directory containing `labeled_prompts.jsonl`. `OUTPUT_ROOT` changes run artifacts, not the FEVER source data. The no-CoT SFT example uses seed 0 to match the committed 5,895-row dataset and the historical configuration. Inspect a completed training run with `sacct -j "$job_id" --format=JobID,State,ExitCode,Elapsed,MaxRSS` and `cat verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/meta.json`. Its learning log is under the same artifacts directory at `logs/train_log.json`; if it fails before those files appear, read `slurm-train-verifier-<job-id>.out` with the actual job ID substituted.

## Evaluate the verifier on SoC

Submit `verifier/evaluation/run.sbatch` with `sbatch` from the SoC repo root. It requests the `gpu` partition, one A100 40 GB GPU, four CPUs, 24 GB CPU RAM and 90 minutes, then invokes `run_eval.sh` inside the job. The model loader uses `bfloat16`; see [SoC GPU and time limits](#soc-gpu-and-time-limits) for compatible types and overrides. On 2026-10-15, a GPU check on `xgpe2` returned `nvidia-smi: No devices found` despite Slurm allocating a GPU, so avoid that node until SoC repairs it. Evaluation needs `shared_task_dev.jsonl`, `pages.sqlite`, a completed `bm25_index/` and `.venv`. Run these checks first; plain `test -f` is silent even on success:

```bash
cd "$HOME/rl-test"
for file in verifier/data/fever/shared_task_dev.jsonl verifier/data/fever/pages.sqlite .venv/bin/python; do
  if test -f "$file"; then echo "OK $file"; else echo "MISSING $file"; fi
done
test -f verifier/data/fever/bm25_index/doc_ids.json && echo 'OK BM25 index' || echo 'MISSING BM25 index'
```

The job calls the shared evaluator, scores predictions and records the run. Both prompt settings use the same FEVER dev IDs and title retrieval; neither trains a model. The job sets `TMPDIR` to a directory in cluster home if none is exported, avoiding the observed `/tmp` write quota.

`CHECKPOINTS` is optional: without it the evaluator loads the untrained `Qwen/Qwen3.5-2B`; with it, the evaluator loads the LoRA adapter and its base model from `adapter_config.json`. Adapter weights must be present on the cluster, since Git does not contain them. `THINKING=off` (default) requests `verdict`, `evidence`, and `confidence` in JSON. `THINKING=on` requests the same fields plus an evidence-based `reasoning` string in that **same JSON object**. This switch changes the prompt; Qwen's internal `enable_thinking` remains disabled in both modes. There is no constrained JSON decoding, so format compliance is measured rather than guaranteed. The current `format_rate` checks that JSON parses and that verdict, evidence, and confidence are valid; it does not require `reasoning` or prove that the output contains nothing outside the JSON. For the evidence-chain run, also inspect `reasoning_present_rate` and `reasoning_valid_citations_rate` in `metrics.json`.

Submission template (`[...]` marks optional text and `|` separates alternatives; these symbols and uppercase path names are **notation, not literal shell arguments**):

```text
[RESULTS_ROOT=ROOT_PATH | RESULT_DIR=RUN_PATH] sbatch [--partition=PARTITION] [--gpus=GPU_TYPE:1] [--time=HH:MM:SS] --export=ALL[,N_CLAIMS=COUNT][,RUN_NAME=NAME][,THINKING=on|off][,CHECKPOINTS=ADAPTER_PATH] verifier/evaluation/run.sbatch
```

`N_CLAIMS` defaults to 900, `RUN_NAME` to `job-<Slurm ID>`, and `THINKING` to `off`. `RESULTS_ROOT` moves the parent results directory but retains `<condition>/<run-name>/`; `RESULT_DIR` specifies the **exact new directory for one run** and does not add those subdirectories. They cannot be set together. Relative paths are resolved from the repo root; absolute paths must be writable and visible on the compute node. The job refuses to overwrite an existing result directory. The condition is still inferred from `THINKING` and `CHECKPOINTS` and saved in `manifest.json`, even when `RESULT_DIR` places the files elsewhere.

Results inside the Git checkout can be committed selectively. The default result directory keeps `predictions.jsonl`, `metrics.json`, and `manifest.json` available to Git while ignoring `logs/*.log`; check file sizes before committing a large run or a custom `RESULT_DIR`.

Run a small smoke test before a full comparison when changing the model or GPU setup. Use a new run name if its result directory already exists. The direct and evidence-chain jobs have separate condition directories, so both can use `RUN_NAME=run1` and be submitted together. Slurm will queue either job until its requested GPU is available.

The direct 30-claim smoke run on a working GPU allocation completed its claim loop in 51 seconds on 2026-10-15; this excludes queueing and model load time. It reached 0.400 label accuracy and 0.233 FEVER score. These 30-claim metrics are a pipeline check, not the full baseline result. The optional `causal_conv1d` and `flash-linear-attention` kernels were absent, so Transformers used its slower reference implementations. A linear extrapolation gives roughly 25 minutes for 900 direct claims of similar length, but actual runtime will vary. Evidence-chain output may take longer, so the example grants it the `gpu` partition's full three-hour limit.

```bash
sbatch --export=ALL,N_CLAIMS=30,RUN_NAME=smoke1,THINKING=off verifier/evaluation/run.sbatch

# Submit both 900-claim conditions; each requests one A100 40 GB GPU.
direct_job=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 --time=01:30:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run1,THINKING=off verifier/evaluation/run.sbatch)
cot_job=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 --time=03:00:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run1,THINKING=on verifier/evaluation/run.sbatch)
echo "direct job: $direct_job; evidence-chain job: $cot_job"
squeue -j "$direct_job,$cot_job"

# After the no-CoT SFT job is COMPLETED with exit code 0:0, test both prompts
# against the same LoRA adapter. Use new run names if these paths already exist.
adapter=verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints
sft_direct_job=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 --time=01:30:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run2,THINKING=off,CHECKPOINTS="$adapter" verifier/evaluation/run.sbatch)
sft_cot_job=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 --time=03:00:00 --export=ALL,N_CLAIMS=900,RUN_NAME=run2,THINKING=on,CHECKPOINTS="$adapter" verifier/evaluation/run.sbatch)
echo "SFT direct job: $sft_direct_job; SFT evidence-chain job: $sft_cot_job"
# Results: verifier/evaluation/results/checkpoint-direct/run2/
#          verifier/evaluation/results/checkpoint-cot/run2/
```

Each job writes `predictions.jsonl` (one claim per line), `metrics.json` (aggregate scores), `manifest.json` (settings and paths), and `logs/run.log` under `verifier/evaluation/results/<condition>/<run>/`. Conditions are `direct-baseline`, `cot-baseline`, `checkpoint-direct` and `checkpoint-cot`. For example, read a completed direct run with:

```bash
cat verifier/evaluation/results/direct-baseline/run1/metrics.json
head -n 1 verifier/evaluation/results/direct-baseline/run1/predictions.jsonl
cat verifier/evaluation/results/direct-baseline/run1/manifest.json
```

To save evaluation runs elsewhere, choose one of these copyable examples. `RESULTS_ROOT` relocates all condition/run directories; `RESULT_DIR` targets a single directory, including a custom experiment name:

```bash
RESULTS_ROOT="$HOME/rl-evaluation" sbatch --export=ALL,N_CLAIMS=900,RUN_NAME=run2,THINKING=off verifier/evaluation/run.sbatch
# Writes $HOME/rl-evaluation/direct-baseline/run2/metrics.json

RESULT_DIR="$HOME/rl-evaluation/my-prompt-study/run2" sbatch --export=ALL,N_CLAIMS=900,RUN_NAME=run2,THINKING=on verifier/evaluation/run.sbatch
# Writes $HOME/rl-evaluation/my-prompt-study/run2/metrics.json
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
| `sft_qwen_3.5-2b_n60` | 2B no-CoT SFT, 60-claim smoke run |
| `sft_qwen_3.5-2b_n6000` | 2B no-CoT SFT, 5,895 usable FEVER claims |
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
