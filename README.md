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

Run commands from the repository root after pushing this branch and pulling it on SoC. Check the [SoC Slurm guide](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm) and [cluster information](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm-info) on the cluster for current partitions, GPU types and limits; resource lines in the supplied batch files may need adjustment.

```bash
git fetch origin
git switch rl
git pull --ff-only origin rl
python3 -m venv .venv
source .venv/bin/activate
# Install a CUDA-compatible PyTorch build for the cluster GPU, then:
python -m pip install -r requirements.txt
sinfo
squeue -u "$USER"
```

Download FEVER and build its BM25/SQLite index using [the data guide](verifier/docs/data.md). After the raw FEVER files are present, submit the CPU indexing job from the repository root:

```bash
sbatch retrieval/build_fever_index.sbatch
# Check its returned job ID:
squeue -j <job-id>
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Wait for the index job to complete successfully and confirm `verifier/data/fever/pages.sqlite` and `verifier/data/fever/bm25_index/` exist. The batch file refuses to overwrite an existing index.

## Evaluate the verifier on SoC

Submit `verifier/evaluation/run.sbatch` with `sbatch` from the repository root. It requests Slurm resources and invokes `verifier/evaluation/run_eval.sh` inside the job. The shell script validates inputs, calls the shared `verifier/evaluation/eval_verifier.py`, scores predictions and records the run. Both prompt settings use the same FEVER dev IDs and title retrieval. Neither evaluation script trains a model.

`CHECKPOINTS` is optional: without it the evaluator loads the untrained `Qwen/Qwen3.5-2B`; with it, the evaluator loads the LoRA adapter and its base model from `adapter_config.json`. Adapter weights must be present on the cluster, since Git does not contain them. `THINKING=off` (default) uses direct JSON; `THINKING=on` asks for a concise evidence chain in the JSON `reasoning` field. This switch changes the prompt; Qwen's internal `enable_thinking` remains disabled in both modes. Choose a new `RUN_NAME` for every attempt.

First run 30-claim smoke tests, then the 900-claim comparison:

```bash
sbatch --export=ALL,N_CLAIMS=30,RUN_NAME=smoke1 verifier/evaluation/run.sbatch
sbatch --export=ALL,N_CLAIMS=30,RUN_NAME=smoke1,THINKING=on verifier/evaluation/run.sbatch
# After both smoke jobs pass:
sbatch --export=ALL,N_CLAIMS=900,RUN_NAME=run1 verifier/evaluation/run.sbatch
sbatch --export=ALL,N_CLAIMS=900,RUN_NAME=run1,THINKING=on verifier/evaluation/run.sbatch

# Example: evaluate a restored or newly trained SFT adapter with CoT prompting.
sbatch --export=ALL,N_CLAIMS=30,RUN_NAME=sft-smoke1,THINKING=on,CHECKPOINTS=verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints verifier/evaluation/run.sbatch
```

Each job writes predictions, metrics, a manifest and `logs/run.log` to `verifier/evaluation/results/<condition>/<run>/`. Conditions are `direct-baseline`, `cot-baseline`, `checkpoint-direct` and `checkpoint-cot`. The script refuses to overwrite an existing run. Use `squeue -u "$USER"`, then `sacct -j <job-id>` and the run log to inspect status. Slurm also writes an ignored root-level `slurm-*.out` fallback log. No cluster jobs have been submitted from this workspace.

Archived predictions can be evaluated without a GPU or raw FEVER:

```bash
python -m verifier.evaluation.score_predictions verifier/evaluation/results/archive/local_n900_titles.jsonl
python -m unittest verifier.tests.test_score_predictions verifier.tests.test_eval_verifier_reasoning
```

## Repeat a historical training configuration

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

Each SFT or GRPO configuration has a `train.py` launcher reconstructed from its saved `runs/run1/artifacts/meta.json`. It creates a new run and stores metadata, structured `train_log.json`, and checkpoints with that run. The checkpoint weights are ignored by Git. SFT needs raw FEVER and the index; GRPO additionally needs the SFT adapter weights restored or retrained.

```bash
python verifier/experiments/sft_qwen_3.5-2b_n6000/train.py --run-name run2 --seed 1
python verifier/experiments/grpo_qwen_3.5-2b_full_n1200_kl/train.py --run-name run3 --seed 2
```

The historical JSONL predictions are in `verifier/evaluation/results/archive/`; new results belong in a named evaluation run. Preserve seeds, input IDs, retrieval condition and resource use when comparing runs.
