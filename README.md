# Fact verifier experiments

This repository studies evidence-grounded claim verification on FEVER. A verifier reads a claim and retrieved sentences, then returns a JSON verdict and cited sentence IDs; some configurations also return a stated confidence. The `rl` branch contains historical SFT and GRPO runs plus direct, CoT, and brief reasoning baselines.

- **Current priorities & experiment checklist**: [`verifier/TODO.md`](verifier/TODO.md)
- **Research journal, experimental insights & error analyses**: `verifier/INSIGHT.md` (local / Git-ignored) and [`verifier/docs/`](verifier/docs/)

---

## Project Map

| Path | What it contains |
|---|---|
| [`run.sbatch`](run.sbatch) | Two-task Slurm array: evaluates 5,500 Native CoT SFT (Task 0) and Unseen-Arm CoT GRPO 4695 (Task 1) |
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

Detailed document map: [data and downloads](verifier/docs/data.md), [system plan](verifier/docs/plan.md), [prompted baseline](verifier/docs/prompted-baseline-results.md), [SFT](verifier/docs/sft-probe.md), [CoT teacher rejection audit](verifier/docs/cot-teacher-rejection-audit.md), [GRPO](verifier/docs/grpo-results.md), [agent traces](verifier/docs/trace-format.md), [models](verifier/docs/models.md), and [related work](verifier/docs/related-work.md).

---

## Repository Workflow & Pipeline

The core repository workflow consists of data and retrieval index preparation, verifier training, and multi-condition evaluation.

### 1. Data & Retrieval Index Preparation

- **Python Environment**: Requires Python 3.10+ and PyTorch with CUDA/MPS/CPU support (`requirements.txt`).
- **FEVER Raw Data**: Download train/dev JSONL files and Wikipedia dump (~21 GB uncompressed):
  ```bash
  mkdir -p verifier/data/fever && cd verifier/data/fever
  curl -fLO https://fever.ai/download/fever/train.jsonl
  curl -fLO https://fever.ai/download/fever/shared_task_dev.jsonl
  curl -fLO https://fever.ai/download/fever/wiki-pages.zip
  unzip -q wiki-pages.zip && rm wiki-pages.zip
  cd ../../..
  ```
- **SQLite Database & BM25 Index**: Builds `pages.sqlite` (Wikipedia sentences) and `bm25_index/` (BM25 term index):
  ```bash
  python -m retrieval.build_fever_index --db verifier/data/fever/pages.sqlite --bm25-dir verifier/data/fever/bm25_index
  ```

### 2. Verifier Training & Methodology Taxonomy

The verifier fine-tunes `Qwen/Qwen3.5-2B` using parameter-efficient LoRA (rank 16) across two distinct training paradigms and two output representations:

#### Method Taxonomy: What Do `sft-direct`, `sft-cot`, and `grpo-cot` Mean?

| Training Method | Description | Input $\to$ Output Flow | Strengths & Trade-offs |
|---|---|---|---|
| **`sft-direct`** | **Direct Supervised Fine-Tuning** (no reasoning tokens). Fine-tunes the base LLM directly on claim and evidence text to emit the final verdict and evidence citations immediately. | $\text{Claim} + \text{Evidence} \to \text{Verdict} + \text{Evidence IDs}$ | **Pros**: Extremely fast inference, low compute, high raw classification accuracy (76.1% on 5500), and strong refutation recall.<br>**Cons**: Narrow attention bottleneck; decoder cannot cite evidence after blindly committing to a verdict; higher false abstention (19.8%). |
| **`sft-cot`** | **Chain-of-Thought Supervised Fine-Tuning** (teacher imitation). Supervised training on teacher-generated explanations (from Qwen-27B / Gemini) prior to emitting the verdict. | $\text{Claim} + \text{Evidence} \to \text{Reasoning} \to \text{Verdict} + \text{Evidence IDs}$ | **Pros**: Provides step-by-step explanatory provenance and sentence matching.<br>**Cons**: Pure supervised imitation suffers from forced rationalization, reducing NEI recall (drops to 50.3%) without reinforcement learning. |
| **`grpo-cot`** | **Group Relative Policy Optimization with Native Thinking** (RLVR). Bootstrapped from an SFT checkpoint, the model explores reasoning trajectories via group rollouts ($G=8$) optimized under verifiable rule-based rewards (verdict harm penalty, evidence F1, format compliance). | $\text{Claim} + \text{Evidence} \to \langle\text{think}\rangle \dots \langle/\text{think}\rangle \to \text{Evidence} + \text{Verdict}$ | **Pros**: Peak repository performance (**76.2% Accuracy, 73.7% Evidence F1** on Unseen Arm); reverses NEI regression (75.0% recall); grounds reasoning via verifiable rewards.<br>**Cons**: Longer training time (~6–10h on A100). |
| **`dr-grpo` / `dapo`** | **Advanced RLVR Variants** (addressing group variance $\sigma_R$ and length bias). | Group-relative advantage with $\sigma_R$ removal (`dr-grpo`) and dynamic sampling of non-unanimous batches (`dapo`). | Eliminates gradient explosion/vanishing on unanimous batches ($\sigma_R = 0$) and strips out token-level length biases. |

#### Dual Output Formats: JSON vs Native Tag Ordering

1. **Path 1 — Standard JSON Format** (Baseline & Historical Compatibility):
   ```json
   {"verdict": "SUPPORTED", "evidence": ["E1", "E3"], "confidence": 1.0}
   ```
   Maintained across `fever-oracle-disjoint-4500` and `fever-oracle-matched-grpo-1000` to preserve continuity with historical benchmarks.

2. **Path 2 — Native Tag Format** (Evidence-First Causal Attention):
   ```text
   <think>
   [Step-by-step evidence verification]
   </think>
   <evidence>E1, E3</evidence>
   <verdict>SUPPORTED</verdict>
   ```
   Maintained across `fever-oracle-tagged-4500` and `fever-oracle-tagged-grpo-1000`. Citing `<evidence>` *before* `<verdict>` allows the causal decoder's self-attention heads to condition the verdict directly on the cited evidence, eliminating JSON syntax token tax.

- **Outputs**: LoRA weights, metadata (`meta.json`), and step logs (`train_log.json`) save under `verifier/experiments/<config>/runs/<run>/artifacts/`.

### 3. Verifier Evaluation & Scoring

Evaluates claim verification on frozen dev claim sets (default 900 claims with title retrieval):
- **Prompt Styles**:
  - `direct`: Output concise JSON directly.
  - `cot_brief`: Output concise evidence reasoning (<=50 words) within the JSON object.
  - `cot_chain` (`cot_long`): Output full evidence reasoning chain within the JSON object.
  - `cot_native`: Output native model thinking tokens (`<think>...</think>`), followed by JSON.
- **Evaluation Results Directory Layout (`verifier/evaluation/results/`)**:
  Evaluation results follow a hierarchical structure mirroring training configurations:
  ```text
  verifier/evaluation/results/
  ├── <eval_model>/               # e.g., base-qwen3p5-2b or sft-direct-from-base-qwen3p5-2b-5500-samples
  │   └── <condition>/            # <prompt_style>-cap<tokens>, e.g., direct-cap1024, cot-brief-cap1024
  │       └── <run>/              # Run directory (e.g. run1, job-12345) containing:
  │           ├── manifest.json   # Reproducibility metadata, command, commit, and file paths
  │           ├── metrics.json    # Accuracy, FEVER score, format compliance, evidence F1, calibration ECE
  │           ├── predictions.jsonl # Claim-by-claim predictions and citation outputs
  │           └── logs/run.log    # Evaluation terminal log
  └── archive/                    # Historical repository of raw per-claim .jsonl prediction files
                                  # from early project baselines and probes (including local_n900_titles.jsonl,
                                  # from which the frozen 900 dev set IDs were established).
  ```
- **Metrics Tracked**:
  - `accuracy`: Overall verdict classification accuracy.
  - `fever_score`: Strict FEVER metric (requires both correct verdict and full gold evidence citation).
  - `format_rate`: JSON parse and field adherence rate.
  - `evidence_f1`: Evidence sentence retrieval F1 on non-NEI claims.
  - `calibration`: Brier score and Expected Calibration Error (ECE) for verdict-token probability and stated text confidence.
- **Offline Scoring Script**:
  ```bash
  python -m verifier.evaluation.score_predictions <predictions.jsonl>
  python -m unittest verifier.tests.test_score_predictions
  ```

### 4. Experiment Catalogue

| Configuration in `verifier/experiments/` | Description |
|---|---|
| `sft-direct-from-base-qwen3p5-2b-5895-samples` | 2B no-CoT SFT, 5,895 usable FEVER claims (main SFT baseline) |
| `sft-direct-from-base-qwen3p5-2b-disjoint-4695-samples` | 2B SFT on the 4,695 disjoint split (unseen GRPO starting checkpoint) |
| `sft-direct-from-sft5895-qwen3p5-2b-1200-samples` | 2B continued SFT on the 1,200-claim second-stage set (seen SFT control) |
| `grpo-cot-from-sft5895-qwen3p5-2b-no-calibration-kl0-1200-samples` | 2B native-thinking CoT GRPO from SFT 5895 (seen arm) |
| `grpo-cot-from-sft4695-qwen3p5-2b-no-calibration-kl0-1200-samples` | 2B native-thinking CoT GRPO from SFT 4695 (unseen arm) |
| `sft-cot-from-base-qwen3p5-2b-5500-samples` | 2B CoT SFT on 5,500 validated and repaired Qwen-27B teacher explanations |
| `sft-direct-from-base-qwen3p5-2b-5500-samples` | 2B direct SFT baseline on the exact same 5,500 claims |
| `sft-direct-from-base-qwen3p5-4b-qlora-5895-samples` | 4B QLoRA SFT on 5,895 examples |

---

## Running on NUS SoC Cluster

If local GPU resources are unavailable (e.g. developing on a laptop or Mac), push your code changes to GitHub and use NUS SoC compute cluster GPU nodes for training and evaluation.

### 1. Prerequisites & Access Setup

1. **Create an SoC account**: Register your account at [MySoC Account Creation](https://mysoc.nus.edu.sg/app/newacct/).
2. **Connect to NUS VPN**: If working outside the NUS network, connect via [NUS VPN](https://nusit.nus.edu.sg/services/wifi_and_network/vpn/).
3. **SSH into the SoC cluster**:
   - **Direct login to Slurm compute nodes** (recommended; required for `sbatch`):
     ```bash
     ssh <soc_account>@xlogin.comp.nus.edu.sg
     ```
   - **Via jump host** (if accessing student general servers):
     ```bash
     ssh -J <soc_account>@stujump.comp.nus.edu.sg <soc_account>@stu.comp.nus.edu.sg
     ```
   > [!IMPORTANT]
   > Always submit Slurm jobs (`sbatch`) from an SoC login node (`xlogin.comp.nus.edu.sg`), not from `stu` or `stu2`.

### 2. Local-to-Cluster Development Workflow

Standard development and experiment cycle:

1. **Commit and push locally**:
   ```bash
   git add .
   git commit -m "commit message"
   git push origin rl
   ```
2. **SSH into the SoC login node**:
   ```bash
   ssh <soc_account>@xlogin.comp.nus.edu.sg
   ```
3. **Navigate to the cluster repository and pull updates**:
   ```bash
   cd "$HOME/rl-test"
   git pull
   ```
4. **Launch GPU jobs via Slurm** (see commands below).

### 3. SoC Partitions & GPU Guidelines

| Partition | Max Wall Time | Usage |
|---|---:|---|
| `gpu` | 3 hours | Evaluation jobs (~90 min requested) |
| `gpu-long` | 3 days | Training jobs (~3–6 hours requested) |
| `normal` | 3 hours | CPU index build (~13 minutes observed) |

- **Single GPU**: Evaluation and training scripts each run **one Python process on one GPU** (`--gpus=1`). Requesting multiple cards will not speed up execution.
- **GPU types**: Use `a100-40:1` (default), `a100-80:1`, `h100-47:1`, or `h100-96:1` which support `bfloat16`. Avoid `nv` (which may allocate older cards like Titan RTX on `xgpe2`).
- **Disk quota**: `/tmp` on SoC login nodes has a strict write quota. Always export `TMPDIR="$HOME/.pip-tmp"` before installing packages or submitting jobs:
  ```bash
  mkdir -p "$HOME/.pip-tmp"
  export TMPDIR="$HOME/.pip-tmp"
  ```

### 4. Cluster Job Submission

#### A. Submit CPU index build job
```bash
export TMPDIR="$HOME/.pip-tmp"
job_id=$(sbatch --parsable --export=ALL retrieval/build_fever_index.sbatch)
echo "Index Job: $job_id"
squeue -j "$job_id"
```

#### B. Submit one-command Slurm array evaluation sweep (`run.sbatch`)
Executes Task 0 (evaluate 5,500 Native CoT SFT) and Task 1 (evaluate Unseen-Arm 4,695 CoT GRPO):
```bash
job=$(sbatch --parsable run.sbatch)
echo "CoT Evaluation Sweep Array Job: $job"
squeue -j "$job"
```

#### C. Submit single model training job (`verifier/experiments/run_train.sbatch`)
```bash
# Example: train standard 5,895-row no-CoT SFT (<3 hours)
job_id=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 --time=03:00:00 \
  --export=ALL,RUN_NAME=run3,SEED=0 verifier/experiments/run_train.sbatch)
```

#### D. Submit single model evaluation job (`verifier/evaluation/run.sbatch`)
```bash
# Example: evaluate SFT checkpoint under cot_brief prompt
adapter=verifier/experiments/sft-direct-from-base-qwen3p5-2b-5500-samples/runs/run1/artifacts/checkpoints
sbatch --partition=gpu --gpus=a100-40:1 --time=01:30:00 \
  --export=ALL,N_CLAIMS=900,RUN_NAME=run1,PROMPT_STYLE=cot_brief,CHECKPOINTS="$adapter" \
  verifier/evaluation/run.sbatch
```

#### E. Job monitoring and status checks
```bash
squeue -u "$USER"                                                               # Check running/queued jobs
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS         # Check completed job status and resource usage
cat slurm-split-followup-<job_id>_0.out                                         # Inspect terminal output log
```
