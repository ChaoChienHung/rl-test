#!/usr/bin/env bash
# Submit the 4,500 SFT and 1,000 GRPO Dual-Path pipeline on SoC with Slurm job dependencies.
# Also submits the 3-condition Base Qwen3.5-2B native-tag baseline evaluation.
set -euo pipefail

cd "$(dirname "$0")/../.."
command -v sbatch >/dev/null || { echo 'sbatch is required; run this on a SoC login node' >&2; exit 2; }
test -x .venv/bin/python || { echo 'Missing .venv/bin/python' >&2; exit 2; }

# Verify splits
.venv/bin/python -m utils.build_4500_1000_splits --check

run_tag="${RUN_TAG:-run1}"
[[ "$run_tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
  echo 'RUN_TAG must use letters, digits, underscore or dash' >&2; exit 2;
}

# Training configurations
p1_sft_config="sft-direct-from-base-qwen3p5-2b-disjoint-4500-samples"
p2_sft_config="sft-direct-from-base-qwen3p5-2b-tagged-4500-samples"
p1_grpo_config="grpo-cot-from-sft4500-qwen3p5-2b-no-calibration-kl0-1000-samples"
p2_grpo_config="grpo-cot-from-sft4500tagged-qwen3p5-2b-no-calibration-kl0-1000-samples"

p1_sft_cp="verifier/experiments/${p1_sft_config}/runs/${run_tag}-json-sft/artifacts/checkpoints"
p2_sft_cp="verifier/experiments/${p2_sft_config}/runs/${run_tag}-tag-sft/artifacts/checkpoints"

p1_grpo_data="verifier/data/train/sft/fever-oracle-matched-grpo-1000"
p2_grpo_data="verifier/data/train/sft/fever-oracle-tagged-grpo-1000"

echo "=== Submitting Dual-Path 4500 SFT & 1000 GRPO Pipeline ==="

# 1. Path 1: 4,500 JSON Direct SFT
p1_sft_job=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 \
  --exclude=xgpe2 --time=03:00:00 \
  --export=ALL,TRAIN_SCRIPT="verifier/experiments/${p1_sft_config}/train.py",RUN_NAME="${run_tag}-json-sft",SEED=0,INIT_CHECKPOINTS=,SFT_CHECKPOINTS=,TRAIN_DATA= \
  verifier/experiments/run_train.sbatch)

# 2. Path 2: 4,500 Tag Direct SFT
p2_sft_job=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 \
  --exclude=xgpe2 --time=03:00:00 \
  --export=ALL,TRAIN_SCRIPT="verifier/experiments/${p2_sft_config}/train.py",RUN_NAME="${run_tag}-tag-sft",SEED=0,INIT_CHECKPOINTS=,SFT_CHECKPOINTS=,TRAIN_DATA= \
  verifier/experiments/run_train.sbatch)

# 3. Path 1 GRPO (after Path 1 SFT completes)
p1_grpo_job=$(sbatch --parsable --partition=gpu-long --gpus=a100-40:1 \
  --exclude=xgpe2 --time=1-00:00:00 --dependency="afterok:$p1_sft_job" \
  --export=ALL,TRAIN_SCRIPT="verifier/experiments/${p1_grpo_config}/train.py",RUN_NAME="${run_tag}-json-grpo",SEED=0,SFT_CHECKPOINTS="$p1_sft_cp",TRAIN_DATA="$p1_grpo_data",INIT_CHECKPOINTS= \
  verifier/experiments/run_train.sbatch)

# 4. Path 2 Tag GRPO (after Path 2 Tag SFT completes)
p2_grpo_job=$(sbatch --parsable --partition=gpu-long --gpus=a100-40:1 \
  --exclude=xgpe2 --time=1-00:00:00 --dependency="afterok:$p2_sft_job" \
  --export=ALL,TRAIN_SCRIPT="verifier/experiments/${p2_grpo_config}/train.py",RUN_NAME="${run_tag}-tag-grpo",SEED=0,SFT_CHECKPOINTS="$p2_sft_cp",TRAIN_DATA="$p2_grpo_data",INIT_CHECKPOINTS= \
  verifier/experiments/run_train.sbatch)

# 5. Base Model Native-Tag Baseline Evaluation (Tasks 0-2: direct_tag, cot_brief_tag, cot_chain_tag)
base_eval_job=$(sbatch --parsable --array=0-2%3 run.sbatch)

printf 'Path 1 (JSON) 4,500 SFT:     %s\n' "$p1_sft_job"
printf 'Path 2 (Tag)  4,500 SFT:     %s\n' "$p2_sft_job"
printf 'Path 1 (JSON) 1,000 GRPO:    %s (afterok:%s)\n' "$p1_grpo_job" "$p1_sft_job"
printf 'Path 2 (Tag)  1,000 GRPO:    %s (afterok:%s)\n' "$p2_grpo_job" "$p2_sft_job"
printf 'Base Native-Tag Baseline:    %s (array 0-2)\n' "$base_eval_job"

squeue -j "$p1_sft_job,$p2_sft_job,$p1_grpo_job,$p2_grpo_job,$base_eval_job"
