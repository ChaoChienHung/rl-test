#!/usr/bin/env bash
# Submit the Fast-Slow Dual-Process Training & Evaluation Pipeline on SoC with Slurm job dependencies.
#
# Pipeline steps:
#   1. Train Fast-Slow SFT on 4,500 claims (Task 7, partition=gpu, ~2 hours)
#   2. Train Fast-Slow Native-Thinking GRPO with Cognitive Self-Correction Reward on 1,000 claims (Task 8, partition=gpu-long, ~7 hours, dependency=afterok:SFT)
#   3. Evaluate Fast-Slow GRPO on FEVER 900 dev claims (Task 9, partition=gpu, ~45 mins, dependency=afterok:GRPO)

set -euo pipefail

cd "$(dirname "$0")/../.."
command -v sbatch >/dev/null || { echo "sbatch is required; run this on a SoC login node" >&2; exit 2; }
test -x .venv/bin/python || { echo "Missing .venv/bin/python" >&2; exit 2; }

run_tag="${RUN_TAG:-run1}"
[[ "$run_tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
  echo "RUN_TAG must use letters, digits, underscore or dash" >&2; exit 2;
}

echo "=== Submitting Fast-Slow Dual-Process Training & Evaluation Pipeline ==="

# 1. Fast-Slow SFT (Task 7)
sft_job=$(sbatch --parsable --partition=gpu --time=03:00:00 --exclude=xgpe2 \
  --job-name="fastslow-sft" \
  --export=ALL,SLURM_ARRAY_TASK_ID=7,RUN_TAG="$run_tag" \
  run.sbatch)

# 2. Fast-Slow GRPO with Self-Correction Reward (Task 8, depends on SFT)
grpo_job=$(sbatch --parsable --partition=gpu-long --time=1-00:00:00 --exclude=xgpe2 \
  --job-name="fastslow-grpo" \
  --dependency="afterok:$sft_job" \
  --export=ALL,SLURM_ARRAY_TASK_ID=8,RUN_TAG="$run_tag" \
  run.sbatch)

# 3. Fast-Slow Evaluation (Task 9, depends on GRPO)
eval_job=$(sbatch --parsable --partition=gpu --time=01:30:00 --exclude=xgpe2 \
  --job-name="fastslow-eval" \
  --dependency="afterok:$grpo_job" \
  --export=ALL,SLURM_ARRAY_TASK_ID=9,RUN_TAG="$run_tag" \
  run.sbatch)

echo "Pipeline successfully queued with dependencies:"
printf "  [1] Fast-Slow SFT 4500:       Job ID %s\n" "$sft_job"
printf "  [2] Fast-Slow GRPO 1000:      Job ID %s (afterok:%s)\n" "$grpo_job" "$sft_job"
printf "  [3] Fast-Slow Dev Evaluation: Job ID %s (afterok:%s)\n" "$eval_job" "$grpo_job"
echo ""
echo "Monitor status with: squeue -u \$USER"
