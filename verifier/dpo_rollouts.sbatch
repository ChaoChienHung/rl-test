#!/usr/bin/env bash
#SBATCH --job-name=dpo-rollouts
#SBATCH --partition=gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
test -f README.md && test -x .venv/bin/python || { echo "Missing repo root or .venv" >&2; exit 2; }
source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false

sft="${SFT_CHECKPOINTS:-verifier/experiments/sft-direct-from-base-qwen3p5-2b-disjoint-4500-samples/runs/run1-json-sft/artifacts/checkpoints}"
test -f "$sft/adapter_config.json" || { echo "Missing SFT4500 checkpoint: $sft" >&2; exit 2; }
name="${ROLLOUT_NAME:-job-${SLURM_JOB_ID:?}}"
[[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid ROLLOUT_NAME" >&2; exit 2; }
output="verifier/data/train/dpo/rollouts-sft4500-pool1000/$name"
echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) output=$output"
srun python -m utils.generate_dpo_rollouts \
  --source verifier/data/train/sft/fever-oracle-matched-grpo-1000/labeled_prompts.jsonl \
  --checkpoint "$sft" --output "$output" --n-claims 1000 \
  --rollouts "${DPO_ROLLOUTS:-6}" --max-new-tokens 1024 --max-prompt-tokens 0 --seed 0
echo "DPO_ROLLOUTS_DIR=$output"
