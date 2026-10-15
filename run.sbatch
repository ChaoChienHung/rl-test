#!/usr/bin/env bash
#SBATCH --job-name=grpo-split
#SBATCH --partition=gpu-long
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --array=0-1%2
#SBATCH --output=slurm-%x-%A_%a.out

# Submit with --dependency=afterok:<disjoint-SFT-job-id>. Both array tasks
# wait for that SFT job; task 0 starts from the already-trained full SFT run2.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f verifier/experiments/grpo-direct-from-sft-qwen3p5-2b-train1200-kl0/train.py || {
  echo "Submit from the repository root" >&2
  exit 2
}
source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false

run_tag="${RUN_TAG:-split1-seed0}"
[[ "$run_tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
  echo "RUN_TAG must use letters, digits, underscore or dash" >&2
  exit 2
}

case "${SLURM_ARRAY_TASK_ID:?Expected two GRPO array tasks}" in
  0)
    arm=seen
    checkpoint=verifier/experiments/sft-direct-from-base-qwen3p5-2b-train5895/runs/run2/artifacts/checkpoints
    fallback_checkpoint=verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints
    ;;
  1)
    arm=unseen
    sft_run_name="${SFT_RUN_NAME:-${run_tag}-unseen-sft}"
    [[ "$sft_run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
      echo "SFT_RUN_NAME must use letters, digits, underscore or dash" >&2
      exit 2
    }
    checkpoint="verifier/experiments/sft_qwen_3.5-2b_disjoint_n4695/runs/${sft_run_name}/artifacts/checkpoints"
    fallback_checkpoint="verifier/experiments/sft-direct-from-base-qwen3p5-2b-disjoint4695/runs/${sft_run_name}/artifacts/checkpoints"
    ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

has_adapter() {
  [[ -f "$1/adapter_config.json" && -s "$1/adapter_model.safetensors" ]]
}
if ! has_adapter "$checkpoint"; then
  if has_adapter "$fallback_checkpoint"; then
    checkpoint="$fallback_checkpoint"
  elif [[ -s "$fallback_checkpoint/adapter_model.safetensors" && -f "$checkpoint/adapter_config.json" ]]; then
    # Git can move the tracked config while leaving ignored weights at the old path.
    cp "$checkpoint/adapter_config.json" "$fallback_checkpoint/adapter_config.json"
    checkpoint="$fallback_checkpoint"
  elif [[ -s "$checkpoint/adapter_model.safetensors" && -f "$fallback_checkpoint/adapter_config.json" ]]; then
    cp "$fallback_checkpoint/adapter_config.json" "$checkpoint/adapter_config.json"
  fi
fi
has_adapter "$checkpoint" || {
  echo "Missing completed SFT adapter in $checkpoint (or $fallback_checkpoint)" >&2
  exit 2
}

data=verifier/data/train/sft/fever-oracle-matched-grpo-1200
test -f "$data/labeled_prompts.jsonl" || {
  echo "Missing fixed GRPO training data: $data/labeled_prompts.jsonl" >&2
  exit 2
}
run_name="${run_tag}-${arm}-grpo"
seed="${SEED:-0}"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }
args=(--run-name "$run_name" --seed "$seed" --sft "$checkpoint" --data "$data")
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  args+=(--output-root "$OUTPUT_ROOT")
fi
echo "GRPO arm=$arm run=$run_name init=$checkpoint data=$data"
srun python verifier/experiments/grpo-direct-from-sft-qwen3p5-2b-train1200-kl0/train.py "${args[@]}"
