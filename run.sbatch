#!/usr/bin/env bash
#SBATCH --job-name=sft-5500-comparison
#SBATCH --partition=gpu-long
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --array=0-1%2
#SBATCH --output=slurm-%x-%A_%a.out

# Array Task 0 trains Direct SFT on the 5,500-claim matched dataset.
# Array Task 1 trains CoT SFT on the 5,500-claim repaired teacher dataset.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f verifier/experiments/sft-direct-from-base-qwen3p5-2b-5500-samples/train.py || {
  echo "Submit from the repository root (missing direct 5500 training script)" >&2
  exit 2
}
test -f verifier/experiments/sft-cot-from-base-qwen3p5-2b-5500-samples/train.py || {
  echo "Submit from the repository root (missing CoT 5500 training script)" >&2
  exit 2
}

source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false

run_tag="${RUN_TAG:-run1}"
[[ "$run_tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
  echo "RUN_TAG must use letters, digits, underscore or dash" >&2
  exit 2
}

seed="${SEED:-0}"
[[ "$seed" =~ ^[0-9]+$ ]] || {
  echo "SEED must be a nonnegative integer" >&2
  exit 2
}

case "${SLURM_ARRAY_TASK_ID:?Expected array task 0 or 1}" in
  0)
    config="sft-direct-from-base-qwen3p5-2b-5500-samples"
    run_name="${DIRECT_RUN_NAME:-${run_tag}}"
    [[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
      echo "DIRECT_RUN_NAME must use letters, digits, underscore or dash" >&2
      exit 2
    }
    script="verifier/experiments/$config/train.py"
    data="verifier/data/train/sft/fever-oracle-direct-5500/labeled_prompts.jsonl"
    test -f "$data" || {
      echo "Missing Direct 5500 data: $data" >&2
      exit 2
    }
    args=(--run-name "$run_name" --seed "$seed")
    if [[ -n "${OUTPUT_ROOT:-}" ]]; then
      args+=(--output-root "$OUTPUT_ROOT")
    fi
    echo "Task 0: Train Direct SFT (5,500 claims): config=$config run=$run_name seed=$seed"
    srun python "$script" "${args[@]}"
    ;;
  1)
    config="sft-cot-from-base-qwen3p5-2b-5500-samples"
    run_name="${COT_RUN_NAME:-${run_tag}}"
    [[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
      echo "COT_RUN_NAME must use letters, digits, underscore or dash" >&2
      exit 2
    }
    script="verifier/experiments/$config/train.py"
    data="${COT_DATA:-verifier/data/train/sft/fever-oracle-cot-soclaas/cot_sft_labeled_prompts.jsonl}"
    test -f "$data" || {
      echo "Missing CoT 5500 data: $data" >&2
      exit 2
    }
    args=(--run-name "$run_name" --seed "$seed" --data "$data")
    if [[ -n "${OUTPUT_ROOT:-}" ]]; then
      args+=(--output-root "$OUTPUT_ROOT")
    fi
    echo "Task 1: Train CoT SFT (5,500 claims): config=$config run=$run_name data=$data seed=$seed"
    srun python "$script" "${args[@]}"
    ;;
  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID (expected 0 or 1)" >&2
    exit 2
    ;;
esac
