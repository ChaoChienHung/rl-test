#!/usr/bin/env bash
#SBATCH --job-name=sft5500-eval-train
#SBATCH --partition=gpu-long
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --array=0-1%2
#SBATCH --output=slurm-%x-%A_%a.out

# Array Task 0: Evaluate Direct SFT (5,500 claims) on FEVER dev (both direct and cot_brief, cap 1024).
# Array Task 1: Train Native Thinking CoT SFT (5,500 claims) on Qwen3.5-2B (1 epoch, ~1.8h).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f verifier/evaluation/run_eval.sh || {
  echo "Submit from the repository root (missing verifier/evaluation/run_eval.sh)" >&2
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

resolve_adapter() {
  local checkpoint
  for checkpoint in "$@"; do
    if [[ -f "$checkpoint/adapter_config.json" &&
          ( -s "$checkpoint/adapter_model.safetensors" || -s "$checkpoint/adapter_model.bin" ) ]]; then
      printf '%s\n' "$checkpoint"
      return 0
    fi
  done
  printf 'Missing completed adapter with weights; checked:\n' >&2
  printf '  %s\n' "$@" >&2
  return 2
}

case "${SLURM_ARRAY_TASK_ID:?Expected array task 0 or 1}" in
  0)
    direct_run_name="${DIRECT_RUN_NAME:-$run_tag}"
    checkpoint="$(resolve_adapter \
      "${DIRECT_CHECKPOINTS:-verifier/experiments/sft-direct-from-base-qwen3p5-2b-5500-samples/runs/${direct_run_name}/artifacts/checkpoints}")"
    max_new_tokens="${MAX_NEW_TOKENS:-1024}"
    eval_styles="${DIRECT_EVAL_STYLES:-"direct cot_brief"}"

    echo "Task 0: Evaluate Direct SFT (5,500 claims): checkpoint=$checkpoint styles=$eval_styles cap=$max_new_tokens"
    for style in $eval_styles; do
      style_hyphen="${style//_/-}"
      echo "--- Evaluating Direct SFT with style=$style (hierarchical layout: sft-direct-from-base-qwen3p5-2b-5500-samples/${style_hyphen}-cap${max_new_tokens}) ---"
      export CHECKPOINTS="$checkpoint"
      export EVAL_MODEL="sft-direct-from-base-qwen3p5-2b-5500-samples"
      export PROMPT_STYLE="$style"
      export THINKING=off
      export MAX_NEW_TOKENS="$max_new_tokens"
      export N_CLAIMS="${N_CLAIMS:-900}"
      export RUN_NAME="${DIRECT_RUN_NAME:-$run_tag}"
      unset EVAL_CONDITION
      bash verifier/evaluation/run_eval.sh
    done
    ;;
  1)
    config="sft-cot-from-base-qwen3p5-2b-5500-samples"
    run_name="${COT_RUN_NAME:-${run_tag}}"
    [[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
      echo "COT_RUN_NAME must use letters, digits, underscore or dash" >&2
      exit 2
    }
    script="verifier/experiments/$config/train.py"
    data="${COT_DATA:-verifier/data/train/sft/fever-oracle-cot-soclaas/cot_native_sft_labeled_prompts.jsonl}"
    test -f "$data" || {
      echo "Missing Native Thinking CoT 5500 data: $data" >&2
      exit 2
    }
    seed="${SEED:-0}"
    [[ "$seed" =~ ^[0-9]+$ ]] || {
      echo "SEED must be a nonnegative integer" >&2
      exit 2
    }
    args=(--run-name "$run_name" --seed "$seed" --data "$data")
    if [[ -n "${OUTPUT_ROOT:-}" ]]; then
      args+=(--output-root "$OUTPUT_ROOT")
    fi
    echo "Task 1: Train Native Thinking CoT SFT (5,500 claims): config=$config run=$run_name data=$data seed=$seed"
    srun python "$script" "${args[@]}"
    ;;
  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID (expected 0 or 1)" >&2
    exit 2
    ;;
esac
