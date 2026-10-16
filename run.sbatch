#!/usr/bin/env bash
#SBATCH --job-name=cot-eval-sweep
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --array=0-1%2
#SBATCH --output=slurm-%x-%A_%a.out

# Array Task 0: Evaluate Native-Thinking CoT SFT (5,500 claims, run1) on FEVER dev (cot_native, cap 1024).
# Array Task 1: Evaluate Unseen-Arm CoT GRPO (from SFT 4695, cot-grpo1) on FEVER dev (cot_native, cap 1024).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f verifier/evaluation/run_eval.sh || {
  echo "Submit from the repository root (missing verifier/evaluation/run_eval.sh)" >&2
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

max_new_tokens="${MAX_NEW_TOKENS:-1024}"
n_claims="${N_CLAIMS:-900}"
prompt_style="${PROMPT_STYLE:-cot_native}"

case "${SLURM_ARRAY_TASK_ID:?Expected array task 0 or 1}" in
  0)
    eval_model="sft-cot-from-base-qwen3p5-2b-5500-samples"
    sft_run="${SFT_COT_RUN:-$run_tag}"
    checkpoint="$(resolve_adapter \
      "${SFT_COT_CHECKPOINTS:-verifier/experiments/${eval_model}/runs/${sft_run}/artifacts/checkpoints}")"

    echo "=== Task 0: Evaluate Native-Thinking CoT SFT (5,500 claims) ==="
    echo "Checkpoint: $checkpoint"
    echo "Prompt Style: $prompt_style | Cap: $max_new_tokens | Claims: $n_claims"

    export CHECKPOINTS="$checkpoint"
    export EVAL_MODEL="$eval_model"
    export PROMPT_STYLE="$prompt_style"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="$sft_run"
    unset EVAL_CONDITION
    bash verifier/evaluation/run_eval.sh
    ;;

  1)
    eval_model="grpo-cot-from-sft4695-qwen3p5-2b-no-calibration-kl0-1200-samples"
    grpo_run="${GRPO_UNSEEN_RUN:-cot-grpo1-unseen-cot-grpo}"
    checkpoint="$(resolve_adapter \
      "${GRPO_UNSEEN_CHECKPOINTS:-verifier/experiments/${eval_model}/runs/${grpo_run}/artifacts/checkpoints}")"

    echo "=== Task 1: Evaluate Unseen-Arm Native-Thinking CoT GRPO (from SFT 4695) ==="
    echo "Checkpoint: $checkpoint"
    echo "Prompt Style: $prompt_style | Cap: $max_new_tokens | Claims: $n_claims"

    export CHECKPOINTS="$checkpoint"
    export EVAL_MODEL="$eval_model"
    export PROMPT_STYLE="$prompt_style"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="${GRPO_EVAL_RUN_NAME:-run1}"
    unset EVAL_CONDITION
    bash verifier/evaluation/run_eval.sh
    ;;

  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID (expected 0 or 1)" >&2
    exit 2
    ;;
esac
