#!/usr/bin/env bash
#SBATCH --job-name=verifier-pipeline
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --array=0-2%3
#SBATCH --output=slurm-%x-%A_%a.out

# Master Execution & Evaluation Array:
#   Array Tasks 0-2 (Default Baseline Evaluation under Native Tag Format):
#     Task 0: Base Qwen3.5-2B on FEVER dev (900 claims) with direct_tag (cap 1024).
#     Task 1: Base Qwen3.5-2B on FEVER dev (900 claims) with cot_brief_tag (cap 1024).
#     Task 2: Base Qwen3.5-2B on FEVER dev (900 claims) with cot_chain_tag (cap 1024).
#
#   Array Tasks 3-6 (Dual-Path 4,500 SFT & 1,000 GRPO Training Pipeline):
#     Task 3: Train Path 1 (JSON) Direct SFT on 4,500 disjoint claims.
#     Task 4: Train Path 2 (Native Tag) Direct SFT on 4,500 disjoint claims.
#     Task 5: Wait for Path 1 SFT checkpoint, then train Path 1 (JSON) CoT GRPO on 1,000 holdout claims.
#     Task 6: Wait for Path 2 Tag SFT checkpoint, then train Path 2 (Tag) CoT GRPO on 1,000 holdout claims.
#
# Note: For native Slurm job dependencies across partitions (gpu vs gpu-long), use:
#   bash verifier/experiments/submit_4500_pipeline.sh

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

seed="${SEED:-0}"
[[ "$seed" =~ ^[0-9]+$ ]] || {
  echo "SEED must be a nonnegative integer" >&2
  exit 2
}

n_claims="${N_CLAIMS:-900}"
max_new_tokens="${MAX_NEW_TOKENS:-1024}"

wait_for_adapter() {
  local target="$1" max_wait=28800 elapsed=0
  echo "Checking for completed adapter in: $target"
  while [[ $elapsed -lt $max_wait ]]; do
    if [[ -f "$target/adapter_config.json" && ( -s "$target/adapter_model.safetensors" || -s "$target/adapter_model.bin" ) ]]; then
      echo "Completed adapter detected: $target (waited ${elapsed}s)"
      return 0
    fi
    echo "Adapter not ready yet (${elapsed}s elapsed). Sleeping 30s..."
    sleep 30
    elapsed=$((elapsed + 30))
  done
  echo "Timeout waiting for adapter in $target" >&2
  return 2
}

case "${SLURM_ARRAY_TASK_ID:?Expected array task 0 to 6}" in
  # -------------------------------------------------------------
  # Baseline Evaluation under Native Tag Format (Base Qwen3.5-2B)
  # -------------------------------------------------------------
  0)
    echo "=== Task 0: Evaluate Base Model under direct_tag (cap $max_new_tokens, $n_claims claims) ==="
    export EVAL_MODEL="base-qwen3p5-2b"
    export PROMPT_STYLE="direct_tag"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="$run_tag"
    unset CHECKPOINTS
    unset EVAL_CONDITION
    bash verifier/evaluation/run_eval.sh
    ;;

  1)
    echo "=== Task 1: Evaluate Base Model under cot_brief_tag (cap $max_new_tokens, $n_claims claims) ==="
    export EVAL_MODEL="base-qwen3p5-2b"
    export PROMPT_STYLE="cot_brief_tag"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="$run_tag"
    unset CHECKPOINTS
    unset EVAL_CONDITION
    bash verifier/evaluation/run_eval.sh
    ;;

  2)
    echo "=== Task 2: Evaluate Base Model under cot_chain_tag (cap $max_new_tokens, $n_claims claims) ==="
    export EVAL_MODEL="base-qwen3p5-2b"
    export PROMPT_STYLE="cot_chain_tag"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="$run_tag"
    unset CHECKPOINTS
    unset EVAL_CONDITION
    bash verifier/evaluation/run_eval.sh
    ;;

  # -------------------------------------------------------------
  # Dual-Path 4,500 SFT Training Tasks
  # -------------------------------------------------------------
  3)
    config="sft-direct-from-base-qwen3p5-2b-disjoint-4500-samples"
    run_name="${SFT_JSON_RUN:-${run_tag}-json-sft}"
    script="verifier/experiments/$config/train.py"
    echo "=== Task 3: Train Path 1 (JSON) Direct SFT (4,500 claims): config=$config run=$run_name ==="
    python "$script" --run-name "$run_name" --seed "$seed"
    ;;

  4)
    config="sft-direct-from-base-qwen3p5-2b-tagged-4500-samples"
    run_name="${SFT_TAG_RUN:-${run_tag}-tag-sft}"
    script="verifier/experiments/$config/train.py"
    echo "=== Task 4: Train Path 2 (Native Tag) Direct SFT (4,500 claims): config=$config run=$run_name ==="
    python "$script" --run-name "$run_name" --seed "$seed"
    ;;

  # -------------------------------------------------------------
  # Dual-Path 1,000 GRPO Training Tasks (with checkpoint dependency)
  # -------------------------------------------------------------
  5)
    sft_config="sft-direct-from-base-qwen3p5-2b-disjoint-4500-samples"
    sft_run="${SFT_JSON_RUN:-${run_tag}-json-sft}"
    sft_cp="verifier/experiments/${sft_config}/runs/${sft_run}/artifacts/checkpoints"
    wait_for_adapter "$sft_cp"

    grpo_config="grpo-cot-from-sft4500-qwen3p5-2b-no-calibration-kl0-1000-samples"
    grpo_run="${GRPO_JSON_RUN:-${run_tag}-json-grpo}"
    grpo_data="verifier/data/train/sft/fever-oracle-matched-grpo-1000"
    script="verifier/experiments/$grpo_config/train.py"

    echo "=== Task 5: Train Path 1 (JSON) CoT GRPO (1,000 claims): config=$grpo_config run=$grpo_run ==="
    python "$script" --run-name "$grpo_run" --seed "$seed" --sft "$sft_cp" --data "$grpo_data"
    ;;

  6)
    sft_config="sft-direct-from-base-qwen3p5-2b-tagged-4500-samples"
    sft_run="${SFT_TAG_RUN:-${run_tag}-tag-sft}"
    sft_cp="verifier/experiments/${sft_config}/runs/${sft_run}/artifacts/checkpoints"
    wait_for_adapter "$sft_cp"

    grpo_config="grpo-cot-from-sft4500tagged-qwen3p5-2b-no-calibration-kl0-1000-samples"
    grpo_run="${GRPO_TAG_RUN:-${run_tag}-tag-grpo}"
    grpo_data="verifier/data/train/sft/fever-oracle-tagged-grpo-1000"
    script="verifier/experiments/$grpo_config/train.py"

    echo "=== Task 6: Train Path 2 (Native Tag) CoT GRPO (1,000 claims): config=$grpo_config run=$grpo_run ==="
    python "$script" --run-name "$grpo_run" --seed "$seed" --sft "$sft_cp" --data "$grpo_data"
    ;;

  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID (expected 0 to 6)" >&2
    exit 2
    ;;
esac
