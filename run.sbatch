#!/usr/bin/env bash
#SBATCH --job-name=fastslow-pipeline
#SBATCH --partition=gpu-long
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --output=slurm-%x-%j.out

# ==============================================================================
# Fast-Slow Dual-Process Training & Evaluation Master Pipeline
# ==============================================================================
# Usage options:
#   1. Run full end-to-end pipeline in one job (SFT -> GRPO -> Eval):
#        sbatch run.sbatch
#
#   2. Run a specific stage:
#        sbatch --export=ALL,STAGE=sft run.sbatch
#        sbatch --export=ALL,STAGE=grpo run.sbatch
#        sbatch --export=ALL,STAGE=eval run.sbatch
#
#   3. Slurm array mode (0=sft, 1=grpo, 2=eval):
#        sbatch --array=0-2%1 run.sbatch
# ==============================================================================

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

seed="${SEED:-0}"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }
n_claims="${N_CLAIMS:-900}"
max_new_tokens="${MAX_NEW_TOKENS:-1024}"

# Base configurations
SFT_CONFIG="sft-fastslow-from-base-qwen3p5-2b-4500-samples"
GRPO_CONFIG="grpo-fastslow-from-sft4500-qwen3p5-2b-no-calibration-kl0-1000-samples"
GRPO_DATA="verifier/data/train/grpo/fever-oracle-fastslow-grpo-1000"

# Collision-proof unique run name resolver
resolve_unique_run() {
  local base_dir="$1"
  local base_name="$2"
  local candidate="$base_name"
  local idx=1
  while [[ -e "$base_dir/$candidate" ]]; do
    idx=$((idx + 1))
    candidate="${base_name}_${idx}"
  done
  echo "$candidate"
}

# Determine base run tag from RUN_TAG or default
requested_tag="${RUN_TAG:-run1}"

# Resolve unique non-colliding run names for each stage
sft_runs_dir="verifier/experiments/$SFT_CONFIG/runs"
mkdir -p "$sft_runs_dir"
if [[ -n "${SFT_RUN:-}" ]]; then
  SFT_RUN_NAME="$SFT_RUN"
else
  SFT_RUN_NAME="$(resolve_unique_run "$sft_runs_dir" "${requested_tag}-fastslow-sft")"
fi
SFT_CHECKPOINTS="verifier/experiments/$SFT_CONFIG/runs/$SFT_RUN_NAME/artifacts/checkpoints"

grpo_runs_dir="verifier/experiments/$GRPO_CONFIG/runs"
mkdir -p "$grpo_runs_dir"
if [[ -n "${GRPO_RUN:-}" ]]; then
  GRPO_RUN_NAME="$GRPO_RUN"
else
  GRPO_RUN_NAME="$(resolve_unique_run "$grpo_runs_dir" "${requested_tag}-fastslow-grpo")"
fi
GRPO_CHECKPOINTS="verifier/experiments/$GRPO_CONFIG/runs/$GRPO_RUN_NAME/artifacts/checkpoints"

eval_results_dir="verifier/evaluation/results/$GRPO_CONFIG/cot-fastslow-cap${max_new_tokens}"
mkdir -p "$eval_results_dir"
if [[ -n "${EVAL_RUN:-}" ]]; then
  EVAL_RUN_NAME="$EVAL_RUN"
else
  EVAL_RUN_NAME="$(resolve_unique_run "$eval_results_dir" "$GRPO_RUN_NAME")"
fi

wait_for_adapter() {
  local target="$1" max_wait=28800 elapsed=0
  echo "Checking for adapter in: $target"
  while [[ $elapsed -lt $max_wait ]]; do
    if [[ -f "$target/adapter_config.json" && ( -s "$target/adapter_model.safetensors" || -s "$target/adapter_model.bin" ) ]]; then
      echo "Ready adapter detected: $target (waited ${elapsed}s)"
      return 0
    fi
    echo "Adapter not ready yet (${elapsed}s elapsed). Sleeping 30s..."
    sleep 30
    elapsed=$((elapsed + 30))
  done
  echo "Timeout waiting for adapter in $target" >&2
  return 2
}

run_sft() {
  echo ""
  echo "=========================================================================="
  echo ">>> [STAGE 1] Train Fast-Slow SFT on 4,500 Claims"
  echo ">>> Config: $SFT_CONFIG"
  echo ">>> Run Name: $SFT_RUN_NAME"
  echo ">>> Output Checkpoints: $SFT_CHECKPOINTS"
  echo "=========================================================================="
  python "verifier/experiments/$SFT_CONFIG/train.py" --run-name "$SFT_RUN_NAME" --seed "$seed"
  echo ">>> [STAGE 1] SFT training completed successfully!"
}

run_grpo() {
  echo ""
  echo "=========================================================================="
  echo ">>> [STAGE 2] Train Fast-Slow Native-Thinking GRPO on 1,000 Claims"
  echo ">>> Config: $GRPO_CONFIG"
  echo ">>> Run Name: $GRPO_RUN_NAME"
  echo ">>> Initial SFT Adapter: $SFT_CHECKPOINTS"
  echo ">>> Reward: Cognitive Self-Correction Reward Matrix"
  echo "=========================================================================="
  wait_for_adapter "$SFT_CHECKPOINTS"
  python "verifier/experiments/$GRPO_CONFIG/train.py" \
    --run-name "$GRPO_RUN_NAME" \
    --seed "$seed" \
    --sft "$SFT_CHECKPOINTS" \
    --data "$GRPO_DATA"
  echo ">>> [STAGE 2] GRPO training completed successfully!"
}

run_eval() {
  echo ""
  echo "=========================================================================="
  echo ">>> [STAGE 3] Evaluate Fast-Slow GRPO Model on FEVER Dev"
  echo ">>> Model: $GRPO_CONFIG"
  echo ">>> Prompt Style: cot_fastslow"
  echo ">>> Max Tokens: $max_new_tokens | Claims: $n_claims"
  echo ">>> Adapter: $GRPO_CHECKPOINTS"
  echo ">>> Run Name: $EVAL_RUN_NAME"
  echo "=========================================================================="
  wait_for_adapter "$GRPO_CHECKPOINTS"
  export EVAL_MODEL="$GRPO_CONFIG"
  export PROMPT_STYLE="cot_fastslow"
  export MAX_NEW_TOKENS="$max_new_tokens"
  export N_CLAIMS="$n_claims"
  export RUN_NAME="$EVAL_RUN_NAME"
  export CHECKPOINTS="$GRPO_CHECKPOINTS"
  unset EVAL_CONDITION || true
  bash verifier/evaluation/run_eval.sh
  echo ">>> [STAGE 3] Evaluation completed successfully!"
}

# Determine execution mode:
# If SLURM_ARRAY_TASK_ID is defined (array job) or STAGE is set, run only that stage.
# Otherwise, run all 3 stages sequentially in this single job!
selected_stage="${STAGE:-}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  case "$SLURM_ARRAY_TASK_ID" in
    0|7) selected_stage="sft" ;;
    1|8) selected_stage="grpo" ;;
    2|9) selected_stage="eval" ;;
    *) echo "Unknown array task ID: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
  esac
fi

case "$selected_stage" in
  sft)
    run_sft
    ;;
  grpo)
    run_grpo
    ;;
  eval)
    run_eval
    ;;
  "")
    echo "=== Running Full Fast-Slow Pipeline Sequentially (SFT -> GRPO -> Eval) ==="
    run_sft
    run_grpo
    run_eval
    echo ""
    echo "=========================================================================="
    echo ">>> Fast-Slow Pipeline Finished End-to-End!"
    echo ">>> Evaluated: $eval_results_dir/$EVAL_RUN_NAME/metrics.json"
    echo "=========================================================================="
    ;;
  *)
    echo "Unknown stage: $selected_stage (expected 'sft', 'grpo', 'eval', or empty for full pipeline)" >&2
    exit 2
    ;;
esac
