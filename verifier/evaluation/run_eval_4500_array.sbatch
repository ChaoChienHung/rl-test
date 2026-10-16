#!/usr/bin/env bash
#SBATCH --job-name=eval-4500-array
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --array=0-4%5
#SBATCH --output=slurm-%x-%A_%a.out

# Multi-Condition Validation Array for 4,500 SFT & 1,000 GRPO Checkpoints:
#
#   Current SFT 4,500 Tasks (Tasks 0-4):
#     Task 0: Path 1 (JSON SFT 4500)   -> direct (cap 1024)
#     Task 1: Path 1 (JSON SFT 4500)   -> cot_brief (cap 1024)
#     Task 2: Path 2 (Tag SFT 4500)    -> direct_tag (cap 1024)
#     Task 3: Path 2 (Tag SFT 4500)    -> cot_brief_tag (cap 1024)
#     Task 4: Path 2 (Tag SFT 4500)    -> cot_chain_tag (cap 1024)
#
#   Subsequent GRPO 1,000 Tasks (Tasks 5-6, run when GRPO finishes):
#     Task 5: Path 1 (JSON GRPO 1000)  -> cot_native (cap 1024)
#     Task 6: Path 2 (Tag GRPO 1000)   -> cot_native_tag (cap 1024)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repo root}"

source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"

n_claims="${N_CLAIMS:-900}"
max_new_tokens="${MAX_NEW_TOKENS:-1024}"
run_tag="${RUN_TAG:-run1}"

# SFT 4,500 Checkpoint Paths
sft_json_cp="verifier/experiments/sft-direct-from-base-qwen3p5-2b-disjoint-4500-samples/runs/run1-json-sft/artifacts/checkpoints"
sft_tag_cp="verifier/experiments/sft-direct-from-base-qwen3p5-2b-tagged-4500-samples/runs/run1-tag-sft/artifacts/checkpoints"

# GRPO 1,000 Checkpoint Paths
grpo_json_cp="verifier/experiments/grpo-cot-from-sft4500-qwen3p5-2b-no-calibration-kl0-1000-samples/runs/run1-json-grpo/artifacts/checkpoints"
grpo_tag_cp="verifier/experiments/grpo-cot-from-sft4500tagged-qwen3p5-2b-no-calibration-kl0-1000-samples/runs/run1-tag-grpo/artifacts/checkpoints"

case "${SLURM_ARRAY_TASK_ID:?Expected SLURM_ARRAY_TASK_ID 0-6}" in
  0)
    echo "=== Task 0: Evaluate Path 1 (JSON SFT 4500) under direct (cap $max_new_tokens) ==="
    export CHECKPOINTS="$sft_json_cp"
    export PROMPT_STYLE="direct"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="$run_tag"
    bash verifier/evaluation/run_eval.sh
    ;;

  1)
    echo "=== Task 1: Evaluate Path 1 (JSON SFT 4500) under cot_brief (cap $max_new_tokens) ==="
    export CHECKPOINTS="$sft_json_cp"
    export PROMPT_STYLE="cot_brief"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="$run_tag"
    bash verifier/evaluation/run_eval.sh
    ;;

  2)
    echo "=== Task 2: Evaluate Path 2 (Tag SFT 4500) under direct_tag (cap $max_new_tokens) ==="
    export CHECKPOINTS="$sft_tag_cp"
    export PROMPT_STYLE="direct_tag"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="$run_tag"
    bash verifier/evaluation/run_eval.sh
    ;;

  3)
    echo "=== Task 3: Evaluate Path 2 (Tag SFT 4500) under cot_brief_tag (cap $max_new_tokens) ==="
    export CHECKPOINTS="$sft_tag_cp"
    export PROMPT_STYLE="cot_brief_tag"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="$run_tag"
    bash verifier/evaluation/run_eval.sh
    ;;

  4)
    echo "=== Task 4: Evaluate Path 2 (Tag SFT 4500) under cot_chain_tag (cap $max_new_tokens) ==="
    export CHECKPOINTS="$sft_tag_cp"
    export PROMPT_STYLE="cot_chain_tag"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="$run_tag"
    bash verifier/evaluation/run_eval.sh
    ;;

  5)
    echo "=== Task 5: Evaluate Path 1 (JSON GRPO 1000) under cot_native (cap $max_new_tokens) ==="
    export CHECKPOINTS="$grpo_json_cp"
    export PROMPT_STYLE="cot_native"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="$run_tag"
    bash verifier/evaluation/run_eval.sh
    ;;

  6)
    echo "=== Task 6: Evaluate Path 2 (Tag GRPO 1000) under cot_native_tag (cap $max_new_tokens) ==="
    export CHECKPOINTS="$grpo_tag_cp"
    export PROMPT_STYLE="cot_native_tag"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export N_CLAIMS="$n_claims"
    export RUN_NAME="$run_tag"
    bash verifier/evaluation/run_eval.sh
    ;;

  *)
    echo "Unexpected task ID: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac
