#!/usr/bin/env bash
#SBATCH --job-name=split-followup
#SBATCH --partition=gpu-long
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --array=0-2%2
#SBATCH --output=slurm-%x-%A_%a.out

# Task 0 evaluates the 1,200-row SFT continuation. Tasks 1 and 2 train
# native-thinking GRPO from the 5,895-row and disjoint 4,695-row SFT adapters.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f verifier/experiments/grpo-cot-from-sft5895-qwen3p5-2b-train1200-full-kl0/train.py || {
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

check_adapter() {
  local checkpoint="$1"
  [[ -f "$checkpoint/adapter_config.json" &&
     ( -s "$checkpoint/adapter_model.safetensors" || -s "$checkpoint/adapter_model.bin" ) ]] || {
    echo "Missing completed adapter config or weights in $checkpoint" >&2
    exit 2
  }
}

case "${SLURM_ARRAY_TASK_ID:?Expected two array tasks}" in
  0)
    sft_run_name="${CONTINUED_SFT_RUN_NAME:-${run_tag}-seen-sft}"
    [[ "$sft_run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
      echo "CONTINUED_SFT_RUN_NAME must use letters, digits, underscore or dash" >&2
      exit 2
    }
    checkpoint="verifier/experiments/sft-direct-from-sft-qwen3p5-2b-train1200/runs/${sft_run_name}/artifacts/checkpoints"
    check_adapter "$checkpoint"
    echo "Evaluate continued SFT: checkpoint=$checkpoint"
    export CHECKPOINTS="$checkpoint"
    export EVAL_CONDITION=sft-continued1200-direct-cap1024
    export RUN_NAME="$run_tag"
    export N_CLAIMS=900 PROMPT_STYLE=direct MAX_NEW_TOKENS=1024
    bash verifier/evaluation/run_eval.sh
    ;;
  1|2)
    if [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then
      arm=seen
      config=grpo-cot-from-sft5895-qwen3p5-2b-train1200-full-kl0
      checkpoint=verifier/experiments/sft-direct-from-base-qwen3p5-2b-train5895/runs/run2/artifacts/checkpoints
    else
      arm=unseen
      config=grpo-cot-from-sft4695-qwen3p5-2b-train1200-full-kl0
      sft_run_name="${DISJOINT_SFT_RUN_NAME:-${run_tag}-unseen-sft}"
      [[ "$sft_run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
        echo "DISJOINT_SFT_RUN_NAME must use letters, digits, underscore or dash" >&2
        exit 2
      }
      checkpoint="verifier/experiments/sft-direct-from-base-qwen3p5-2b-disjoint4695/runs/${sft_run_name}/artifacts/checkpoints"
    fi
    check_adapter "$checkpoint"
    data=verifier/data/train/sft/fever-oracle-matched-grpo-1200
    test -f "$data/labeled_prompts.jsonl" || {
      echo "Missing fixed GRPO training data: $data/labeled_prompts.jsonl" >&2
      exit 2
    }
    seed="${SEED:-0}"
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }
    run_name="${run_tag}-${arm}-cot-grpo"
    args=(--run-name "$run_name" --seed "$seed" --sft "$checkpoint" --data "$data")
    if [[ -n "${OUTPUT_ROOT:-}" ]]; then
      args+=(--output-root "$OUTPUT_ROOT")
    fi
    echo "Train native-thinking GRPO: config=$config run=$run_name init=$checkpoint data=$data"
    srun python "verifier/experiments/$config/train.py" "${args[@]}"
    ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
