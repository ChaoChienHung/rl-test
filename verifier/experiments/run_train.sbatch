#!/usr/bin/env bash
#SBATCH --job-name=train-verifier
#SBATCH --partition=gpu-long
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
test -f README.md || { echo "Submit from the repository root (README.md missing)" >&2; exit 2; }
source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false

train_script="${TRAIN_SCRIPT:-verifier/experiments/sft-direct-from-base-qwen3p5-2b-train5895/train.py}"
case "$train_script" in
  verifier/experiments/*/train.py) ;;
  *) echo "TRAIN_SCRIPT must be a verifier/experiments/<config>/train.py path" >&2; exit 2 ;;
esac
test -f "$train_script" || { echo "Missing training script: $train_script" >&2; exit 2; }
run_name="${RUN_NAME:-job-${SLURM_JOB_ID:?}}"
seed="${SEED:-0}"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }
args=(--run-name "$run_name" --seed "$seed")
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  args+=(--output-root "$OUTPUT_ROOT")
fi
if [[ -n "${SFT_CHECKPOINTS:-}" ]]; then
  case "$train_script" in
    verifier/experiments/grpo-*/train.py)
      [[ -f "$SFT_CHECKPOINTS/adapter_config.json" ]] || {
        echo "Missing SFT adapter config: $SFT_CHECKPOINTS/adapter_config.json" >&2
        exit 2
      }
      [[ -s "$SFT_CHECKPOINTS/adapter_model.safetensors" || -s "$SFT_CHECKPOINTS/adapter_model.bin" ]] || {
        echo "Missing SFT adapter weights in $SFT_CHECKPOINTS" >&2
        exit 2
      }
      args+=(--sft "$SFT_CHECKPOINTS")
      ;;
    *) echo "SFT_CHECKPOINTS applies only to GRPO training" >&2; exit 2 ;;
  esac
fi
if [[ -n "${INIT_CHECKPOINTS:-}" ]]; then
  case "$train_script" in
    verifier/experiments/sft-direct-*/train.py) ;;
    *) echo "INIT_CHECKPOINTS applies only to SFT training" >&2; exit 2 ;;
  esac
  [[ -f "$INIT_CHECKPOINTS/adapter_config.json" ]] || {
    echo "Missing initial adapter config: $INIT_CHECKPOINTS/adapter_config.json" >&2; exit 2;
  }
  [[ -s "$INIT_CHECKPOINTS/adapter_model.safetensors" || -s "$INIT_CHECKPOINTS/adapter_model.bin" ]] || {
    echo "Missing initial adapter weights in $INIT_CHECKPOINTS" >&2; exit 2;
  }
  args+=(--init-adapter "$INIT_CHECKPOINTS")
fi
if [[ -n "${TRAIN_DATA:-}" ]]; then
  case "$train_script" in
    verifier/experiments/sft-cot-*/train.py) args+=(--data "$TRAIN_DATA") ;;
    verifier/experiments/grpo-direct-*/train.py) args+=(--data "$TRAIN_DATA") ;;
    *) echo "TRAIN_DATA applies only to CoT SFT or single-turn GRPO training" >&2; exit 2 ;;
  esac
fi
echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) script=$train_script run=$run_name seed=$seed output_root=${OUTPUT_ROOT:-verifier/experiments}"
srun python "$train_script" "${args[@]}"
