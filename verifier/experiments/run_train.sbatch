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
resolve_renamed_path() {
  python -c 'import sys; from utils.experiment_layout import current_experiment_path; print(current_experiment_path(sys.argv[1]))' "$1"
}
if [[ ! -f "$train_script" ]]; then
  resolved_script="$(resolve_renamed_path "$train_script")"
  if [[ "$resolved_script" != "$train_script" ]]; then
    echo "Renamed training script: $train_script -> $resolved_script"
    train_script="$resolved_script"
  fi
fi
test -f "$train_script" || { echo "Missing training script: $train_script" >&2; exit 2; }
if [[ -n "${SFT_CHECKPOINTS:-}" ]]; then
  resolved_checkpoint="$(resolve_renamed_path "$SFT_CHECKPOINTS")"
  if [[ "$resolved_checkpoint" != "$SFT_CHECKPOINTS" ]]; then
    echo "Renamed SFT checkpoint path: $SFT_CHECKPOINTS -> $resolved_checkpoint"
    SFT_CHECKPOINTS="$resolved_checkpoint"
  fi
fi
if [[ -n "${INIT_CHECKPOINTS:-}" ]]; then
  resolved_checkpoint="$(resolve_renamed_path "$INIT_CHECKPOINTS")"
  if [[ "$resolved_checkpoint" != "$INIT_CHECKPOINTS" ]]; then
    echo "Renamed initial checkpoint path: $INIT_CHECKPOINTS -> $resolved_checkpoint"
    INIT_CHECKPOINTS="$resolved_checkpoint"
  fi
fi
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
    verifier/experiments/grpo-*/train.py) args+=(--data "$TRAIN_DATA") ;;
    *) echo "TRAIN_DATA applies only to CoT SFT or GRPO training" >&2; exit 2 ;;
  esac
fi
echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) script=$train_script run=$run_name seed=$seed output_root=${OUTPUT_ROOT:-verifier/experiments}"
srun python "$train_script" "${args[@]}"
