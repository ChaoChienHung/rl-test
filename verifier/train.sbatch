#!/usr/bin/env bash
#SBATCH --job-name=train-verifier
#SBATCH --partition=gpu-long
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --output=slurm-%x-%j.out

# Submit from the repository root with TRAIN_SCRIPT set to a configuration's
# train.py, or the shared DPO pilot trainer selected by DPO_PAIRS.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
test -f README.md || { echo "Submit from the repository root (README.md missing)" >&2; exit 2; }
test -x .venv/bin/python || { echo "Missing .venv/bin/python" >&2; exit 2; }
source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false

train_script="${TRAIN_SCRIPT:?Set TRAIN_SCRIPT to a verifier training script}"
case "$train_script" in
  verifier/experiments/*/train.py|verifier/experiments/_shared/dpo_cot.py) ;;
  *) echo "TRAIN_SCRIPT must be a verifier/experiments/<config>/train.py path or the DPO shared trainer" >&2; exit 2 ;;
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
    verifier/experiments/_shared/dpo_cot.py)
      [[ -f "$SFT_CHECKPOINTS/adapter_config.json" ]] || {
        echo "Missing SFT adapter config: $SFT_CHECKPOINTS/adapter_config.json" >&2; exit 2;
      }
      [[ -s "$SFT_CHECKPOINTS/adapter_model.safetensors" || -s "$SFT_CHECKPOINTS/adapter_model.bin" ]] || {
        echo "Missing SFT adapter weights in $SFT_CHECKPOINTS" >&2; exit 2;
      }
      args+=(--sft "$SFT_CHECKPOINTS")
      ;;
    *) echo "SFT_CHECKPOINTS applies only to GRPO or DPO training" >&2; exit 2 ;;
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
if [[ -n "${DPO_CHECKPOINTS:-}" || -n "${GRPO_IDS_FILE:-}" ]]; then
  case "$train_script" in
    verifier/experiments/grpo-cot-from-sft4500dpo500-*/train.py) ;;
    *) echo "DPO_CHECKPOINTS and GRPO_IDS_FILE apply only to DPO500-to-GRPO500" >&2; exit 2 ;;
  esac
  [[ -f "${DPO_CHECKPOINTS:?Set DPO_CHECKPOINTS}/adapter_config.json" ]] || {
    echo "Missing DPO adapter config" >&2; exit 2;
  }
  [[ -s "$DPO_CHECKPOINTS/adapter_model.safetensors" || -s "$DPO_CHECKPOINTS/adapter_model.bin" ]] || {
    echo "Missing DPO adapter weights" >&2; exit 2;
  }
  [[ -f "${GRPO_IDS_FILE:?Set GRPO_IDS_FILE}" ]] || { echo "Missing GRPO IDs file" >&2; exit 2; }
  args+=(--dpo "$DPO_CHECKPOINTS" --ids-file "$GRPO_IDS_FILE")
  [[ -z "${GRPO_IDS_KEY:-}" ]] || args+=(--ids-key "$GRPO_IDS_KEY")
fi
if [[ "$train_script" == verifier/experiments/_shared/dpo_cot.py ]]; then
  [[ -f "${DPO_PAIRS:?Set DPO_PAIRS to a generated pairs.jsonl}" ]] || {
    echo "Missing DPO pairs: $DPO_PAIRS" >&2; exit 2;
  }
  args+=(--pairs "$DPO_PAIRS")
  [[ -z "${DPO_REPORT:-}" ]] || args+=(--report "$DPO_REPORT")
  [[ -z "${DPO_MAX_STEPS:-}" ]] || args+=(--max-steps "$DPO_MAX_STEPS")
  [[ -z "${DPO_POOL_CLAIMS:-}" ]] || args+=(--pool-claims "$DPO_POOL_CLAIMS")
fi
echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) script=$train_script run=$run_name seed=$seed output_root=${OUTPUT_ROOT:-verifier/experiments}"
srun python "$train_script" "${args[@]}"
