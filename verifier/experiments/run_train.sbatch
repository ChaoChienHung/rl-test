#!/usr/bin/env bash
#SBATCH --job-name=train-verifier
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
test -f README.md
source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false

train_script="${TRAIN_SCRIPT:-verifier/experiments/sft_qwen_3.5-2b_n6000/train.py}"
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
echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) script=$train_script run=$run_name seed=$seed output_root=${OUTPUT_ROOT:-verifier/experiments}"
srun python "$train_script" "${args[@]}"
