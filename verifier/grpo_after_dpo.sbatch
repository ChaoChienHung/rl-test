#!/usr/bin/env bash
#SBATCH --job-name=grpo-after-dpo
#SBATCH --partition=gpu-long
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
test -f README.md && test -x .venv/bin/python || { echo "Missing repo root or .venv" >&2; exit 2; }
source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false
python -m utils.build_dpo_grpo_split --check

sft="${SFT_CHECKPOINTS:-verifier/experiments/sft-direct-from-base-qwen3p5-2b-disjoint-4500-samples/runs/run1-json-sft/artifacts/checkpoints}"
if [[ -n "${DPO_REPORT:-}" ]]; then
  dpo="$(python - "$DPO_REPORT" <<'PY'
import json, sys
from pathlib import Path
print(Path(json.loads(Path(sys.argv[1]).read_text())['artifacts']) / 'checkpoints')
PY
)"
else
  dpo="${DPO_CHECKPOINTS:?Set DPO_REPORT or DPO_CHECKPOINTS to the completed DPO500 run}"
fi
test -f "$sft/adapter_config.json" && test -f "$dpo/adapter_config.json" || {
  echo "Missing SFT4500 or DPO500 checkpoint" >&2; exit 2;
}
run="${RUN_NAME:-job-${SLURM_JOB_ID:?}}"
[[ "$run" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid RUN_NAME" >&2; exit 2; }
config=grpo-cot-from-sft4500dpo500-qwen3p5-2b-no-calibration-kl0-500-samples
export TRAIN_SCRIPT="verifier/experiments/$config/train.py"
export SFT_CHECKPOINTS="$sft"
export DPO_CHECKPOINTS="$dpo"
export TRAIN_DATA=verifier/data/train/sft/fever-oracle-matched-grpo-1000
export GRPO_IDS_FILE=verifier/data/train/sft/fever-oracle-matched-grpo-1000/dpo-grpo-split.json
export GRPO_IDS_KEY=grpo500_ids
export RUN_NAME="$run"
bash verifier/train.sbatch

grpo="${OUTPUT_ROOT:-verifier/experiments}/$config/runs/$run/artifacts/checkpoints"
export CHECKPOINTS="$sft,$dpo,$grpo"
export EVAL_MODEL="$config"
export PROMPT_STYLE=cot_native
export MAX_NEW_TOKENS=1024
export N_CLAIMS=900
bash verifier/evaluation/run_eval.sh
