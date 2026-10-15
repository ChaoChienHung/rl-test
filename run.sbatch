#!/usr/bin/env bash
#SBATCH --job-name=verifier-cap1024
#SBATCH --partition=gpu-long
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --array=0-5%2
#SBATCH --output=slurm-%x-%A_%a.out

# Matched 900-claim evaluation: Base/SFT x direct/cot_chain/cot_brief,
# all with the same 1024-token generation cap.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f verifier/evaluation/run_eval.sh || {
  echo "Submit from the repository root" >&2
  exit 2
}

if [[ -n "${RESULT_DIR:-}" ]]; then
  echo "RESULT_DIR selects one exact directory and cannot be used for this array; use RESULTS_ROOT" >&2
  exit 2
fi

sft_checkpoints="${SFT_CHECKPOINTS:-verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints}"
[[ -f "$sft_checkpoints/adapter_config.json" ]] || {
  echo "Missing SFT adapter config: $sft_checkpoints/adapter_config.json" >&2
  exit 2
}
[[ -s "$sft_checkpoints/adapter_model.safetensors" || -s "$sft_checkpoints/adapter_model.bin" ]] || {
  echo "Missing SFT adapter weights in $sft_checkpoints" >&2
  exit 2
}

export N_CLAIMS=900
export MAX_NEW_TOKENS=1024
export THINKING=off
export RUN_NAME="${RUN_NAME:-run1}"

case "${SLURM_ARRAY_TASK_ID:?}" in
  0) variant=base; style=direct ;;
  1) variant=base; style=cot_chain ;;
  2) variant=base; style=cot_brief ;;
  3) variant=sft; style=direct ;;
  4) variant=sft; style=cot_chain ;;
  5) variant=sft; style=cot_brief ;;
  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

export PROMPT_STYLE="$style"
export EVAL_CONDITION="${variant}-${style}-cap1024"
if [[ "$variant" == sft ]]; then
  export CHECKPOINTS="$sft_checkpoints"
else
  unset CHECKPOINTS
fi

echo "array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} condition=$EVAL_CONDITION run=$RUN_NAME cap=$MAX_NEW_TOKENS"
bash verifier/evaluation/run_eval.sh
