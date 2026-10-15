#!/usr/bin/env bash
#SBATCH --job-name=cot-brief-retest
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --array=0-1
#SBATCH --output=slurm-%x-%A_%a.out

# Next experiment: matched 900-claim cot_brief retests of the base model
# (task 0) and the completed no-CoT SFT adapter (task 1).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f verifier/evaluation/run_eval.sh || {
  echo "Submit from the repository root" >&2
  exit 2
}

export N_CLAIMS=900
export PROMPT_STYLE=cot_brief
export THINKING=off
export RUN_NAME="${RUN_NAME:-prompt-retest1}"
unset RESULT_DIR

case "${SLURM_ARRAY_TASK_ID:?}" in
  0)
    unset CHECKPOINTS
    export EVAL_CONDITION=base-cot-brief-eval
    ;;
  1)
    export CHECKPOINTS="${SFT_CHECKPOINTS:-verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints}"
    export EVAL_CONDITION=sft-cot-brief-eval
    ;;
  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

echo "array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} condition=$EVAL_CONDITION run=$RUN_NAME"
bash verifier/evaluation/run_eval.sh
