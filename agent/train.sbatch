#!/usr/bin/env bash
#SBATCH --job-name=agent-train
#SBATCH --partition=gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --output=slurm-%x-%j.out

# Run from the repository root. PHASE=rollouts prepares train-only Qwen draws;
# PHASE=rl updates only the RSI-Jev scorer with terminal reward.
set -euo pipefail
repo_root="$(git -C "${SLURM_SUBMIT_DIR:?}" rev-parse --show-toplevel)"
cd "$repo_root"
test -x .venv/bin/python || { echo 'Missing .venv' >&2; exit 2; }
test -f verifier/data/fever/train.jsonl || { echo 'Missing FEVER train.jsonl' >&2; exit 2; }
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"

phase="${PHASE:-rollouts}"
variant="${CONTROLLER_VARIANT:-citation}"
case "$variant" in citation|selection) ;; *) echo 'CONTROLLER_VARIANT must be citation or selection' >&2; exit 2 ;; esac
run_name="${RUN_NAME:-job-${SLURM_JOB_ID:?}}"
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo 'Invalid RUN_NAME' >&2; exit 2; }
ids_file=verifier/data/train/sft/fever-oracle-direct-5500/labeled_prompts.jsonl
test -f "$ids_file" || { echo "Missing train IDs: $ids_file" >&2; exit 2; }
checkpoints="${CHECKPOINTS:-verifier/experiments/sft-direct-from-base-qwen3p5-2b-5500-samples/runs/run1/artifacts/checkpoints}"
test -f "$checkpoints/adapter_config.json" || { echo "Missing adapter: $checkpoints" >&2; exit 2; }
test -f "$checkpoints/adapter_model.safetensors" || test -f "$checkpoints/adapter_model.bin" || {
  echo "Missing adapter weights: $checkpoints" >&2; exit 2;
}
source_run="${TRAIN_SOURCE_RUN:-agent/data/train_rollouts/sft-direct-from-base-qwen3p5-2b-5500-samples/direct-cap160/stochastic-3-draws-per-chunk-5500-train-claims}"
case "$phase" in
  rollouts)
    test -f verifier/data/fever/pages.sqlite || { echo 'Missing SQLite index' >&2; exit 2; }
    test -f verifier/data/fever/bm25_index/doc_ids.json || { echo 'Missing BM25 index' >&2; exit 2; }
    result_dir="${RESULT_DIR:-$source_run}"
    test ! -e "$result_dir" || { echo "Rollout output exists: $result_dir" >&2; exit 2; }
    echo "Building 5,500 train-only three-draw chunks at $result_dir"
    srun python -m agent.data.build_evidence_batches \
      --split train --claim-ids-file "$ids_file" --n 5500 \
      --model Qwen/Qwen3.5-2B --adapter "$checkpoints" \
      --prompt-style direct --max-new-tokens 160 --batch-size 10 \
      --draws-per-chunk 3 --temperature 0.7 --top-p 0.9 \
      --batches-only --output-dir "$result_dir"
    ;;
  rl)
    test -f "$source_run/metrics.json" || { echo "Missing completed train rollouts: $source_run" >&2; exit 2; }
    python -c 'import rsijev; from serve.release import load_release' || {
      echo 'Install the RSI-Jev GitHub package in .venv first' >&2; exit 2;
    }
    result_dir="${RESULT_DIR:-agent/experiments/jev-rl-$variant-from-rsi-v2p1-5500-samples/runs/$run_name}"
    test ! -e "$result_dir" || { echo "Training output exists: $result_dir" >&2; exit 2; }
    echo "Training variant=$variant on train-only rollouts=$source_run into $result_dir"
    srun python -m agent.train.train_jev_rl \
      --source-run "$source_run" --variant "$variant" --output-dir "$result_dir" \
      --jev-model "${JEV_MODEL:-shgao/rsi-jev-v2.1-qwen3.5-2b}" \
      --epochs "${EPOCHS:-1}" --seed "${SEED:-0}" \
      --lr "${LR:-0.00001}" --kl-coef "${KL_COEF:-0.01}" \
      --entropy-coef "${ENTROPY_COEF:-0.001}"
    ;;
  *) echo "Unsupported PHASE: $phase" >&2; exit 2 ;;
esac
