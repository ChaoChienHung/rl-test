#!/usr/bin/env bash
#SBATCH --job-name=agent-eval
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --output=slurm-%x-%j.out

# Submit from anywhere inside this repo; see README for examples.
# Default: sampling_retry on 900 claims. Set EVAL_METHOD=evidence_batches
# for exhaustive 10-sentence batches on the frozen 900 claims.
# Slurm writes slurm-agent-eval-<job-id>.out in the submission directory.

set -euo pipefail
repo_root="$(git -C "${SLURM_SUBMIT_DIR:?missing Slurm submission directory}" rev-parse --show-toplevel)"
cd "$repo_root"
echo "agent evaluation job ${SLURM_JOB_ID:?} started in $PWD"
test -f README.md || { echo 'Repository root has no README.md' >&2; exit 2; }
method="${EVAL_METHOD:-sampling_retry}"
case "$method" in
  sampling_retry) module=agent.evaluation.results.sampling_retry.eval_sampling_retry ;;
  evidence_batches) module=agent.evaluation.results.evidence_batches.eval_evidence_batches ;;
  jev_decision) module=agent.evaluation.results.jev_decision.eval_jev_decision ;;
  *) echo "Unsupported EVAL_METHOD: $method" >&2; exit 2 ;;
esac
test -f "${module//.//}.py" || { echo "Missing evaluator: $module" >&2; exit 2; }
if [[ "$method" != jev_decision ]]; then
  test -f verifier/data/fever/shared_task_dev.jsonl || { echo 'Missing FEVER dev data' >&2; exit 2; }
  test -f verifier/data/fever/pages.sqlite || { echo 'Missing SQLite index' >&2; exit 2; }
  test -f verifier/data/fever/bm25_index/doc_ids.json || { echo 'Missing BM25 index' >&2; exit 2; }
fi
test -x .venv/bin/python || { echo 'Missing .venv; see README' >&2; exit 2; }
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"

if [[ "$method" == jev_decision ]]; then
  source_run="${JEV_SOURCE_RUN:?Set JEV_SOURCE_RUN to a completed evidence_batches run directory}"
  test -f "$source_run/predictions.jsonl" || { echo "Missing source predictions: $source_run" >&2; exit 2; }
  test -f verifier/data/fever/shared_task_dev.jsonl || { echo 'Missing FEVER dev data' >&2; exit 2; }
  python -c 'import wev' || { echo 'Missing wev-ai; install it in .venv' >&2; exit 2; }
  run_name="${RUN_NAME:-job-${SLURM_JOB_ID:?}}"
  [[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo 'Invalid RUN_NAME' >&2; exit 2; }
  result_dir="${RESULT_DIR:-agent/evaluation/results/jev_decision/runs/openjev-1p7b/full-vs-union/$run_name}"
  test ! -e "$result_dir" || { echo "Result directory exists: $result_dir" >&2; exit 2; }
  echo "job=${SLURM_JOB_ID} code=$module source=$source_run result=$result_dir"
  srun python -m "$module" --model "${JEV_MODEL:-alanhuangya/OpenJev-1.7B}" \
    --source-run "$source_run" \
    --citation-threshold "${JEV_CITATION_THRESHOLD:-0.5}" \
    --question-batch-size "${JEV_QUESTION_BATCH_SIZE:-20}" \
    --output-dir "$result_dir"
  exit
fi

n="${N_CLAIMS:-900}"
run_name="${RUN_NAME:-job-${SLURM_JOB_ID:?}}"
style="${PROMPT_STYLE:-direct}"
case "$style" in
  direct|direct_tag|cot_brief|cot_native|cot_native_tag) ;;
  *) echo 'Unsupported PROMPT_STYLE' >&2; exit 2 ;;
esac
[[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo 'N_CLAIMS must be positive' >&2; exit 2; }
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo 'Invalid RUN_NAME' >&2; exit 2; }

checkpoints="${CHECKPOINTS:-verifier/experiments/sft-direct-from-base-qwen3p5-2b-5500-samples/runs/run1/artifacts/checkpoints}"
IFS=, read -r -a adapters <<< "$checkpoints"
for adapter in "${adapters[@]}"; do
  test -f "$adapter/adapter_config.json" || { echo "Missing adapter: $adapter" >&2; exit 2; }
  test -f "$adapter/adapter_model.safetensors" || test -f "$adapter/adapter_model.bin" || {
    echo "Missing adapter weights: $adapter" >&2; exit 2;
  }
done
model="$(python - "${adapters[0]}/adapter_config.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['base_model_name_or_path'])
PY
)"
eval_model="${EVAL_MODEL:-$(python - "${adapters[${#adapters[@]}-1]}" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]).resolve().parts
print(p[p.index('experiments') + 1] if 'experiments' in p else Path(sys.argv[1]).parent.name)
PY
)}"
[[ "$eval_model" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid EVAL_MODEL' >&2; exit 2; }

case "$style" in
  direct|direct_tag) cap="${MAX_NEW_TOKENS:-160}" ;;
  cot_brief) cap="${MAX_NEW_TOKENS:-512}" ;;
  cot_native|cot_native_tag) cap="${MAX_NEW_TOKENS:-1024}" ;;
esac
[[ "$cap" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_NEW_TOKENS must be positive' >&2; exit 2; }
condition="${style//_/-}-cap${cap}"
result_dir="${RESULT_DIR:-agent/evaluation/results/$method/runs/$eval_model/$condition/$run_name}"
test ! -e "$result_dir" || { echo "Result directory exists: $result_dir" >&2; exit 2; }

echo "job=${SLURM_JOB_ID} code=$module model=$model adapters=$checkpoints n=$n style=$style result=$result_dir"
if [[ "$method" == evidence_batches ]]; then
  srun python -m "$module" \
    --model "$model" --adapter "$checkpoints" --prompt-style "$style" \
    --n "$n" --seed "${SEED:-0}" --batch-size "${EVIDENCE_BATCH_SIZE:-10}" \
    --model-batch-size "${MODEL_BATCH_SIZE:-4}" \
    --max-new-tokens "$cap" --output-dir "$result_dir"
else
  srun python -m "$module" \
    --model "$model" --adapter "$checkpoints" --prompt-style "$style" \
    --n "$n" --seed "${SEED:-0}" \
    --temperature "${TEMPERATURE:-0.7}" --top-p "${TOP_P:-0.9}" \
    --threshold "${THRESHOLD:-0.5}" --max-new-tokens "$cap" \
    --output-dir "$result_dir"
fi
