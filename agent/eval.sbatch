#!/usr/bin/env bash
#SBATCH --job-name=agent-eval
#SBATCH --partition=gpu-long
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --output=slurm-%x-%j.out

# Submit from anywhere inside this repo; see README for examples.
# Default: save exhaustive 10-sentence batch data for 900 frozen dev claims.
# Set EVIDENCE_DRAWS=3 for three stochastic verifier draws on each fixed chunk.
# Set EVAL_METHOD=rsi_jev_controller_baseline after the batch job completes.
# Set EVAL_METHOD=rsi_jev_retry_baseline for matched STOP/RETRY policies.
# Set EVAL_METHOD=sampling_retry for the older three-draw evaluator.
# Slurm writes slurm-agent-eval-<job-id>.out in the submission directory.

set -euo pipefail
repo_root="$(git -C "${SLURM_SUBMIT_DIR:?missing Slurm submission directory}" rev-parse --show-toplevel)"
cd "$repo_root"
echo "agent evaluation job ${SLURM_JOB_ID:?} started in $PWD"
test -f README.md || { echo 'Repository root has no README.md' >&2; exit 2; }
method="${EVAL_METHOD:-evidence_batches}"
case "$method" in
  sampling_retry) module=agent.evaluation.results.sampling_retry.eval_sampling_retry ;;
  evidence_batches) module=agent.data.build_evidence_batches ;;
  rsi_jev_controller_baseline|jev_controller) module=agent.evaluation.results.rsi_jev_controller_baseline.eval_rsi_jev_controller_baseline ;;
  rsi_jev_retry_baseline) module=agent.evaluation.results.rsi_jev_retry_baseline.eval_rsi_jev_retry_baseline ;;
  rsi_jev_evidence_selection_baseline) module=agent.evaluation.results.rsi_jev_evidence_selection_baseline.eval_rsi_jev_evidence_selection_baseline ;;
  *) echo "Unsupported EVAL_METHOD: $method" >&2; exit 2 ;;
esac
test -f "${module//.//}.py" || { echo "Missing evaluator: $module" >&2; exit 2; }
if [[ "$method" != rsi_jev_controller_baseline && "$method" != jev_controller && "$method" != rsi_jev_retry_baseline && "$method" != rsi_jev_evidence_selection_baseline ]]; then
  test -f verifier/data/fever/shared_task_dev.jsonl || { echo 'Missing FEVER dev data' >&2; exit 2; }
  test -f verifier/data/fever/pages.sqlite || { echo 'Missing SQLite index' >&2; exit 2; }
  test -f verifier/data/fever/bm25_index/doc_ids.json || { echo 'Missing BM25 index' >&2; exit 2; }
fi
if [[ "$method" == rsi_jev_controller_baseline || "$method" == jev_controller || "$method" == rsi_jev_retry_baseline || "$method" == rsi_jev_evidence_selection_baseline ]]; then
  test -f verifier/data/fever/shared_task_dev.jsonl || { echo 'Missing FEVER dev data for final Qwen scoring' >&2; exit 2; }
fi
test -x .venv/bin/python || { echo 'Missing .venv; see README' >&2; exit 2; }
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"

if [[ "$method" == rsi_jev_controller_baseline || "$method" == jev_controller || "$method" == rsi_jev_retry_baseline || "$method" == rsi_jev_evidence_selection_baseline ]]; then
  source_run="${JEV_SOURCE_RUN:?Set JEV_SOURCE_RUN to a completed evidence_batches run directory}"
  test -f "$source_run/predictions.jsonl" || { echo "Missing source predictions: $source_run" >&2; exit 2; }
  test -f "$source_run/metrics.json" || { echo "Source batch job is incomplete: $source_run/metrics.json is missing" >&2; exit 2; }
  backend="${JEV_BACKEND:-rsi-jev}"
  case "$backend" in
    rsi-jev)
      python -c 'import rsijev' || { echo 'Missing rsi-jev; see agent/evaluation/results/rsi_jev_controller_baseline/README.md' >&2; exit 2; }
      jev_model="${JEV_MODEL:-shgao/rsi-jev-v2.1-qwen3.5-2b}"
      model_dir=rsi-jev-v2p1-qwen3p5-2b ;;
    openjev)
      python -c 'import wev' || { echo 'Missing wev-ai; install it in .venv' >&2; exit 2; }
      jev_model="${JEV_MODEL:-alanhuangya/OpenJev-1.7B}"
      model_dir=openjev-1p7b ;;
    *) echo "Unsupported JEV_BACKEND: $backend" >&2; exit 2 ;;
  esac
  run_name="${RUN_NAME:-job-${SLURM_JOB_ID:?}}"
  [[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo 'Invalid RUN_NAME' >&2; exit 2; }
  if [[ -n "${JEV_TRAINED_HEAD:-}" ]]; then
    test -f "$JEV_TRAINED_HEAD" || { echo "Missing trained head: $JEV_TRAINED_HEAD" >&2; exit 2; }
    model_dir="$(python - "$JEV_TRAINED_HEAD" <<'PY'
import sys
from pathlib import Path
parts = Path(sys.argv[1]).resolve().parts
print(parts[parts.index('experiments') + 1] if 'experiments' in parts else Path(sys.argv[1]).stem)
PY
)"
  fi
  if [[ "$method" == rsi_jev_retry_baseline ]]; then
    default_result_dir="agent/evaluation/results/rsi-jev-retry-baseline/runs/$model_dir/stop-retry-direct-cap160/$run_name"
  elif [[ "$method" == rsi_jev_evidence_selection_baseline ]]; then
    default_result_dir="agent/evaluation/results/rsi-jev-evidence-selection-baseline/runs/$model_dir/keep-drop-direct-cap160/$run_name"
  else
    default_result_dir="agent/evaluation/results/rsi-jev-controller-baseline/runs/$model_dir/route-full-vs-union/$run_name"
  fi
  result_dir="${RESULT_DIR:-$default_result_dir}"
  test ! -e "$result_dir" || { echo "Result directory exists: $result_dir" >&2; exit 2; }
  echo "job=${SLURM_JOB_ID} code=$module backend=$backend model=$jev_model source=$source_run result=$result_dir"
  trained_head_args=()
  if [[ -n "${JEV_TRAINED_HEAD:-}" ]]; then
    trained_head_args=(--trained-head "$JEV_TRAINED_HEAD")
  fi
  pilot_args=()
  if [[ -n "${PILOT_CLAIMS:-}" ]]; then
    [[ "$method" == rsi_jev_evidence_selection_baseline ]] || {
      echo 'PILOT_CLAIMS is only supported for evidence-selection evaluation' >&2; exit 2;
    }
    [[ "$PILOT_CLAIMS" =~ ^[1-9][0-9]*$ ]] || { echo 'PILOT_CLAIMS must be positive' >&2; exit 2; }
    pilot_args=(--max-claims "$PILOT_CLAIMS")
  fi
  srun python -m "$module" --backend "$backend" --model "$jev_model" \
    --source-run "$source_run" --output-dir "$result_dir" \
    "${trained_head_args[@]}" "${pilot_args[@]}"
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
if [[ "$method" == evidence_batches && "${EVIDENCE_BATCHES_ONLY:-1}" == 1 ]]; then
  default_result_dir="agent/data/evidence_batches/$eval_model/$condition/$run_name"
else
  default_result_dir="agent/evaluation/results/$method/runs/$eval_model/$condition/$run_name"
fi
result_dir="${RESULT_DIR:-$default_result_dir}"
test ! -e "$result_dir" || { echo "Result directory exists: $result_dir" >&2; exit 2; }

echo "job=${SLURM_JOB_ID} code=$module model=$model adapters=$checkpoints n=$n style=$style result=$result_dir"
if [[ "$method" == evidence_batches ]]; then
  draws="${EVIDENCE_DRAWS:-1}"
  [[ "$draws" =~ ^[1-9][0-9]*$ ]] || { echo 'EVIDENCE_DRAWS must be positive' >&2; exit 2; }
  batch_only=()
  if [[ "${EVIDENCE_BATCHES_ONLY:-1}" == 1 ]]; then
    batch_only=(--batches-only)
  fi
  srun python -m "$module" \
    --model "$model" --adapter "$checkpoints" --prompt-style "$style" \
    --n "$n" --seed "${SEED:-0}" --batch-size "${EVIDENCE_BATCH_SIZE:-10}" \
    --draws-per-chunk "$draws" \
    --temperature "${TEMPERATURE:-0.7}" --top-p "${TOP_P:-0.9}" \
    --model-batch-size "${MODEL_BATCH_SIZE:-4}" \
    --max-new-tokens "$cap" --output-dir "$result_dir" "${batch_only[@]}"
else
  srun python -m "$module" \
    --model "$model" --adapter "$checkpoints" --prompt-style "$style" \
    --n "$n" --seed "${SEED:-0}" \
    --temperature "${TEMPERATURE:-0.7}" --top-p "${TOP_P:-0.9}" \
    --threshold "${THRESHOLD:-0.5}" --max-new-tokens "$cap" \
    --output-dir "$result_dir"
fi
