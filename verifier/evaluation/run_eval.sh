#!/usr/bin/env bash
set -euo pipefail

# THINKING selects evidence-chain prompting; model-internal thinking stays disabled.
thinking="${THINKING:-off}"
case "$thinking" in
  off) style=direct; condition=direct-baseline ;;
  on) style=cot_chain; condition=cot-baseline ;;
  *) echo "THINKING must be on or off" >&2; exit 2 ;;
esac

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"
test -f README.md || { echo "Submit from the repository root (README.md missing)" >&2; exit 2; }
test -f verifier/data/fever/shared_task_dev.jsonl || { echo "Missing FEVER dev claims; see README Data preparation" >&2; exit 2; }
test -f verifier/data/fever/pages.sqlite || { echo "Missing SQLite index; run retrieval/build_fever_index.sbatch" >&2; exit 2; }
test -f verifier/data/fever/bm25_index/doc_ids.json || { echo "Missing or incomplete BM25 index; check retrieval/build_fever_index.sbatch" >&2; exit 2; }
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

n="${N_CLAIMS:-900}"
[[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "N_CLAIMS must be a positive integer" >&2; exit 2; }
job_id="${SLURM_JOB_ID:?}"
run_name="${RUN_NAME:-job-$job_id}"
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "RUN_NAME must use letters, digits, underscore or dash" >&2; exit 2; }

checkpoints="${CHECKPOINTS:-}"
adapter_args=()
model="Qwen/Qwen3.5-2B"
if [[ -n "$checkpoints" ]]; then
  [[ -f "$checkpoints/adapter_config.json" ]] || { echo "Missing $checkpoints/adapter_config.json" >&2; exit 2; }
  [[ -f "$checkpoints/adapter_model.safetensors" || -f "$checkpoints/adapter_model.bin" ]] || {
    echo "Missing adapter weights in $checkpoints" >&2; exit 2;
  }
  model="$(python - "$checkpoints/adapter_config.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding='utf-8') as f:
    print(json.load(f)['base_model_name_or_path'])
PY
)"
  adapter_args=(--adapter "$checkpoints")
  condition="checkpoint-${condition%-baseline}"
fi

if [[ -n "${RESULT_DIR:-}" && -n "${RESULTS_ROOT:-}" ]]; then
  echo "Set either RESULT_DIR (exact run directory) or RESULTS_ROOT (parent of condition/run), not both" >&2
  exit 2
fi
if [[ -n "${RESULT_DIR:-}" ]]; then
  result_dir="${RESULT_DIR%/}"
  [[ -n "$result_dir" && "$result_dir" != "." ]] || { echo "RESULT_DIR must name a new run directory" >&2; exit 2; }
else
  results_root="${RESULTS_ROOT:-verifier/evaluation/results}"
  result_dir="${results_root%/}/$condition/$run_name"
fi
if [[ -e "$result_dir" || -L "$result_dir" ]]; then
  echo "Result directory already exists: $result_dir (choose another RUN_NAME or RESULT_DIR)" >&2
  exit 2
fi
mkdir -p "$result_dir/logs"
output="$result_dir/predictions.jsonl"
metrics="$result_dir/metrics.json"
log="$result_dir/logs/run.log"
commit="$(git rev-parse HEAD)"
{
  echo "job=$job_id run=$run_name commit=$commit n=$n thinking=$thinking retrieval=titles model=$model checkpoints=${checkpoints:-none} result_dir=$result_dir"
  echo "started_at=$(date -u +%FT%TZ)"
  srun python -m verifier.evaluation.eval_verifier --model local --n "$n" --retrieval titles \
    --prompt-style "$style" --output "$output" "${adapter_args[@]}"
  if [[ "$n" == 900 ]]; then
    python - "$output" verifier/data/eval/fever-dev-900-ids.json <<'PY'
import json
import sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
expected = json.loads(Path(sys.argv[2]).read_text())['ids']
actual = [row['id'] for row in rows]
if len(actual) != 900 or len(set(actual)) != 900 or set(actual) != set(expected):
    raise SystemExit('900-claim run does not match the frozen FEVER dev ID manifest')
PY
  fi
  python -m verifier.evaluation.score_predictions "$output" > "$metrics"
  echo "finished_at=$(date -u +%FT%TZ)"
  echo "predictions=$output metrics=$metrics"
} 2>&1 | tee "$log"
python - "$result_dir/manifest.json" "$job_id" "$run_name" "$commit" "$model" "$checkpoints" "$thinking" "$style" "$n" "$output" "$metrics" "$log" "$condition" "$result_dir" <<'PY'
import json
import sys
from pathlib import Path

(_, manifest, job_id, run_name, commit, model, checkpoints, thinking,
 style, n, predictions, metrics, log, condition, result_dir) = sys.argv
record = {
    'job_id': job_id, 'run_name': run_name, 'commit': commit,
    'model': model, 'checkpoints': checkpoints or None,
    'thinking': thinking, 'prompt_style': style, 'retrieval': 'titles',
    'condition': condition, 'result_dir': result_dir,
    'n_claims': int(n), 'predictions': predictions, 'metrics': metrics,
    'log': log,
}
Path(manifest).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
PY
