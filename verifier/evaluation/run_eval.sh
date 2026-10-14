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
test -f README.md
test -f verifier/data/fever/shared_task_dev.jsonl
test -f verifier/data/fever/pages.sqlite
test -d verifier/data/fever/bm25_index
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
  condition="checkpoint-${style}"
fi

result_dir="verifier/evaluation/results/$condition/$run_name"
if [[ -e "$result_dir" ]]; then
  echo "Run name already exists: $result_dir (choose another RUN_NAME)" >&2
  exit 2
fi
mkdir -p "$result_dir/logs"
output="$result_dir/predictions.jsonl"
metrics="$result_dir/metrics.json"
log="$result_dir/logs/run.log"
commit="$(git rev-parse HEAD)"
{
  echo "job=$job_id run=$run_name commit=$commit n=$n thinking=$thinking retrieval=titles model=$model checkpoints=${checkpoints:-none}"
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
python - "$result_dir/manifest.json" "$job_id" "$run_name" "$commit" "$model" "$checkpoints" "$thinking" "$style" "$n" "$output" "$metrics" "$log" <<'PY'
import json
import sys
from pathlib import Path

(_, manifest, job_id, run_name, commit, model, checkpoints, thinking,
 style, n, predictions, metrics, log) = sys.argv
record = {
    'job_id': job_id, 'run_name': run_name, 'commit': commit,
    'model': model, 'checkpoints': checkpoints or None,
    'thinking': thinking, 'prompt_style': style, 'retrieval': 'titles',
    'n_claims': int(n), 'predictions': predictions, 'metrics': metrics,
    'log': log,
}
Path(manifest).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
PY
