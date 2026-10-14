#!/usr/bin/env bash
set -euo pipefail

version="${1:?usage: run_eval.sh direct-baseline|cot-baseline}"
case "$version" in direct-baseline) style=direct;; cot-baseline) style=cot_chain;; *) echo "invalid experiment: $version" >&2; exit 2;; esac
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
test -f README.md
test -f data/fever/shared_task_dev.jsonl
test -f data/fever/pages.sqlite
test -d data/fever/bm25_index
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
n="${N_CLAIMS:-900}"
[[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "N_CLAIMS must be a positive integer" >&2; exit 2; }
job_id="${SLURM_JOB_ID:?}"
run_name="${RUN_NAME:-job-$job_id}"
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "RUN_NAME must use letters, digits, underscore or dash" >&2; exit 2; }
run_dir="experiments/$version/runs/$run_name"
result_dir="results/eval/$version/$run_name"
if [[ -e "$run_dir" || -e "$result_dir" ]]; then
  echo "Run name already exists: $run_name (choose another RUN_NAME)" >&2
  exit 2
fi
mkdir -p "$run_dir/artifacts/logs" "$result_dir"
output="$result_dir/predictions.jsonl"
metrics="$result_dir/metrics.json"
log="$run_dir/artifacts/logs/run.log"
{
  echo "job=$job_id run=$run_name commit=$(git rev-parse HEAD) n=$n style=$style retrieval=titles model=Qwen/Qwen3.5-2B"
  echo "started_at=$(date -u +%FT%TZ)"
  srun python src/stage0.py --model local --n "$n" --retrieval titles --prompt-style "$style" --output "$output"
  if [[ "$n" == 900 ]]; then
    python - "$output" data/eval/fever-dev-900-ids.json <<'PY'
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
  python src/eval_saved_runs.py "$output" > "$metrics"
  echo "finished_at=$(date -u +%FT%TZ)"
  echo "predictions=$output metrics=$metrics"
} 2>&1 | tee "$log"
printf '{"job_id":"%s","run_name":"%s","commit":"%s","model":"Qwen/Qwen3.5-2B","retrieval":"titles","prompt_style":"%s","n_claims":%s,"predictions":"%s","metrics":"%s","log":"%s"}\n' \
  "$job_id" "$run_name" "$(git rev-parse HEAD)" "$style" "$n" "$output" "$metrics" "$log" > "$result_dir/manifest.json"
