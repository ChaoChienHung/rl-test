#!/usr/bin/env bash
set -euo pipefail

style="${1:?usage: run_baseline.sh direct|cot_chain}"
case "$style" in direct|cot_chain) ;; *) echo "invalid style: $style" >&2; exit 2;; esac
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
test -f factcheck/README.md
test -f factcheck/data/fever/shared_task_dev.jsonl
test -f factcheck/data/fever/pages.sqlite
test -d factcheck/data/fever/bm25_index
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
n="${N_CLAIMS:-900}"
[[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "N_CLAIMS must be a positive integer" >&2; exit 2; }
version="$style"
[[ "$style" == cot_chain ]] && version="cot-chain"
run_dir="factcheck/results/eval/$version/job-${SLURM_JOB_ID:?}"
mkdir -p "$run_dir"
output="$run_dir/predictions.jsonl"
metrics="$run_dir/metrics.json"
{
  echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) n=$n style=$style retrieval=titles model=Qwen/Qwen3.5-2B"
  echo "started_at=$(date -u +%FT%TZ)"
  srun python factcheck/src/stage0.py --model local --n "$n" --retrieval titles --prompt-style "$style" --output "$output"
  if [[ "$n" == 900 ]]; then
    python - "$output" factcheck/data/eval/fever-dev-900-ids.json <<'PY'
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
  python factcheck/src/eval_saved_runs.py "$output" > "$metrics"
  echo "finished_at=$(date -u +%FT%TZ)"
  echo "predictions=$output metrics=$metrics"
} 2>&1 | tee "$run_dir/run.log"
printf '{"job_id":"%s","commit":"%s","model":"Qwen/Qwen3.5-2B","retrieval":"titles","prompt_style":"%s","n_claims":%s,"predictions":"%s","metrics":"%s"}\n' \
  "$SLURM_JOB_ID" "$(git rev-parse HEAD)" "$style" "$n" "$output" "$metrics" > "$run_dir/manifest.json"
