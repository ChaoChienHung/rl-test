#!/bin/bash
#SBATCH --job-name=fever-index
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f factcheck/README.md
test -f factcheck/data/fever/train.jsonl
test -f factcheck/data/fever/shared_task_dev.jsonl
test -d factcheck/data/fever/wiki-pages
if [[ -e factcheck/data/fever/pages.sqlite || -e factcheck/data/fever/bm25_index ]]; then
    echo "Index artifacts already exist. Inspect them before rebuilding; this script will not overwrite them." >&2
    exit 1
fi
source .venv/bin/activate
echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD)"
cd factcheck/src
srun python build_bm25_index.py
