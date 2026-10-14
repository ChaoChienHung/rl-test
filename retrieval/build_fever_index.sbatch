#!/bin/bash
#SBATCH --job-name=fever-index
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f README.md
test -f verifier/data/fever/train.jsonl
test -f verifier/data/fever/shared_task_dev.jsonl
test -d verifier/data/fever/wiki-pages
if [[ -e verifier/data/fever/pages.sqlite || -e verifier/data/fever/bm25_index ]]; then
    echo "Index artifacts already exist. Inspect them before rebuilding; this script will not overwrite them." >&2
    exit 1
fi
source .venv/bin/activate
echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD)"
srun python -m retrieval.build_bm25_index
