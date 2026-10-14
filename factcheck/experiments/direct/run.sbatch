#!/usr/bin/env bash
#SBATCH --job-name=verifier-direct
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
bash factcheck/experiments/run_baseline.sh direct
