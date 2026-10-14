#!/usr/bin/env bash
#SBATCH --job-name=eval-verifier
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
bash verifier/evaluation/run_eval.sh
