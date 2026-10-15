#!/usr/bin/env bash
# Submit four training jobs; the unseen-claim GRPO job waits for its SFT parent.
set -euo pipefail

cd "$(dirname "$0")/../.."
command -v sbatch >/dev/null || { echo 'sbatch is required; run this on a SoC login node' >&2; exit 2; }
test -x .venv/bin/python || { echo 'Missing .venv/bin/python' >&2; exit 2; }
.venv/bin/python -m utils.build_disjoint_sft_grpo_split --check
[[ -z "${OUTPUT_ROOT:-}" ]] || {
  echo 'Unset OUTPUT_ROOT: this comparison uses fixed checkpoint paths for its dependency' >&2; exit 2;
}

run_tag="${RUN_TAG:-split1-seed0}"
[[ "$run_tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
  echo 'RUN_TAG must use letters, digits, underscore or dash' >&2; exit 2;
}
existing_sft=verifier/experiments/sft-direct-from-base-qwen3p5-2b-train5895/runs/run2/artifacts/checkpoints
new_sft=verifier/experiments/sft-direct-from-base-qwen3p5-2b-disjoint4695/runs/${run_tag}-unseen-sft/artifacts/checkpoints
grpo_data=verifier/data/train/sft/fever-oracle-matched-grpo-1200
test -f "$existing_sft/adapter_config.json" && test -s "$existing_sft/adapter_model.safetensors" || {
  echo "Missing existing SFT adapter in $existing_sft" >&2; exit 2;
}
for output in \
  "verifier/experiments/sft-direct-from-sft-qwen3p5-2b-train1200/runs/${run_tag}-seen-sft/artifacts" \
  "verifier/experiments/grpo-cot-from-sft-qwen3p5-2b-train1200-full-kl0/runs/${run_tag}-seen-cot-grpo/artifacts" \
  "verifier/experiments/sft-direct-from-base-qwen3p5-2b-disjoint4695/runs/${run_tag}-unseen-sft/artifacts" \
  "verifier/experiments/grpo-cot-from-sft-qwen3p5-2b-train1200-full-kl0/runs/${run_tag}-unseen-cot-grpo/artifacts"; do
  test ! -e "$output" || { echo "Run already exists: $output" >&2; exit 2; }
done

seen_sft_job=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 \
  --exclude=xgpe2 --time=03:00:00 \
  --export=ALL,TRAIN_SCRIPT=verifier/experiments/sft-direct-from-sft-qwen3p5-2b-train1200/train.py,RUN_NAME="${run_tag}-seen-sft",SEED=0,INIT_CHECKPOINTS="$existing_sft",SFT_CHECKPOINTS=,TRAIN_DATA= \
  verifier/experiments/run_train.sbatch)
seen_grpo_job=$(sbatch --parsable --partition=gpu-long --gpus=a100-40:1 \
  --exclude=xgpe2 --time=1-00:00:00 \
  --export=ALL,TRAIN_SCRIPT=verifier/experiments/grpo-cot-from-sft-qwen3p5-2b-train1200-full-kl0/train.py,RUN_NAME="${run_tag}-seen-cot-grpo",SEED=0,SFT_CHECKPOINTS="$existing_sft",TRAIN_DATA="$grpo_data",INIT_CHECKPOINTS= \
  verifier/experiments/run_train.sbatch)
unseen_sft_job=$(sbatch --parsable --partition=gpu --gpus=a100-40:1 \
  --exclude=xgpe2 --time=03:00:00 \
  --export=ALL,TRAIN_SCRIPT=verifier/experiments/sft-direct-from-base-qwen3p5-2b-disjoint4695/train.py,RUN_NAME="${run_tag}-unseen-sft",SEED=0,INIT_CHECKPOINTS=,SFT_CHECKPOINTS=,TRAIN_DATA= \
  verifier/experiments/run_train.sbatch)
unseen_grpo_job=$(sbatch --parsable --partition=gpu-long --gpus=a100-40:1 \
  --exclude=xgpe2 --time=1-00:00:00 --dependency="afterok:$unseen_sft_job" \
  --export=ALL,TRAIN_SCRIPT=verifier/experiments/grpo-cot-from-sft-qwen3p5-2b-train1200-full-kl0/train.py,RUN_NAME="${run_tag}-unseen-cot-grpo",SEED=0,SFT_CHECKPOINTS="$new_sft",TRAIN_DATA="$grpo_data",INIT_CHECKPOINTS= \
  verifier/experiments/run_train.sbatch)

printf 'Seen 1,200 -> continued SFT: %s\n' "$seen_sft_job"
printf 'Seen 1,200 -> GRPO:          %s\n' "$seen_grpo_job"
printf 'Unseen 1,200 -> SFT 4,695:  %s\n' "$unseen_sft_job"
printf 'Unseen 1,200 -> GRPO:       %s (afterok:%s)\n' "$unseen_grpo_job" "$unseen_sft_job"
squeue -j "$seen_sft_job,$seen_grpo_job,$unseen_sft_job,$unseen_grpo_job"
