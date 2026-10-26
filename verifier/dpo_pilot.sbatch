#!/usr/bin/env bash
#SBATCH --job-name=dpo-verifier
#SBATCH --partition=gpu-long
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100-40:1
#SBATCH --exclude=xgpe2
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
test -f README.md && test -x .venv/bin/python || { echo "Missing repo root or .venv" >&2; exit 2; }
source .venv/bin/activate
export TMPDIR="${TMPDIR:-$HOME/.pip-tmp}"
mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false

pool="${DPO_POOL:-500}"
[[ "$pool" == 500 || "$pool" == 1000 ]] || { echo "DPO_POOL must be 500 or 1000" >&2; exit 2; }
run="${RUN_NAME:-job-${SLURM_JOB_ID:?}}"
[[ "$run" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid RUN_NAME" >&2; exit 2; }
rollouts="${DPO_ROLLOUTS_DIR:?Set DPO_ROLLOUTS_DIR to the completed 1000-claim rollout directory}"
test -f "$rollouts/candidates.jsonl" && test -f "$rollouts/manifest.json" || {
  echo "Missing rollout candidates or manifest: $rollouts" >&2; exit 2;
}
sft="${SFT_CHECKPOINTS:-verifier/experiments/sft-direct-from-base-qwen3p5-2b-disjoint-4500-samples/runs/run1-json-sft/artifacts/checkpoints}"
test -f "$sft/adapter_config.json" || { echo "Missing SFT4500 adapter: $sft" >&2; exit 2; }
python -m utils.build_dpo_grpo_split --check
python - "$rollouts/manifest.json" "$sft" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
source = Path('verifier/data/train/sft/fever-oracle-matched-grpo-1000/labeled_prompts.jsonl')
expected = {str(json.loads(line)['id']) for line in source.open() if line.strip()}
if manifest['n_claims'] != 1000 or set(manifest['claim_ids']) != expected or len(expected) != 1000:
    raise SystemExit('rollout manifest is not the fixed 1000-claim pool')
if Path(manifest['checkpoint']).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit('rollouts came from a different SFT checkpoint')
PY
work="verifier/data/train/dpo/pool${pool}-$run"
test ! -e "$work/pairs" || { echo "Pair directory exists: $work/pairs" >&2; exit 2; }
split_args=()
if [[ "$pool" == 500 ]]; then
  split_args=(--split verifier/data/train/sft/fever-oracle-matched-grpo-1000/dpo-grpo-split.json --split-key dpo500_ids)
fi
python -m utils.build_dpo_pairs \
  --source verifier/data/train/sft/fever-oracle-matched-grpo-1000/labeled_prompts.jsonl \
  --candidate "$rollouts/candidates.jsonl" --output "$work/pairs" "${split_args[@]}"
pair_count="$(python - "$work/pairs/manifest.json" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())['count'])
PY
)"
min_pairs="${DPO_MIN_PAIRS:-100}"
[[ "$min_pairs" =~ ^[1-9][0-9]*$ ]] || { echo "DPO_MIN_PAIRS must be positive" >&2; exit 2; }
(( pair_count >= min_pairs )) || { echo "Only $pair_count pairs (< $min_pairs); inspect pair coverage" >&2; exit 2; }
export TRAIN_SCRIPT=verifier/experiments/_shared/dpo_cot.py
export SFT_CHECKPOINTS="$sft"
export DPO_PAIRS="$work/pairs/pairs.jsonl"
export DPO_REPORT="$work/train_report.json"
export DPO_POOL_CLAIMS="$pool"
bash verifier/train.sbatch

artifacts="$(python - "$work/train_report.json" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())['artifacts'])
PY
)"
dpo="$artifacts/checkpoints"
config="$(basename "$(dirname "$(dirname "$(dirname "$artifacts")")")")"
export CHECKPOINTS="$sft,$dpo"
export EVAL_MODEL="$config"
export PROMPT_STYLE=cot_native
export MAX_NEW_TOKENS=1024
export N_CLAIMS=900
export RUN_NAME="$run"
bash verifier/evaluation/run_eval.sh
