#!/usr/bin/env bash
# Submit the 4,500 SFT multi-condition evaluations to SoC Slurm cluster.
set -euo pipefail

cd "$(dirname "$0")/../.."
command -v sbatch >/dev/null || { echo "sbatch is required; run this on a SoC login node" >&2; exit 2; }

sft_json_cp="verifier/experiments/sft-direct-from-base-qwen3p5-2b-disjoint-4500-samples/runs/run1-json-sft/artifacts/checkpoints"
sft_tag_cp="verifier/experiments/sft-direct-from-base-qwen3p5-2b-tagged-4500-samples/runs/run1-tag-sft/artifacts/checkpoints"

echo "=== Checking 4,500 SFT Checkpoints ==="
[[ -d "$sft_json_cp" ]] || { echo "Missing $sft_json_cp" >&2; exit 2; }
[[ -d "$sft_tag_cp" ]] || { echo "Missing $sft_tag_cp" >&2; exit 2; }

echo "Submitting 5-condition SFT 4,500 evaluation array (Tasks 0-4)..."
job_id=$(sbatch --parsable --array=0-4%5 verifier/evaluation/run_eval_4500_array.sbatch)
echo "Submitted SFT 4,500 evaluation job array: $job_id"
echo "  Task 0: Path 1 (JSON SFT 4500) -> direct (cap 1024)"
echo "  Task 1: Path 1 (JSON SFT 4500) -> cot_brief (cap 1024)"
echo "  Task 2: Path 2 (Tag SFT 4500)  -> direct_tag (cap 1024)"
echo "  Task 3: Path 2 (Tag SFT 4500)  -> cot_brief_tag (cap 1024)"
echo "  Task 4: Path 2 (Tag SFT 4500)  -> cot_chain_tag (cap 1024)"
echo ""
echo "Monitor progress with: squeue --job $job_id"
