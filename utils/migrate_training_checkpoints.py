"""Move local, Git-ignored checkpoint weights after training directories were renamed.

Run only after any SoC jobs writing to the former paths have finished. The default
mode lists planned moves; --apply moves files without overwriting existing targets.
"""

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "verifier" / "experiments"
RENAMED = {
    "cot_sft_qwen_3.5-2b_oracle_pilot300": "sft-cot-from-base-qwen3p5-2b-pilot300",
    "grpo_mt_qwen_3.5-2b_mt_base_full_n16": "grpo-multiturn-from-base-qwen3p5-2b-smoke16-kl004",
    "grpo_qwen_3.5-2b_base_full_n1200": "grpo-direct-from-base-qwen3p5-2b-train1200-kl004",
    "grpo_qwen_3.5-2b_full_n1200": "grpo-direct-from-sft-qwen3p5-2b-train1200-kl0",
    "grpo_qwen_3.5-2b_full_n1200_kl": "grpo-direct-from-sft-qwen3p5-2b-filtered1200-kl004",
    "grpo_qwen_3.5-2b_full_n1200_kl05": "grpo-direct-from-sft-qwen3p5-2b-filtered1200-kl05",
    "grpo_qwen_3.5-2b_full_n16_smoke": "grpo-direct-from-sft-qwen3p5-2b-smoke16-kl0",
    "sft_qwen_3.5-2b_continue_n1200": "sft-direct-from-sft-qwen3p5-2b-train1200",
    "sft_qwen_3.5-2b_disjoint_n4695": "sft-direct-from-base-qwen3p5-2b-disjoint4695",
    "sft_qwen_3.5-2b_n1200_from_qwen_3.5-2b_n6000": "sft-direct-from-base-qwen3p5-2b-matched1200",
    "sft_qwen_3.5-2b_n60": "sft-direct-from-base-qwen3p5-2b-smoke60",
    "sft_qwen_3.5-2b_n6000": "sft-direct-from-base-qwen3p5-2b-train5895",
    "sft_qwen_3.5-4b_n6000_qlora": "sft-direct-from-base-qwen3p5-4b-train5895-qlora",
    "sft_qwen_3.5-4b_n60_qlora": "sft-direct-from-base-qwen3p5-4b-smoke60-qlora",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="move checkpoint files; default is a dry run")
    args = parser.parse_args()
    count = 0
    for old, new in RENAMED.items():
        for run in (ROOT / old / "runs").glob("*"):
            source = run / "artifacts" / "checkpoints"
            if not source.is_dir():
                continue
            destination = ROOT / new / "runs" / run.name / "artifacts" / "checkpoints"
            for item in source.iterdir():
                target = destination / item.name
                if target.exists():
                    continue
                print(f"{'MOVE' if args.apply else 'WOULD MOVE'} {item} -> {target}")
                if args.apply:
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(target))
                count += 1
    print(f"{count} checkpoint item(s) {'moved' if args.apply else 'to move'}")


if __name__ == "__main__":
    main()
