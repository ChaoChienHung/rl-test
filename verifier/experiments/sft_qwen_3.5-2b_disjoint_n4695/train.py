"""Train no-CoT SFT on claims disjoint from the fixed 1,200-claim GRPO set."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "verifier/experiments/_shared/sft.py"
DATA = "verifier/data/train/sft/fever-oracle-disjoint-4695"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(SCRIPT), "--model", "Qwen/Qwen3.5-2B",
                    "--n-train", "4695", "--rank", "16", "--lr", "0.0001",
                    "--data", DATA, "--experiment-name", "Qwen_Qwen3.5-2B_disjoint_n4695",
                    "--run-name", args.run_name, "--seed", str(args.seed),
                    *(["--output-root", args.output_root] if args.output_root else [])],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
