"""Repeat this historical experiment configuration in a new run."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "verifier/experiments/_shared/grpo.py"
PARAMS = {'prompts_per_step': 2, 'model': 'Qwen/Qwen3.5-2B', 'max_steps': 3, 'n_train': 16, 'tag': 'smoke', 'num_generations': 8, 'lr': 2e-05, 'reward': 'full'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="new run label, e.g. run3")
    parser.add_argument("--output-root", help="experiments directory for run artifacts")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cmd = [sys.executable, str(SCRIPT)]
    for key, value in PARAMS.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value: cmd.append(flag)
        else:
            if key == "subset_of": value = str(ROOT / value)
            cmd.extend([flag, str(value)])
    cmd.extend(["--run-name", args.run_name, "--seed", str(args.seed)])
    if args.output_root:
        cmd.extend(["--output-root", args.output_root])
    subprocess.run(cmd, cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
