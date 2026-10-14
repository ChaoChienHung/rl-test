"""Paths for one experiment configuration and its repeatable runs."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def artifacts_dir(kind, legacy_name, seed=0, run_name=None, output_root=None):
    """Keep configuration in the experiment name and repetitions in runs/."""
    stem = legacy_name.replace("Qwen_Qwen3.5-", "qwen_3.5-").lower()
    if seed and stem.endswith(f"_s{seed}"):
        stem = stem[:-(len(str(seed)) + 2)]
    label = run_name or f"run{seed + 1}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
        raise ValueError("run name must use letters, digits, underscore or dash")
    experiments_root = Path(output_root).expanduser().resolve() if output_root else ROOT / "verifier" / "experiments"
    return experiments_root / f"{kind}_{stem}" / "runs" / label / "artifacts"
