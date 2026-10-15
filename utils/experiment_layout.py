"""Paths for one experiment configuration and its repeatable runs."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def artifacts_dir(kind, legacy_name, seed=0, run_name=None, output_root=None, config_name=None):
    """Keep configuration in the experiment name and repetitions in runs/."""
    stem = legacy_name.replace("Qwen_Qwen3.5-", "qwen_3.5-").lower()
    if seed and stem.endswith(f"_s{seed}"):
        stem = stem[:-(len(str(seed)) + 2)]
    label = run_name or f"run{seed + 1}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
        raise ValueError("run name must use letters, digits, underscore or dash")
    experiments_root = Path(output_root).expanduser().resolve() if output_root else ROOT / "verifier" / "experiments"
    if config_name and not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", config_name):
        raise ValueError("config name must use lowercase letters, digits and hyphens")
    configuration = config_name or f"{kind}_{stem}"
    return experiments_root / configuration / "runs" / label / "artifacts"
