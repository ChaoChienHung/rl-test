"""Paths for one experiment configuration and its repeatable runs."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEGACY_CONFIG_NAMES = {
    "cot_sft_qwen_3.5-2b_oracle_pilot300": "sft227-cot-from-base-qwen3p5-2b-pilot300",
    "grpo_mt_qwen_3.5-2b_mt_base_full_n16": "grpo16-multiturn-from-base-qwen3p5-2b-kl004",
    "grpo_qwen_3.5-2b_base_full_n1200": "grpo1200-direct-from-base-qwen3p5-2b-kl004",
    "grpo_qwen_3.5-2b_full_n1200": "grpo1200-direct-from-sft5895-qwen3p5-2b-kl0",
    "grpo_qwen_3.5-2b_full_n1200_kl": "grpo926-direct-from-sft5895-qwen3p5-2b-pool1200-kl004",
    "grpo_qwen_3.5-2b_full_n1200_kl05": "grpo926-direct-from-sft5895-qwen3p5-2b-pool1200-kl05",
    "grpo_qwen_3.5-2b_full_n16_smoke": "grpo16-direct-from-sft5895-qwen3p5-2b-kl0",
    "sft_qwen_3.5-2b_continue_n1200": "sft1200-direct-from-sft5895-qwen3p5-2b",
    "sft_qwen_3.5-2b_disjoint_n4695": "sft4695-direct-from-base-qwen3p5-2b-disjoint",
    "sft_qwen_3.5-2b_n1200_from_qwen_3.5-2b_n6000": "sft1200-direct-from-base-qwen3p5-2b-matched",
    "sft_qwen_3.5-2b_n60": "sft60-direct-from-base-qwen3p5-2b",
    "sft_qwen_3.5-2b_n6000": "sft5895-direct-from-base-qwen3p5-2b",
    "sft_qwen_3.5-4b_n6000_qlora": "sft5895-direct-from-base-qwen3p5-4b-qlora",
    "sft_qwen_3.5-4b_n60_qlora": "sft60-direct-from-base-qwen3p5-4b-qlora",
}


def current_experiment_path(path):
    """Translate only the configuration component of a former experiment path."""
    parts = list(Path(path).parts)
    for i in range(len(parts) - 2):
        if parts[i:i + 2] == ["verifier", "experiments"]:
            parts[i + 2] = LEGACY_CONFIG_NAMES.get(parts[i + 2], parts[i + 2])
            break
    return str(Path(*parts))


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
