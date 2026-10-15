"""Paths for one experiment configuration and its repeatable runs."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEGACY_CONFIG_NAMES = {
    # Temporary size-first names used during the 2026-10-15 cleanup.
    "sft60-direct-from-base-qwen3p5-2b": "sft-direct-from-base-qwen3p5-2b-60-samples",
    "sft5895-direct-from-base-qwen3p5-2b": "sft-direct-from-base-qwen3p5-2b-5895-samples",
    "sft4695-direct-from-base-qwen3p5-2b-disjoint": "sft-direct-from-base-qwen3p5-2b-disjoint-4695-samples",
    "sft1200-direct-from-sft5895-qwen3p5-2b": "sft-direct-from-sft5895-qwen3p5-2b-1200-samples",
    "sft227-cot-from-base-qwen3p5-2b-pilot300": "sft-cot-from-base-qwen3p5-2b-pilot300-227-samples",
    "sft1200-direct-from-base-qwen3p5-2b-matched": "sft-direct-from-base-qwen3p5-2b-matched-1200-samples",
    "sft60-direct-from-base-qwen3p5-4b-qlora": "sft-direct-from-base-qwen3p5-4b-qlora-60-samples",
    "sft5895-direct-from-base-qwen3p5-4b-qlora": "sft-direct-from-base-qwen3p5-4b-qlora-5895-samples",
    "grpo16-direct-from-sft5895-qwen3p5-2b-kl0": "grpo-direct-from-sft5895-qwen3p5-2b-kl0-16-samples",
    "grpo1200-direct-from-sft5895-qwen3p5-2b-kl0": "grpo-direct-from-sft5895-qwen3p5-2b-kl0-1200-samples",
    "grpo1200-direct-from-base-qwen3p5-2b-kl004": "grpo-direct-from-base-qwen3p5-2b-kl004-1200-samples",
    "grpo926-direct-from-sft5895-qwen3p5-2b-pool1200-kl004": "grpo-direct-from-sft5895-qwen3p5-2b-pool1200-kl004-926-samples",
    "grpo926-direct-from-sft5895-qwen3p5-2b-pool1200-kl05": "grpo-direct-from-sft5895-qwen3p5-2b-pool1200-kl05-926-samples",
    "grpo16-multiturn-from-base-qwen3p5-2b-kl004": "grpo-multiturn-from-base-qwen3p5-2b-kl004-16-samples",
    "grpo1200-cot-from-sft5895-qwen3p5-2b-no-calibration-kl0": "grpo-cot-from-sft5895-qwen3p5-2b-no-calibration-kl0-1200-samples",
    "grpo1200-cot-from-sft4695-qwen3p5-2b-no-calibration-kl0": "grpo-cot-from-sft4695-qwen3p5-2b-no-calibration-kl0-1200-samples",
    # Names used before sample counts were moved to the end.
    "sft-direct-from-base-qwen3p5-2b-smoke60": "sft-direct-from-base-qwen3p5-2b-60-samples",
    "sft-direct-from-base-qwen3p5-2b-train5895": "sft-direct-from-base-qwen3p5-2b-5895-samples",
    "sft-direct-from-base-qwen3p5-2b-disjoint4695": "sft-direct-from-base-qwen3p5-2b-disjoint-4695-samples",
    "sft-direct-from-sft-qwen3p5-2b-train1200": "sft-direct-from-sft5895-qwen3p5-2b-1200-samples",
    "sft-cot-from-base-qwen3p5-2b-pilot300": "sft-cot-from-base-qwen3p5-2b-pilot300-227-samples",
    "sft-direct-from-base-qwen3p5-2b-matched1200": "sft-direct-from-base-qwen3p5-2b-matched-1200-samples",
    "sft-direct-from-base-qwen3p5-4b-smoke60-qlora": "sft-direct-from-base-qwen3p5-4b-qlora-60-samples",
    "sft-direct-from-base-qwen3p5-4b-train5895-qlora": "sft-direct-from-base-qwen3p5-4b-qlora-5895-samples",
    "grpo-direct-from-sft-qwen3p5-2b-smoke16-kl0": "grpo-direct-from-sft5895-qwen3p5-2b-kl0-16-samples",
    "grpo-direct-from-sft-qwen3p5-2b-train1200-kl0": "grpo-direct-from-sft5895-qwen3p5-2b-kl0-1200-samples",
    "grpo-direct-from-base-qwen3p5-2b-train1200-kl004": "grpo-direct-from-base-qwen3p5-2b-kl004-1200-samples",
    "grpo-direct-from-sft-qwen3p5-2b-filtered1200-kl004": "grpo-direct-from-sft5895-qwen3p5-2b-pool1200-kl004-926-samples",
    "grpo-direct-from-sft-qwen3p5-2b-filtered1200-kl05": "grpo-direct-from-sft5895-qwen3p5-2b-pool1200-kl05-926-samples",
    "grpo1200-cot-from-sft5895-qwen3p5-2b-full-kl0": "grpo-cot-from-sft5895-qwen3p5-2b-no-calibration-kl0-1200-samples",
    "grpo1200-cot-from-sft4695-qwen3p5-2b-full-kl0": "grpo-cot-from-sft4695-qwen3p5-2b-no-calibration-kl0-1200-samples",
    "grpo-multiturn-from-base-qwen3p5-2b-smoke16-kl004": "grpo-multiturn-from-base-qwen3p5-2b-kl004-16-samples",
    "cot_sft_qwen_3.5-2b_oracle_pilot300": "sft-cot-from-base-qwen3p5-2b-pilot300-227-samples",
    "grpo_mt_qwen_3.5-2b_mt_base_full_n16": "grpo-multiturn-from-base-qwen3p5-2b-kl004-16-samples",
    "grpo_qwen_3.5-2b_base_full_n1200": "grpo-direct-from-base-qwen3p5-2b-kl004-1200-samples",
    "grpo_qwen_3.5-2b_full_n1200": "grpo-direct-from-sft5895-qwen3p5-2b-kl0-1200-samples",
    "grpo_qwen_3.5-2b_full_n1200_kl": "grpo-direct-from-sft5895-qwen3p5-2b-pool1200-kl004-926-samples",
    "grpo_qwen_3.5-2b_full_n1200_kl05": "grpo-direct-from-sft5895-qwen3p5-2b-pool1200-kl05-926-samples",
    "grpo_qwen_3.5-2b_full_n16_smoke": "grpo-direct-from-sft5895-qwen3p5-2b-kl0-16-samples",
    "sft_qwen_3.5-2b_continue_n1200": "sft-direct-from-sft5895-qwen3p5-2b-1200-samples",
    "sft_qwen_3.5-2b_disjoint_n4695": "sft-direct-from-base-qwen3p5-2b-disjoint-4695-samples",
    "sft_qwen_3.5-2b_n1200_from_qwen_3.5-2b_n6000": "sft-direct-from-base-qwen3p5-2b-matched-1200-samples",
    "sft_qwen_3.5-2b_n60": "sft-direct-from-base-qwen3p5-2b-60-samples",
    "sft_qwen_3.5-2b_n6000": "sft-direct-from-base-qwen3p5-2b-5895-samples",
    "sft_qwen_3.5-4b_n6000_qlora": "sft-direct-from-base-qwen3p5-4b-qlora-5895-samples",
    "sft_qwen_3.5-4b_n60_qlora": "sft-direct-from-base-qwen3p5-4b-qlora-60-samples",
    "cot_sft_qwen_3.5-2b_oracle_5500": "sft-cot-from-base-qwen3p5-2b-5500-samples",
    "Qwen_Qwen3.5-2B_direct_n5500": "sft-direct-from-base-qwen3p5-2b-5500-samples",
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
