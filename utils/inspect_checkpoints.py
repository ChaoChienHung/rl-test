#!/usr/bin/env python3
"""
Inspect and audit verifier checkpoints across experiment directories.
Identifies disk usage, final weights vs intermediate step checkpoints,
and categorizes runs by lifecycle (Core Active, Phase 1 Archive, Legacy/Deletable).
"""

import os
import sys
import argparse
from pathlib import Path


def get_dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(Path(entry.path))
    except Exception:
        pass
    return total


def format_size(bytes_val: int) -> str:
    val = float(bytes_val)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if val < 1024.0:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} PB"


def categorize_config(name: str) -> str:
    # 1. Legacy / Smoke / Toy tests -> Safe to delete
    if any(k in name for k in ["-60-samples", "-16-samples", "4b-qlora", "pool1200"]):
        return "SAFE_TO_DELETE (Legacy / Pilot)"
    # 2. Phase 2 Current Benchmarks (4500 SFT, 1000 GRPO, 5500 SFT)
    if any(k in name for k in ["4500", "1000-samples", "5500-samples"]):
        return "CORE_BENCHMARK (Keep Final Weights)"
    # 3. Phase 1 Splits (4695, 5895, 1200 seen/unseen)
    if any(k in name for k in ["4695", "5895", "1200-samples"]):
        return "PHASE_1_ARCHIVE (Historical Reference)"
    return "OTHER"


def scan_checkpoints(root_dir: Path):
    records = []
    
    # Checkpoints typically live under verifier/experiments/<config>/runs/<run>/artifacts/checkpoints
    pattern = "**/artifacts/checkpoints"
    for cp_dir in root_dir.glob(pattern):
        if not cp_dir.is_dir():
            continue
        
        # Parse hierarchy
        parts = cp_dir.relative_to(root_dir).parts
        # Expected: (verifier, experiments, <config>, runs, <run>, artifacts, checkpoints)
        config_name = "unknown"
        run_name = "unknown"
        for i, p in enumerate(parts):
            if p == "experiments" and i + 1 < len(parts):
                config_name = parts[i + 1]
            if p == "runs" and i + 1 < len(parts):
                run_name = parts[i + 1]

        # Scan contents
        step_dirs = sorted([d for d in cp_dir.glob("checkpoint-*") if d.is_dir()])
        weights = list(cp_dir.glob("*.safetensors")) + list(cp_dir.glob("*.bin")) + list(cp_dir.glob("*.pt"))
        for s in step_dirs:
            weights.extend(list(s.glob("*.safetensors")) + list(s.glob("*.bin")) + list(s.glob("*.pt")))
            
        total_size = get_dir_size(cp_dir)
        step_dirs_size = sum(get_dir_size(s) for s in step_dirs)
        category = categorize_config(config_name)

        records.append({
            "config": config_name,
            "run": run_name,
            "path": cp_dir,
            "category": category,
            "total_size": total_size,
            "step_dirs_size": step_dirs_size,
            "num_steps": len(step_dirs),
            "step_dirs": step_dirs,
            "has_weights": len(weights) > 0,
            "weights_count": len(weights),
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Inspect and clean experiment checkpoints.")
    parser.add_argument("--root", type=str, default=".", help="Root search directory (defaults to repo root)")
    parser.add_argument("--category", type=str, default="", help="Filter by category substring")
    args = parser.parse_args()

    root_path = Path(args.root).resolve()
    print(f"\nScanning for checkpoints under: {root_path} ...\n")
    records = scan_checkpoints(root_path)

    if not records:
        print("No checkpoints found.")
        return

    # Sort by total size descending
    records.sort(key=lambda x: x["total_size"], reverse=True)

    total_bytes = sum(r["total_size"] for r in records)
    total_step_bytes = sum(r["step_dirs_size"] for r in records)

    print(f"{'Category':<32} | {'Size':>10} | {'Interm.':>10} | {'Steps':>5} | Config / Run")
    print("-" * 105)
    for r in records:
        if args.category and args.category.lower() not in r["category"].lower():
            continue
        print(f"{r['category']:<32} | {format_size(r['total_size']):>10} | {format_size(r['step_dirs_size']):>10} | {r['num_steps']:>5} | {r['config']} [{r['run']}]")

    print("-" * 105)
    print(f"Total Checkpoint Storage: {format_size(total_bytes)}")
    print(f"Total Intermediate Steps (Safe to delete if runs finished): {format_size(total_step_bytes)}\n")


if __name__ == "__main__":
    main()
