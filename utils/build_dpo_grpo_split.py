"""Freeze a balanced 500/500 split of the existing SFT4500 holdout pool."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "verifier/data/train/sft/fever-oracle-matched-grpo-1000/labeled_prompts.jsonl"
SFT = ROOT / "verifier/data/train/sft/fever-oracle-disjoint-4500/labeled_prompts.jsonl"
DEV = ROOT / "verifier/data/eval/fever-dev-900-ids.json"
DEST = ROOT / "verifier/data/train/sft/fever-oracle-matched-grpo-1000/dpo-grpo-split.json"


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build():
    pool = records(POOL)
    ids = [str(row["id"]) for row in pool]
    sft_ids = {str(row["id"]) for row in records(SFT)}
    dev_ids = set(map(str, json.loads(DEV.read_text(encoding="utf-8"))["ids"]))
    if len(ids) != 1000 or len(set(ids)) != 1000 or set(ids) & (sft_ids | dev_ids):
        raise ValueError("expected 1,000 unique claims disjoint from SFT4500 and frozen dev")
    groups = {}
    for row in pool:
        groups.setdefault(row["gold"], []).append(str(row["id"]))
    if set(groups) != {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}:
        raise ValueError("unexpected label distribution")
    labels = sorted(groups)
    quotas = {label: len(groups[label]) / 2 for label in labels}
    allocation = {label: int(quotas[label]) for label in labels}
    for label in sorted(labels, key=lambda label: (-(quotas[label] % 1), label))[:500 - sum(allocation.values())]:
        allocation[label] += 1
    dpo_ids = set()
    for label in labels:
        claim_ids = sorted(groups[label], key=int)
        random.Random(0).shuffle(claim_ids)
        dpo_ids.update(claim_ids[:allocation[label]])
    grpo_ids = set(ids) - dpo_ids
    if len(dpo_ids) != 500 or len(grpo_ids) != 500 or dpo_ids & grpo_ids:
        raise AssertionError("split invariant failed")
    label_by_id = {str(row["id"]): row["gold"] for row in pool}
    return {
        "source": str(POOL.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(POOL.read_bytes()).hexdigest(),
        "sft": str(SFT.relative_to(ROOT)), "seed": 0,
        "method": "stratified shuffle within gold labels; half to DPO, remainder to GRPO",
        "dpo500_ids": sorted(dpo_ids, key=int),
        "grpo500_ids": sorted(grpo_ids, key=int),
        "dpo_labels": dict(Counter(label_by_id[i] for i in dpo_ids)),
        "grpo_labels": dict(Counter(label_by_id[i] for i in grpo_ids)),
        "overlap": {"sft": 0, "dev": 0, "dpo_grpo": 0},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    manifest = build()
    if args.check:
        if json.loads(DEST.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("split manifest differs from source or generator")
        print("DPO500/GRPO500 split: OK")
    else:
        if DEST.exists():
            raise SystemExit(f"split already exists: {DEST}; use --check")
        DEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(DEST)


if __name__ == "__main__":
    main()
