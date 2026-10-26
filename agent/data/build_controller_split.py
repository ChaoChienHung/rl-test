"""Freeze disjoint 3,000/2,500 controller train IDs from the committed 5,500 IDs."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path("verifier/data/train/sft/fever-oracle-direct-5500/labeled_prompts.jsonl")
DEV = Path("verifier/data/eval/fever-dev-900-ids.json")
OUTPUT = Path("agent/data/controller_split/sft3000-rl2500/manifest.json")
LABELS = ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")


def make_manifest(source=SOURCE, dev=DEV, seed=0):
    with source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 5500 or len({row["id"] for row in rows}) != 5500:
        raise ValueError("source must have 5,500 unique claims")
    by_label = defaultdict(list)
    for row in rows:
        if row["gold"] not in LABELS:
            raise ValueError(f"unexpected label: {row['gold']}")
        by_label[row["gold"]].append(row["id"])
    rng = random.Random(seed)
    sft, rl = [], []
    for label in LABELS:
        pool = sorted(by_label[label])
        rng.shuffle(pool)
        if len(pool) < 1000:
            raise ValueError(f"not enough {label} claims for balanced SFT")
        sft.extend(pool[:1000])
        rl.extend(pool[1000:])
    rng.shuffle(sft)
    rng.shuffle(rl)
    dev_ids = set(json.loads(dev.read_text(encoding="utf-8"))["ids"])
    if len(sft) != 3000 or len(rl) != 2500 or set(sft) & set(rl):
        raise ValueError("controller split size or overlap is wrong")
    if (set(sft) | set(rl)) & dev_ids:
        raise ValueError("controller train split overlaps frozen dev")
    labels_by_id = {row["id"]: row["gold"] for row in rows}
    return {"source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "seed": seed,
            "method": "stratified; 1,000 of each FEVER label for SFT, remainder for RL",
            "sft_ids": sft, "rl_ids": rl,
            "counts": {"sft": dict(Counter(labels_by_id[i] for i in sft)),
                       "rl": dict(Counter(labels_by_id[i] for i in rl))},
            "dev_overlap": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = make_manifest()
    if args.check:
        if json.loads(OUTPUT.read_text(encoding="utf-8")) != expected:
            raise ValueError("committed controller split differs from deterministic source")
        print(f"Controller split OK: {len(expected['sft_ids'])} SFT, {len(expected['rl_ids'])} RL")
        return
    if OUTPUT.exists():
        raise FileExistsError(f"split already exists: {OUTPUT}; use --check")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}: {len(expected['sft_ids'])} SFT, {len(expected['rl_ids'])} RL")


if __name__ == "__main__":
    main()
