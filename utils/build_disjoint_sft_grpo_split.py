"""Freeze the 4,695/1,200 claim-disjoint split of prepared FEVER oracle prompts."""

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from utils.experiment_layout import ROOT


SOURCE = ROOT / "verifier/data/train/sft/fever-oracle-full/labeled_prompts.jsonl"
GRPO = ROOT / "verifier/data/train/sft/fever-oracle-matched-grpo-1200/labeled_prompts.jsonl"
SFT = ROOT / "verifier/data/train/sft/fever-oracle-disjoint-4695/labeled_prompts.jsonl"
MANIFEST = SFT.parent / "split_manifest.json"
DEV_IDS = ROOT / "verifier/data/eval/fever-dev-900-ids.json"


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(check=False):
    source = read_rows(SOURCE)
    grpo = read_rows(GRPO)
    source_by_id = {str(row["id"]): row for row in source}
    grpo_by_id = {str(row["id"]): row for row in grpo}
    if len(source) != 5895 or len(source_by_id) != len(source):
        raise ValueError("expected 5,895 unique source claim IDs")
    if len(grpo) != 1200 or len(grpo_by_id) != len(grpo):
        raise ValueError("expected 1,200 unique GRPO claim IDs")
    shuffled = source.copy()
    random.Random(0).shuffle(shuffled)
    if [str(row["id"]) for row in shuffled[:1200]] != [str(row["id"]) for row in grpo]:
        raise ValueError("GRPO subset no longer matches the fixed seed-0 selection")
    for claim_id, row in grpo_by_id.items():
        if source_by_id.get(claim_id) != row:
            raise ValueError(f"GRPO claim {claim_id} is absent or differs from the source")
    sft = [row for row in source if str(row["id"]) not in grpo_by_id]
    if len(sft) != 4695:
        raise ValueError("SFT remainder is not 4,695 rows")
    dev_ids = set(map(str, json.loads(DEV_IDS.read_text(encoding="utf-8"))["ids"]))
    if set(source_by_id) & dev_ids:
        raise ValueError("training claim IDs overlap frozen FEVER dev IDs")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sft)
    if check:
        if not SFT.is_file() or SFT.read_text(encoding="utf-8") != payload:
            raise ValueError("saved SFT split differs from the frozen source and GRPO subset")
    else:
        SFT.parent.mkdir(parents=True, exist_ok=True)
        SFT.write_text(payload, encoding="utf-8")
    manifest = {
        "split": "fever-oracle-disjoint-sft4695-grpo1200",
        "selection": "Existing seed-0 1,200-row GRPO subset; SFT is its claim-ID complement in the 5,895-row source.",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest(SOURCE),
        "sft": str(SFT.relative_to(ROOT)),
        "sft_sha256": digest(SFT),
        "grpo": str(GRPO.relative_to(ROOT)),
        "grpo_sha256": digest(GRPO),
        "counts": {"source": len(source), "sft": len(sft), "grpo": len(grpo)},
        "labels": {
            "sft": dict(collections.Counter(row["gold"] for row in sft)),
            "grpo": dict(collections.Counter(row["gold"] for row in grpo)),
        },
        "dev_claim_id_overlap": 0,
    }
    serialized = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if check:
        if not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != serialized:
            raise ValueError("split manifest differs from the frozen inputs")
    else:
        MANIFEST.write_text(serialized, encoding="utf-8")
    print(f"Verified SFT {len(sft)} + GRPO {len(grpo)} = {len(source)} unique claims; dev overlap 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the committed split without changing files")
    build(parser.parse_args().check)
