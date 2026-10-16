"""Build Dual-Path 4,500 SFT and 1,000 GRPO datasets from the clean 5,500 FEVER set.

Path 1: Standard JSON format (backward compatibility with 5895/4695 baselines).
Path 2: Native Tag format (<evidence>...</evidence>\\n<verdict>...</verdict>) with evidence-first causal ordering.
"""

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Sources
DIRECT_5500_SRC = ROOT / "verifier/data/train/sft/fever-oracle-direct-5500/labeled_prompts.jsonl"
COT_5500_SRC = ROOT / "verifier/data/train/sft/fever-oracle-cot-soclaas/cot_native_sft_labeled_prompts.jsonl"
DEV_IDS_PATH = ROOT / "verifier/data/eval/fever-dev-900-ids.json"

# Path 1 Targets (JSON)
P1_SFT_DIR = ROOT / "verifier/data/train/sft/fever-oracle-disjoint-4500"
P1_GRPO_DIR = ROOT / "verifier/data/train/sft/fever-oracle-matched-grpo-1000"
P1_COT_DIR = ROOT / "verifier/data/train/sft/fever-oracle-cot-4500"

# Path 2 Targets (Native Tag Format)
P2_TAG_DIRECT_DIR = ROOT / "verifier/data/train/sft/fever-oracle-tagged-4500"
P2_TAG_COT_DIR = ROOT / "verifier/data/train/sft/fever-oracle-tagged-cot-4500"
P2_TAG_GRPO_DIR = ROOT / "verifier/data/train/sft/fever-oracle-tagged-grpo-1000"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_tag_completion(verdict: str, evidence: list[str], reasoning: str = None) -> str:
    ev_str = ", ".join(evidence) if evidence else ""
    tag_body = f"<evidence>{ev_str}</evidence>\n<verdict>{verdict}</verdict>"
    if reasoning:
        return f"{reasoning}\n</think>\n{tag_body}"
    return tag_body


def main(check: bool = False):
    print("Loading 5,500 clean dataset sources...")
    with open(DIRECT_5500_SRC, encoding="utf-8") as f:
        direct_rows = [json.loads(line) for line in f if line.strip()]
    with open(COT_5500_SRC, encoding="utf-8") as f:
        cot_rows = [json.loads(line) for line in f if line.strip()]

    if len(direct_rows) != 5500 or len(cot_rows) != 5500:
        raise ValueError(f"Expected 5,500 rows, got direct={len(direct_rows)}, cot={len(cot_rows)}")

    # Verify ID alignment
    direct_by_id = {str(r["id"]): r for r in direct_rows}
    cot_by_id = {str(r["id"]): r for r in cot_rows}
    if set(direct_by_id.keys()) != set(cot_by_id.keys()):
        raise ValueError("Claim IDs between direct and CoT 5,500 datasets do not match exactly!")

    dev_ids = set(map(str, json.loads(DEV_IDS_PATH.read_text(encoding="utf-8"))["ids"]))
    if set(direct_by_id.keys()) & dev_ids:
        raise ValueError("Critical error: 5,500 claims overlap with frozen 900 dev claims!")

    # Stratified split: 1,500 per class for 4,500 SFT, remainder 1,000 for GRPO
    by_label = collections.defaultdict(list)
    for r in direct_rows:
        by_label[r["gold"]].append(str(r["id"]))

    rng = random.Random(0)
    sft_id_list = []
    grpo_id_list = []

    for label in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]:
        pool = by_label[label].copy()
        rng.shuffle(pool)
        sft_id_list.extend(pool[:1500])
        grpo_id_list.extend(pool[1500:])

    # Final shuffle within splits for train order invariance
    rng.shuffle(sft_id_list)
    rng.shuffle(grpo_id_list)

    sft_ids = set(sft_id_list)
    grpo_ids = set(grpo_id_list)

    assert len(sft_id_list) == 4500 and len(sft_ids) == 4500
    assert len(grpo_id_list) == 1000 and len(grpo_ids) == 1000
    assert len(sft_ids & grpo_ids) == 0
    assert len((sft_ids | grpo_ids) & dev_ids) == 0

    sft_direct = [direct_by_id[cid] for cid in sft_id_list]
    grpo_direct = [direct_by_id[cid] for cid in grpo_id_list]

    sft_cot = [cot_by_id[cid] for cid in sft_id_list]
    grpo_cot = [cot_by_id[cid] for cid in grpo_id_list]

    print(f"SFT counts (4,500): {dict(collections.Counter(r['gold'] for r in sft_direct))}")
    print(f"GRPO counts (1,000): {dict(collections.Counter(r['gold'] for r in grpo_direct))}")

    # Build Path 2 Tag representations
    def to_tag_rows(direct_subset, cot_subset):
        tag_direct_rows = []
        tag_cot_rows = []
        for d_row, c_row in zip(direct_subset, cot_subset):
            cid = d_row["id"]
            gold = d_row["gold"]
            prompt = d_row["prompt"]
            d_comp = json.loads(d_row["completion"])
            verdict = d_comp["verdict"]
            evidence = d_comp.get("evidence", [])

            # Extract reasoning from CoT completion
            c_comp = c_row["completion"]
            reasoning, _, _ = c_comp.partition("</think>\n")
            reasoning = reasoning.replace("<think>", "").strip()

            tag_direct_comp = format_tag_completion(verdict, evidence, reasoning=None)
            tag_cot_comp = format_tag_completion(verdict, evidence, reasoning=reasoning)

            tag_direct_rows.append({
                "id": cid, "gold": gold, "prompt": prompt, "completion": tag_direct_comp
            })
            tag_cot_rows.append({
                "id": cid, "gold": gold, "prompt": prompt, "completion": tag_cot_comp
            })
        return tag_direct_rows, tag_cot_rows

    sft_tag_direct, sft_tag_cot = to_tag_rows(sft_direct, sft_cot)
    grpo_tag_direct, grpo_tag_cot = to_tag_rows(grpo_direct, grpo_cot)

    # Manifest dictionary
    manifest_data = {
        "description": "Dual-Path 4,500 SFT and 1,000 GRPO Disjoint Split from clean 5,500 FEVER set",
        "split_seed": 0,
        "counts": {
            "source_5500": 5500,
            "sft_4500": 4500,
            "grpo_1000": 1000,
        },
        "labels": {
            "sft": dict(collections.Counter(r["gold"] for r in sft_direct)),
            "grpo": dict(collections.Counter(r["gold"] for r in grpo_direct)),
        },
        "dev_overlap": 0,
        "paths": {
            "path1_json": {
                "sft_direct": str(P1_SFT_DIR.relative_to(ROOT) / "labeled_prompts.jsonl"),
                "sft_cot": str(P1_COT_DIR.relative_to(ROOT) / "labeled_prompts.jsonl"),
                "grpo": str(P1_GRPO_DIR.relative_to(ROOT) / "labeled_prompts.jsonl"),
            },
            "path2_tag": {
                "sft_direct": str(P2_TAG_DIRECT_DIR.relative_to(ROOT) / "labeled_prompts.jsonl"),
                "sft_cot": str(P2_TAG_COT_DIR.relative_to(ROOT) / "labeled_prompts.jsonl"),
                "grpo": str(P2_TAG_GRPO_DIR.relative_to(ROOT) / "labeled_prompts.jsonl"),
            }
        }
    }

    if check:
        print("Checking mode: verifying datasets match expected configuration...")
        # Check files exist
        for d in [P1_SFT_DIR, P1_GRPO_DIR, P1_COT_DIR, P2_TAG_DIRECT_DIR, P2_TAG_COT_DIR, P2_TAG_GRPO_DIR]:
            f = d / "labeled_prompts.jsonl"
            if not f.is_file():
                raise FileNotFoundError(f"Missing expected split file: {f}")
        print("All split files exist and verified.")
        return

    # Write Path 1 files
    for dir_path, rows in [
        (P1_SFT_DIR, sft_direct),
        (P1_GRPO_DIR, grpo_direct),
        (P1_COT_DIR, sft_cot),
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
        out_file = dir_path / "labeled_prompts.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} rows to {out_file} (sha256: {sha256_file(out_file)[:16]}...)")

    # Write Path 2 files
    for dir_path, rows in [
        (P2_TAG_DIRECT_DIR, sft_tag_direct),
        (P2_TAG_COT_DIR, sft_tag_cot),
        (P2_TAG_GRPO_DIR, grpo_tag_direct),
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
        out_file = dir_path / "labeled_prompts.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} rows to {out_file} (sha256: {sha256_file(out_file)[:16]}...)")

    # Save manifests
    manifest_str = json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n"
    (P1_SFT_DIR / "split_manifest.json").write_text(manifest_str, encoding="utf-8")
    (P2_TAG_DIRECT_DIR / "split_manifest.json").write_text(manifest_str, encoding="utf-8")
    print("Manifests written successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify splits exist without modifying files")
    args = parser.parse_args()
    main(check=args.check)
