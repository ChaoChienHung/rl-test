"""Build DPO contrastive preference pairs (x, y_w, y_l) from teacher rejections.

Constructs preference datasets for Direct Preference Optimization (DPO):
- Chosen (y_w): Ground-truth evidence citations and aligned verdict.
- Rejected (y_l): Teacher error traces (excessive citations, hallucinated justifications, wrong relations).
Outputs both standard JSON and Native Tag formats.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REJECTED_PATH = ROOT / "verifier/data/train/sft/fever-oracle-cot-soclaas/rejected.jsonl"
FULL_PATH = ROOT / "verifier/data/train/sft/fever-oracle-full/labeled_prompts.jsonl"
DEV_IDS_PATH = ROOT / "verifier/data/eval/fever-dev-900-ids.json"

OUT_JSON_DIR = ROOT / "verifier/data/train/dpo/fever-dpo-teacher-rejected-json"
OUT_TAG_DIR = ROOT / "verifier/data/train/dpo/fever-dpo-teacher-rejected-tag"


def relation_to_verdict(relation: str) -> str:
    mapping = {
        "supports": "SUPPORTED",
        "refutes": "REFUTED",
        "insufficient": "NOT_ENOUGH_INFO",
    }
    return mapping.get(relation.lower(), "NOT_ENOUGH_INFO")


def build():
    print("Loading rejected teacher traces and full oracle prompts...")
    with open(REJECTED_PATH, encoding="utf-8") as f:
        rejected_rows = [json.loads(line) for line in f if line.strip()]

    with open(FULL_PATH, encoding="utf-8") as f:
        full_rows = [json.loads(line) for line in f if line.strip()]

    dev_ids = set(map(str, json.loads(DEV_IDS_PATH.read_text(encoding="utf-8"))["ids"]))

    full_by_id = {str(r["id"]): r for r in full_rows}
    dpo_pairs_json = []
    dpo_pairs_tag = []

    reason_counts = collections.Counter()

    for r in rejected_rows:
        cid = str(r["id"])
        if cid in dev_ids:
            continue  # Strict 0 dev overlap guarantee

        if cid not in full_by_id:
            continue

        full_row = full_by_id[cid]
        gold_comp = json.loads(full_row["completion"])
        gold_verdict = full_row["gold"]
        gold_evidence = gold_comp.get("evidence", [])

        # Construct rejected completion (y_l)
        reasoning = (r.get("teacher_reasoning") or "").strip()
        rej_relation = r.get("teacher_relation") or ""
        rej_verdict = relation_to_verdict(rej_relation)
        rej_evidence = r.get("teacher_cited") or []

        reason = r.get("reason", "teacher_rejection")
        reason_counts[reason] += 1

        # JSON formats
        chosen_json = (
            f"{reasoning}\n</think>\n" +
            json.dumps({"verdict": gold_verdict, "evidence": gold_evidence}, ensure_ascii=False)
        )
        rejected_json = (
            f"{reasoning}\n</think>\n" +
            json.dumps({"verdict": rej_verdict, "evidence": rej_evidence}, ensure_ascii=False)
        )

        # Tag formats
        gold_ev_str = ", ".join(gold_evidence) if gold_evidence else ""
        rej_ev_str = ", ".join(rej_evidence) if rej_evidence else ""

        chosen_tag = f"{reasoning}\n</think>\n<evidence>{gold_ev_str}</evidence>\n<verdict>{gold_verdict}</verdict>"
        rejected_tag = f"{reasoning}\n</think>\n<evidence>{rej_ev_str}</evidence>\n<verdict>{rej_verdict}</verdict>"

        dpo_pairs_json.append({
            "id": int(cid) if cid.isdigit() else cid,
            "gold": gold_verdict,
            "prompt": full_row["prompt"],
            "chosen": chosen_json,
            "rejected": rejected_json,
            "rejection_reason": reason,
        })

        dpo_pairs_tag.append({
            "id": int(cid) if cid.isdigit() else cid,
            "gold": gold_verdict,
            "prompt": full_row["prompt"],
            "chosen": chosen_tag,
            "rejected": rejected_tag,
            "rejection_reason": reason,
        })

    # Save outputs
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_JSON_DIR / "pairs.jsonl"
    with open(json_path, "w", encoding="utf-8") as f:
        for p in dpo_pairs_json:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    OUT_TAG_DIR.mkdir(parents=True, exist_ok=True)
    tag_path = OUT_TAG_DIR / "pairs.jsonl"
    with open(tag_path, "w", encoding="utf-8") as f:
        for p in dpo_pairs_tag:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    manifest = {
        "description": "DPO preference pairs constructed from teacher rejection traces",
        "count": len(dpo_pairs_json),
        "rejection_reasons": dict(reason_counts),
        "dev_overlap": 0,
        "json_path": str(json_path.relative_to(ROOT)),
        "tag_path": str(tag_path.relative_to(ROOT)),
    }
    (OUT_JSON_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (OUT_TAG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"Constructed {len(dpo_pairs_json)} DPO preference pairs.")
    print(f"Rejection reason breakdown: {dict(reason_counts)}")
    print(f"Zero dev overlap confirmed.")


if __name__ == "__main__":
    build()
