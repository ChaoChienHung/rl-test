"""Build 5,500-sample Native Thinking CoT SFT dataset from validated teacher responses."""

import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COT_SFT_PATH = ROOT / "verifier/data/train/sft/fever-oracle-cot-soclaas/cot_sft_labeled_prompts.jsonl"
OUT_PATH = ROOT / "verifier/data/train/sft/fever-oracle-cot-soclaas/cot_native_sft_labeled_prompts.jsonl"
DEV_IDS_PATH = ROOT / "verifier/data/eval/fever-dev-900-ids.json"


def main():
    if not COT_SFT_PATH.is_file():
        raise FileNotFoundError(f"Missing source CoT SFT data: {COT_SFT_PATH}")

    dev_ids = set(map(str, json.loads(DEV_IDS_PATH.read_text(encoding="utf-8"))["ids"]))
    rows = [json.loads(line) for line in COT_SFT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    print(f"Loaded {len(rows)} rows from {COT_SFT_PATH}")
    if len(rows) != 5500:
        raise ValueError(f"Expected exactly 5,500 rows, got {len(rows)}")

    native_rows = []
    ids = []
    for r in rows:
        cid = str(r["id"])
        ids.append(cid)
        target = json.loads(r["completion"])
        reasoning = target["reasoning"].strip()
        verdict = target["verdict"]
        evidence = target["evidence"]

        if verdict != r["gold"]:
            raise ValueError(f"Verdict mismatch for claim {cid}: {verdict} vs gold {r['gold']}")
        if "confidence" in target:
            raise ValueError(f"Confidence found in teacher target for claim {cid}")

        final_json = json.dumps({"verdict": verdict, "evidence": evidence}, ensure_ascii=False)
        native_completion = f"{reasoning}\n</think>\n{final_json}"

        native_rows.append({
            "id": r["id"],
            "gold": r["gold"],
            "prompt": r["prompt"],
            "completion": native_completion,
        })

    # Validation checks
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate claim IDs found")
    if set(ids) & dev_ids:
        raise ValueError("Claim IDs overlap frozen FEVER dev evaluation IDs")

    label_counts = dict(collections.Counter(r["gold"] for r in native_rows))
    print(f"Native Thinking label counts: {label_counts}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in native_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"Wrote {len(native_rows)} rows to {OUT_PATH} (sha256: {sha})")


if __name__ == "__main__":
    main()
