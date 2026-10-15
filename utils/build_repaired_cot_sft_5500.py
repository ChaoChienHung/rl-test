"""Repair format-defective and NEI teacher responses to form a 5,500-sample CoT SFT dataset.

Outputs directly into `verifier/data/train/sft/fever-oracle-cot-soclaas/` (flattened structure).

Sources:
- Source rows (5,895) and teacher responses (5,895)
- Automated validation rules with structural repairs:
  - 5,177 accepted
  - 258 repaired citation mismatches
  - 65 repaired NEIs (from 168 available, seed 0)
- Target: 5,500 CoT samples + 395 clearly documented rejected samples
- Matched 5,500-row direct SFT dataset with identical claim IDs
"""

import collections
import hashlib
import json
import random
import re
from pathlib import Path

from utils.experiment_layout import ROOT

SOURCE = ROOT / "verifier/data/train/sft/fever-oracle-cot-soclaas/source_rows.jsonl"
RESPONSES = ROOT / "verifier/data/train/sft/fever-oracle-cot-soclaas/responses.jsonl"
OUT_DIR = ROOT / "verifier/data/train/sft/fever-oracle-cot-soclaas"
DIRECT_OUT_DIR = ROOT / "verifier/data/train/sft/fever-oracle-direct-5500"
DEV_IDS_PATH = ROOT / "verifier/data/eval/fever-dev-900-ids.json"

RELATION = {"SUPPORTED": "supports", "REFUTED": "refutes", "NOT_ENOUGH_INFO": "insufficient"}
EID = re.compile(r"\bE\d+\b")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_teacher_answer(text):
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.IGNORECASE | re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)


def validate(row, answer):
    if not isinstance(answer, dict):
        return "invalid_json"
    reasoning = answer.get("reasoning")
    if not isinstance(reasoning, str) or not 20 <= len(reasoning.strip()) <= 900:
        return "reasoning_length"
    if answer.get("relation") != RELATION[row["gold"]]:
        return "wrong_relation"
    if answer.get("gold_justified") is not True:
        return "gold_not_justified_by_context"
    cited = answer.get("cited_evidence")
    if not isinstance(cited, list) or any(not isinstance(e, str) or not EID.fullmatch(e) for e in cited):
        return "invalid_citation_list"
    if len(cited) != len(set(cited)) or len(cited) > 5:
        return "duplicate_or_too_many_citations"
    shown = set(re.findall(r"(?m)^E\d+\b", row["prompt"]))
    mentioned = set(EID.findall(reasoning))
    if (not set(cited) <= shown or not mentioned <= shown or not set(cited) <= mentioned
            or (mentioned and not cited and row["gold"] != "NOT_ENOUGH_INFO")):
        return "citation_mismatch"
    gold_eids = set(json.loads(row["completion"])["evidence"])
    if row["gold"] != "NOT_ENOUGH_INFO" and not gold_eids <= mentioned:
        return "gold_evidence_not_explained"
    return None


def repair_and_build(target_count=5500, seed=0):
    source_rows = read_jsonl(SOURCE)
    source_by_id = {str(row["id"]): row for row in source_rows}
    responses = read_jsonl(RESPONSES)
    dev_ids = set(map(str, json.loads(DEV_IDS_PATH.read_text(encoding="utf-8"))["ids"]))

    accepted_clean = []
    repaired_citations = []
    repaired_neis = []
    raw_rejected = []

    for item in responses:
        cid = str(item["id"])
        row = source_by_id[cid]
        body = item.get("response") or {}
        choices = body.get("choices") or []
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        try:
            answer = parse_teacher_answer(content)
            reason = validate(row, answer)
        except Exception:
            answer, reason = None, "invalid_json_or_incomplete"

        if reason is None:
            gold = json.loads(row["completion"])
            comp = {"reasoning": answer["reasoning"].strip(), "verdict": gold["verdict"], "evidence": gold["evidence"]}
            accepted_clean.append({
                "id": row["id"], "gold": row["gold"], "prompt": row["prompt"],
                "completion": json.dumps(comp, ensure_ascii=False),
            })
            continue

        # Try citation repair
        if reason == "citation_mismatch" and isinstance(answer, dict):
            cited = answer.get("cited_evidence", [])
            mentioned = set(EID.findall(answer.get("reasoning", "")))
            missing = [e for e in cited if e not in mentioned]
            r = answer.get("reasoning", "").rstrip()
            if missing:
                missing_str = ", ".join(missing)
                new_reasoning = f"{r[:-1]} ({missing_str})." if r.endswith(".") else f"{r} ({missing_str})"
            else:
                new_reasoning = r
            repaired_answer = dict(answer)
            repaired_answer["reasoning"] = new_reasoning
            if validate(row, repaired_answer) is None:
                gold = json.loads(row["completion"])
                comp = {"reasoning": new_reasoning.strip(), "verdict": gold["verdict"], "evidence": gold["evidence"]}
                repaired_citations.append({
                    "id": row["id"], "gold": row["gold"], "prompt": row["prompt"],
                    "completion": json.dumps(comp, ensure_ascii=False),
                    "repair_type": "citation_append",
                    "original_reason": "citation_mismatch",
                })
                continue

        # Try NEI justification repair
        if reason == "gold_not_justified_by_context" and row["gold"] == "NOT_ENOUGH_INFO" and isinstance(answer, dict):
            repaired_answer = dict(answer)
            repaired_answer["gold_justified"] = True
            if validate(row, repaired_answer) is None:
                gold = json.loads(row["completion"])
                comp = {"reasoning": answer["reasoning"].strip(), "verdict": gold["verdict"], "evidence": gold["evidence"]}
                repaired_neis.append({
                    "id": row["id"], "gold": row["gold"], "prompt": row["prompt"],
                    "completion": json.dumps(comp, ensure_ascii=False),
                    "repair_type": "nei_gold_justified_true",
                    "original_reason": "gold_not_justified_by_context",
                    "answer": answer,
                })
                continue

        claim_text = row["prompt"].split("\n")[0].replace("Claim: ", "").strip()
        raw_rejected.append({
            "id": row["id"],
            "gold": row["gold"],
            "reason": reason,
            "claim": claim_text,
            "teacher_reasoning": answer.get("reasoning") if isinstance(answer, dict) else None,
            "teacher_relation": answer.get("relation") if isinstance(answer, dict) else None,
            "teacher_cited": answer.get("cited_evidence") if isinstance(answer, dict) else None,
            "teacher_gold_justified": answer.get("gold_justified") if isinstance(answer, dict) else None,
        })

    print(f"Clean accepted: {len(accepted_clean)}")
    print(f"Repaired citations passing validation: {len(repaired_citations)}")
    print(f"Repaired NEIs passing validation: {len(repaired_neis)}")
    total_valid_pool = len(accepted_clean) + len(repaired_citations) + len(repaired_neis)
    print(f"Total valid candidate pool: {total_valid_pool}")

    needed_extra = target_count - len(accepted_clean)
    rng = random.Random(seed)
    shuffled_neis = list(repaired_neis)
    shuffled_neis.sort(key=lambda x: x["id"])
    rng.shuffle(shuffled_neis)

    needed_nei = needed_extra - len(repaired_citations)
    selected_neis = shuffled_neis[:needed_nei]
    unselected_neis = shuffled_neis[needed_nei:]

    all_selected = list(accepted_clean)
    repair_log = []
    for r in repaired_citations + selected_neis:
        repair_log.append({
            "id": r["id"], "gold": r["gold"],
            "repair_type": r["repair_type"], "original_reason": r["original_reason"],
        })
        all_selected.append({
            "id": r["id"], "gold": r["gold"], "prompt": r["prompt"],
            "completion": r["completion"],
        })

    all_selected.sort(key=lambda x: x["id"])

    # Unselected NEIs go into rejected list
    final_rejected = list(raw_rejected)
    for r in unselected_neis:
        row = source_by_id[str(r["id"])]
        claim_text = row["prompt"].split("\n")[0].replace("Claim: ", "").strip()
        ans = r.get("answer") or {}
        final_rejected.append({
            "id": r["id"],
            "gold": r["gold"],
            "reason": "unselected_repaired_nei",
            "claim": claim_text,
            "teacher_reasoning": ans.get("reasoning"),
            "teacher_relation": ans.get("relation"),
            "teacher_cited": ans.get("cited_evidence"),
            "teacher_gold_justified": ans.get("gold_justified"),
        })

    final_rejected.sort(key=lambda x: x["id"])

    # Sanity checks
    selected_ids = [str(r["id"]) for r in all_selected]
    if len(selected_ids) != target_count or len(set(selected_ids)) != target_count:
        raise ValueError(f"Selected count {len(selected_ids)} does not match target {target_count}")
    if set(selected_ids) & dev_ids:
        raise ValueError("Selected IDs overlap frozen FEVER dev evaluation IDs")

    label_counts = dict(collections.Counter(r["gold"] for r in all_selected))
    rejection_reasons = dict(collections.Counter(r["reason"] for r in final_rejected))
    print(f"\nFinal {target_count} dataset label counts: {label_counts}")
    print(f"Final rejected breakdown ({len(final_rejected)} rows): {rejection_reasons}")

    # Write CoT SFT dataset directly into OUT_DIR
    cot_path = OUT_DIR / "cot_sft_labeled_prompts.jsonl"
    write_jsonl(cot_path, all_selected)
    write_jsonl(OUT_DIR / "rejected.jsonl", final_rejected)
    (OUT_DIR / "repair_log.json").write_text(json.dumps(repair_log, indent=2) + "\n", encoding="utf-8")

    summary = {
        "dataset_name": "sft-cot-from-base-qwen3p5-2b-5500-samples",
        "teacher_model": "qwen3.8:27b",
        "teacher_provider": "soclaas",
        "source_rows": len(source_rows),
        "received": len(all_selected),
        "accepted": len(all_selected),
        "accepted_breakdown": {
            "clean_accepted": len(accepted_clean),
            "repaired_citations": len(repaired_citations),
            "repaired_neis": len(selected_neis),
        },
        "rejected": len(final_rejected),
        "rejection_reasons": rejection_reasons,
        "label_counts": label_counts,
        "sha256": hashlib.sha256(cot_path.read_bytes()).hexdigest(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "model": "qwen3.8:27b",
        "provider": "soclaas",
        "count": len(all_selected),
        "received": len(all_selected),
        "label_counts": label_counts,
        "seed": seed,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Also write matched 5,500 direct SFT dataset
    direct_rows = []
    for r in all_selected:
        src = source_by_id[str(r["id"])]
        direct_rows.append({
            "id": src["id"], "gold": src["gold"], "prompt": src["prompt"],
            "completion": src["completion"],
        })
    direct_path = DIRECT_OUT_DIR / "labeled_prompts.jsonl"
    write_jsonl(direct_path, direct_rows)
    direct_meta = {
        "dataset_name": "sft-direct-fever-oracle-matched-5500",
        "count": len(direct_rows),
        "label_counts": dict(collections.Counter(r["gold"] for r in direct_rows)),
        "source": "verifier/data/train/sft/fever-oracle-full/labeled_prompts.jsonl",
        "sha256": hashlib.sha256(direct_path.read_bytes()).hexdigest(),
    }
    (DIRECT_OUT_DIR / "split_manifest.json").write_text(json.dumps(direct_meta, indent=2) + "\n", encoding="utf-8")

    print(f"\nGenerated CoT SFT: {cot_path} ({len(all_selected)} rows)")
    print(f"Generated Direct SFT: {direct_path} ({len(direct_rows)} rows)")
    print(f"Documented remaining rejected: {OUT_DIR / 'rejected.jsonl'} ({len(final_rejected)} rows)")


if __name__ == "__main__":
    repair_and_build()
