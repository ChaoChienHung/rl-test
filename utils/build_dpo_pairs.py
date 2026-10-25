"""Pair actual CoT rollouts for DPO on FEVER training claims.

Each candidate JSONL row needs id, prompt, system_prompt, completion (the
unedited model output), and model. Supply multiple files or generations per
claim. Chosen and rejected must share both prompt messages and output format.
This checks structure and gold labels, not semantic faithfulness of reasoning.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from verifier.model import parse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "verifier/data/train/sft/fever-oracle-full/labeled_prompts.jsonl"
DEV_IDS = ROOT / "verifier/data/eval/fever-dev-900-ids.json"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def output_format(text: str) -> str:
    final = text.rsplit("</think>", 1)[-1]
    return "tag" if "<verdict>" in final.lower() else "json"


def has_reasoning(text: str, parsed: dict) -> bool:
    if "</think>" in text:
        before = text.rsplit("</think>", 1)[0].replace("<think>", "").strip()
        return bool(before)
    return bool(str(parsed.get("reasoning") or "").strip())


def grade(text: str, gold: str, gold_ids: list[str], shown: set[str]):
    parsed, ok = parse(text)
    fmt = output_format(text)
    if not all(ok.get(key) for key in ("json", "verdict", "evidence")):
        return "invalid_format", fmt
    if not has_reasoning(text, parsed):
        return "missing_reasoning", fmt
    cited = parsed["evidence"]
    if len(cited) > 5 or len(set(cited)) != len(cited) or not set(cited) <= shown:
        return "invalid_citations", fmt
    if parsed["verdict"] != gold:
        return "wrong_verdict", fmt
    if set(cited) != set(gold_ids):
        return "wrong_evidence", fmt
    return "correct", fmt


def build(candidate_paths: list[Path], output: Path, source: Path = SOURCE):
    if output.exists():
        raise ValueError(f"output already exists: {output}; choose a new directory")
    source_rows = {str(row["id"]): row for row in read_jsonl(source)}
    dev_ids = set(map(str, json.loads(DEV_IDS.read_text(encoding="utf-8"))["ids"]))
    if set(source_rows) & dev_ids:
        raise ValueError("source train data overlaps frozen dev IDs")

    grouped = collections.defaultdict(list)
    for path in candidate_paths:
        for number, row in enumerate(read_jsonl(path), 1):
            if not isinstance(row, dict) or not all(row.get(k) for k in ("id", "prompt", "system_prompt", "completion", "model")):
                raise ValueError(f"{path}:{number}: require id, prompt, system_prompt, completion and model")
            cid = str(row["id"])
            if cid in dev_ids or cid not in source_rows:
                raise ValueError(f"{path}:{number}: ID {cid} is dev or outside the training set")
            if row["prompt"] != source_rows[cid]["prompt"]:
                raise ValueError(f"{path}:{number}: prompt mismatch for ID {cid}")
            grouped[(cid, row["system_prompt"])].append({
                "completion": row["completion"], "model": row["model"],
                "run": row.get("run"), "file": str(path), "line": number,
            })

    pairs = []
    counts = collections.Counter()
    for (cid, system_prompt), candidates in grouped.items():
        row = source_rows[cid]
        gold_ids = json.loads(row["completion"])["evidence"]
        shown = set(re.findall(r"(?m)^E\d+\b", row["prompt"]))
        by_format = collections.defaultdict(lambda: collections.defaultdict(list))
        for candidate in candidates:
            category, fmt = grade(candidate["completion"], row["gold"], gold_ids, shown)
            counts[category] += 1
            by_format[fmt][category].append(candidate)
        for fmt, outcomes in by_format.items():
            winners = outcomes.get("correct", [])
            losers = [
                item
                for reason in ("wrong_verdict", "wrong_evidence", "invalid_citations", "invalid_format")
                for item in outcomes.get(reason, [])
            ]
            if not winners or not losers:
                continue
            winner = min(winners, key=lambda x: (len(x["completion"]), x["file"], x["line"]))
            loser = next((x for x in losers if x["completion"] != winner["completion"]), None)
            if loser is None:
                continue
            pairs.append({
                "id": cid, "gold": row["gold"], "prompt": row["prompt"],
                "system_prompt": system_prompt,
                "chosen": winner["completion"], "rejected": loser["completion"],
                "format": fmt, "chosen_model": winner["model"],
                "rejected_model": loser["model"], "chosen_run": winner["run"],
                "rejected_run": loser["run"], "chosen_source": winner["file"],
                "rejected_source": loser["file"],
            })

    if not pairs:
        raise ValueError("no genuine correct/incorrect CoT pairs on the same train prompt; no files written")
    pairs.sort(key=lambda row: (int(row["id"]), row["format"]))
    output.mkdir(parents=True)
    pair_path = output / "pairs.jsonl"
    with pair_path.open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    manifest = {
        "count": len(pairs), "candidate_files": [str(p) for p in candidate_paths],
        "candidate_grades": dict(counts), "source": str(source),
        "dev_overlap": 0, "sha256": hashlib.sha256(pair_path.read_bytes()).hexdigest(),
        "caveat": "Structural correctness only; audit reasoning faithfulness before DPO training.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True,
                        help="repeat for each train-claim rollout JSONL")
    parser.add_argument("--output", type=Path, required=True, help="new DPO dataset directory")
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    print(json.dumps(build(args.candidate, args.output, args.source), indent=2))


if __name__ == "__main__":
    main()
