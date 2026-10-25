"""Evaluate OpenJev as a typed FEVER verifier on the same retrieved sentences.

Each sentence is a binary citation decision; verdict is a three-way choice.
Compare a full-context pass with exhaustive evidence batches and a final pass
over every sentence selected by the batch decisions. No Qwen citations are
used by the OpenJev method.
"""

import argparse
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

from core.reward import evidence_f1
from verifier.model import FEVER_LABEL, MAX_CITED, SEED, gold_sets, sample_claims
from agent.evaluation.results.evidence_batches.eval_evidence_batches import quality


MODEL = "alanhuangya/OpenJev-1.7B"
CRITERIA = {
    "SUPPORTED": "The provided Wikipedia evidence fully supports the claim.",
    "REFUTED": "The provided Wikipedia evidence contradicts the claim.",
    "NOT_ENOUGH_INFO": "The provided Wikipedia evidence is insufficient to decide the claim.",
}
VERDICT_QUESTION = {"type": "choice",
                    "instructions": "Decide the claim using only the evidence shown. Choose NOT_ENOUGH_INFO if the evidence is insufficient. Do not use outside knowledge.",
                    "criteria": CRITERIA}


def state(claim, evidence):
    lines = [f"E{i + 1} [{item['id'].rsplit(':', 1)[0].replace('_', ' ')}] {item['text']}"
             for i, item in enumerate(evidence)]
    return f"Claim: {claim}\n\nEvidence:\n" + "\n".join(lines)


def questions(evidence, first_index, include_verdict):
    out = {"verdict": VERDICT_QUESTION} if include_verdict else {}
    for offset, item in enumerate(evidence):
        number = first_index + offset + 1
        out[f"e{number}"] = {
            "type": "noul",
            "instructions": f"Should sentence E{number} be cited as evidence relevant to deciding the claim? Consider it even if other sentences are also needed.",
            "criteria": {"true": f"E{number} is relevant evidence: {item['text']}",
                         "false": f"E{number} should not be cited."},
        }
    return out


def decision(model, claim, evidence, question_batch_size, threshold):
    """Ask verdict and every citation question against one fixed context."""
    input_state = state(claim, evidence)
    started = time.monotonic()
    answers = {}
    calls = 0
    for start in range(0, max(1, len(evidence)), question_batch_size):
        part = evidence[start:start + question_batch_size]
        answer = model.predict(state=input_state,
                               questions=questions(part, start, include_verdict=start == 0))["answers"]
        answers.update(answer)
        calls += 1
    verdict = answers["verdict"]["choice"]
    if verdict not in CRITERIA:
        raise ValueError(f"OpenJev returned unexpected verdict: {verdict!r}")
    probabilities = answers["verdict"].get("probabilities", {})
    relevance = []
    for index, item in enumerate(evidence, 1):
        probability = answers[f"e{index}"]["noul"]
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError(f"OpenJev returned invalid citation probability for E{index}")
        relevance.append({"id": item["id"], "probability": probability})
    return {"pred": verdict, "probabilities": probabilities,
            "confidence": probabilities.get(verdict),
            "relevance": relevance,
            "selected": [r["id"] for r in relevance if r["probability"] >= threshold],
            "n_calls": calls, "seconds": time.monotonic() - started}


def citation_union(shown, batch_decisions):
    selected = {eid for output in batch_decisions for eid in output["selected"]}
    return [item for item in shown if item["id"] in selected]


def final_citations(decision_output):
    """Choose at most FEVER's MAX_CITED sentences from relevant candidates."""
    selected = set(decision_output["selected"])
    ranked = sorted((row for row in decision_output["relevance"]
                     if row["id"] in selected),
                    key=lambda row: -row["probability"])
    return [row["id"] for row in ranked[:MAX_CITED]]


def scored(claim, decision_output):
    cited = final_citations(decision_output)
    gold = FEVER_LABEL[claim["label"]]
    sets = gold_sets(claim)
    cited_pairs = {tuple(eid.rsplit(":", 1)) for eid in cited}
    # Gold line numbers are integers; normalize citations before set comparison.
    cited_pairs = {(page, int(line)) for page, line in cited_pairs}
    gold_ids = sorted({f"{page}:{line}" for group in sets for page, line in group})
    return {"gold": gold, "pred": decision_output["pred"],
            "correct": decision_output["pred"] == gold,
            "fever_score": decision_output["pred"] == gold
            and (not sets or any(group <= cited_pairs for group in sets)),
            "evidence_f1": evidence_f1(gold_ids, cited) if sets else None,
            "format_ok": True, "evidence": cited}


def summarize(rows):
    def group(part):
        full = [row["jev_full_score"] for row in part]
        batched = [row["jev_final_score"] for row in part]
        return {"n": len(part), "jev_full_context": quality(full),
                "jev_evidence_batches": quality(batched),
                "accuracy_gained": sum(not a["correct"] and b["correct"]
                                       for a, b in zip(full, batched)),
                "accuracy_lost": sum(a["correct"] and not b["correct"]
                                     for a, b in zip(full, batched)),
                "fever_gained": sum(not a["fever_score"] and b["fever_score"]
                                    for a, b in zip(full, batched)),
                "fever_lost": sum(a["fever_score"] and not b["fever_score"]
                                  for a, b in zip(full, batched))}

    def mean(values):
        return sum(values) / len(values) if values else None

    return {"overall": group(rows),
            "gold_in_full_context": group([r for r in rows if r["gold_in_context"] is True]),
            "gold_not_in_full_context": group([r for r in rows if r["gold_in_context"] is False]),
            "gold_nei": group([r for r in rows if r["gold"] == "NOT_ENOUGH_INFO"]),
            "mean_jev_union_sentences": mean([r["n_jev_union"] for r in rows]),
            "empty_jev_union_claims": sum(r["n_jev_union"] == 0 for r in rows),
            "mean_full_calls": mean([r["jev_full"]["n_calls"] for r in rows]),
            "mean_batched_calls": mean([sum(d["n_calls"] for d in r["jev_batches"])
                                        + r["jev_final"]["n_calls"] for r in rows]),
            "qwen_source_single_pass": quality([r["qwen_baseline"] for r in rows]),
            "qwen_source_batches": quality([r["qwen_final"] for r in rows]),
            "note": "OpenJev citation probabilities are uncalibrated for FEVER; 0.5 and top-5 are fixed exploratory rules."}


def run(args):
    if not 0 <= args.citation_threshold <= 1 or args.question_batch_size <= 0:
        raise ValueError("citation-threshold must be in [0,1]; question-batch-size must be positive")
    source = Path(args.source_run)
    manifest_path = source / "manifest.json"
    predictions_path = source / "predictions.jsonl"
    if not manifest_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"missing batch run manifest or predictions in {source}")
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("evaluation_code") != "agent.evaluation.results.evidence_batches.eval_evidence_batches":
        raise ValueError("source-run must be an evidence_batches result")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"result directory exists: {output_dir}; use a new run name")
    claims = {claim["id"]: claim for claim in sample_claims(source_manifest["n_claims"], SEED)}
    import wev
    model = wev.load(args.model)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip() or None
    manifest = {"evaluation_code": "agent.evaluation.results.jev_decision.eval_jev_decision",
                "model": args.model, "source_run": str(source),
                "source_commit": source_manifest.get("commit"), "commit": revision,
                "wev_ai_version": importlib.metadata.version("wev-ai"),
                "citation_threshold": args.citation_threshold,
                "question_batch_size": args.question_batch_size,
                "evidence_batch_size": source_manifest["batch_size"],
                "questions": {"verdict": VERDICT_QUESTION,
                              "citation": "Binary relevance decision for every sentence"},
                "note": "OpenJev makes its own citation decisions; Qwen citations are never passed to it."}
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with predictions_path.open(encoding="utf-8") as source_stream, \
            (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as out_stream:
        for index, line in enumerate(source_stream, 1):
            source_row = json.loads(line)
            claim = claims.get(source_row["id"])
            if claim is None or FEVER_LABEL[claim["label"]] != source_row["gold"]:
                raise ValueError(f"source claim {source_row['id']} does not match FEVER dev")
            shown = source_row["shown"]
            full = decision(model, claim["claim"], shown,
                            args.question_batch_size, args.citation_threshold)
            parts = [shown[start:start + source_manifest["batch_size"]]
                     for start in range(0, len(shown), source_manifest["batch_size"])]
            part_decisions = [decision(model, claim["claim"], part,
                                       args.question_batch_size, args.citation_threshold)
                              for part in parts]
            union = citation_union(shown, part_decisions)
            final = decision(model, claim["claim"], union,
                             args.question_batch_size, args.citation_threshold)
            row = {"id": claim["id"], "claim": claim["claim"],
                   "gold": source_row["gold"],
                   "gold_in_context": source_row["gold_in_context"],
                   "n_shown": len(shown), "n_jev_union": len(union),
                   "jev_union": union,
                   "jev_full": full, "jev_full_score": scored(claim, full),
                   "jev_batches": part_decisions,
                   "jev_final": final, "jev_final_score": scored(claim, final),
                   "qwen_baseline": source_row["baseline"],
                   "qwen_final": source_row["final"]}
            rows.append(row)
            out_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_stream.flush()
            if index % 10 == 0:
                print(f"{index} claims evaluated with OpenJev", flush=True)
    if not rows:
        raise ValueError("source run has no predictions")
    metrics = summarize(rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--citation-threshold", type=float, default=0.5)
    parser.add_argument("--question-batch-size", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
