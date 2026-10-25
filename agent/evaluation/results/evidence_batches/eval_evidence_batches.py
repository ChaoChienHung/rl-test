"""Verify every evidence batch; optionally run the historical fixed-union evaluation.

The matched single-pass baseline and the batched method use the same claim,
retrieval context, model, adapter, prompt style and deterministic decoding.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from verifier.model import (FEVER_LABEL, LOCAL_MODEL, SEED, LocalModel, build_context, gold_sets,
                            render, sample_claims, score)
from agent.evaluation.results.sampling_retry.eval_sampling_retry import PROMPTS, LABELS, NEI


def batches(shown, size):
    """Renumber prompt-local E ids while retaining page and line identity."""
    for start in range(0, len(shown), size):
        yield [(f"E{i + 1}", page, line, text)
               for i, (_, page, line, text) in enumerate(shown[start:start + size])]


def citation_union(shown, batch_outputs):
    """Keep original retrieval order for every valid, cited global sentence."""
    cited = {(page, line) for output in batch_outputs if output["format_ok"]
             for page, line in output["cited"]}
    selected = [(page, line, text) for _, page, line, text in shown
                if (page, line) in cited]
    return [(f"E{i + 1}", page, line, text)
            for i, (page, line, text) in enumerate(selected)]


def context_for(claim, shown):
    keys = {(page, line) for _, page, line, _ in shown}
    gold = gold_sets(claim)
    return {"shown": shown,
            "gold_in_context": any(group <= keys for group in gold) if gold else None,
            "gold_page_retrieved": None}


def scored(claim, shown, generation):
    item = score(claim, context_for(claim, shown), generation,
                 require_confidence=False)
    fmt = item["format"]
    item["format_ok"] = all(fmt.get(key) for key in ("json", "verdict", "evidence"))
    item["evidence"] = [f"{page}:{line}" for page, line in item["cited"]]
    return item


def quality(items):
    def mean(values):
        return sum(values) / len(values) if values else None

    return {"n": len(items),
            "accuracy": mean([int(s["correct"]) for s in items]),
            "fever_score": mean([int(s["fever_score"]) for s in items]),
            "evidence_f1_non_nei": mean([s["evidence_f1"] for s in items
                                          if s["evidence_f1"] is not None]),
            "format_rate": mean([int(s["format_ok"]) for s in items]),
            "recall": {label: mean([int(s["pred"] == label) for s in items
                                    if s["gold"] == label]) for label in LABELS}}


def summarize(rows):
    def group(part):
        baseline = [row["baseline"] for row in part]
        batched = [row["final"] for row in part]
        return {"n": len(part), "single_pass": quality(baseline),
                "evidence_batches": quality(batched),
                "accuracy_gained": sum(not a["correct"] and b["correct"]
                                       for a, b in zip(baseline, batched)),
                "accuracy_lost": sum(a["correct"] and not b["correct"]
                                     for a, b in zip(baseline, batched)),
                "fever_gained": sum(not a["fever_score"] and b["fever_score"]
                                    for a, b in zip(baseline, batched)),
                "fever_lost": sum(a["fever_score"] and not b["fever_score"]
                                  for a, b in zip(baseline, batched))}

    def mean(values):
        return sum(values) / len(values) if values else None

    def tokens(row, method, field):
        if method == "single_pass":
            return row["baseline"][field]
        return sum(s[field] for s in row["batches"]) + row["final"][field]

    return {"overall": group(rows),
            "gold_in_full_context": group([r for r in rows if r["gold_in_context"] is True]),
            "gold_not_in_full_context": group([r for r in rows if r["gold_in_context"] is False]),
            "gold_nei": group([r for r in rows if r["gold"] == NEI]),
            "mean_batches": mean([len(r["batches"]) for r in rows]),
            "mean_union_sentences": mean([r["n_union"] for r in rows]),
            "empty_union_claims": sum(r["n_union"] == 0 for r in rows),
            "gold_in_union": sum(r["gold_in_union"] is True for r in rows),
            "single_pass_calls": len(rows),
            "batched_calls": sum(len(r["batches"]) + 1 for r in rows),
            "tokens": {method: {field: sum(tokens(r, method, field) for r in rows)
                                for field in ("in_tokens", "out_tokens")}
                       for method in ("single_pass", "evidence_batches")}}


def run(args):
    from retrieval.fever_retrieval import FeverRetriever

    if args.n <= 0 or args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("n, batch-size and max-new-tokens must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"result directory exists: {output_dir}; use a new run name")
    claims = sample_claims(args.n, SEED)
    if args.n == 900:
        frozen = json.loads(Path("verifier/data/eval/fever-dev-900-ids.json").read_text())
        if [claim["id"] for claim in claims] != frozen["ids"]:
            raise ValueError("sampled claims differ from the frozen 900-ID eval manifest")
    retriever = FeverRetriever()
    system, thinking, _ = PROMPTS[args.prompt_style]
    model = LocalModel(args.model, batch_size=args.model_batch_size,
                       adapter=args.adapter, system_prompt=system,
                       max_new_tokens=args.max_new_tokens,
                       enable_thinking=thinking)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip() or None
    manifest = {"evaluation_code": "agent.evaluation.results.evidence_batches.eval_evidence_batches",
                "model": args.model, "adapter": args.adapter,
                "prompt_style": args.prompt_style, "retrieval": "titles",
                "n_claims": args.n, "claim_seed": SEED, "generation_seed": args.seed,
                "batch_size": args.batch_size, "model_batch_size": args.model_batch_size,
                "max_new_tokens": args.max_new_tokens, "do_sample": False,
                "batches_only": args.batches_only,
                "commit": revision,
                "method": ("Run every nonempty evidence batch and save the citation union for later controller decisions. No final verification in this run."
                           if args.batches_only else
                           "Run every nonempty evidence batch; union valid citations in original retrieval order; reverify once. No gold labels or confidence in aggregation.")}
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    started = time.monotonic()
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, claim in enumerate(claims, 1):
            ctx = build_context(retriever, [claim], retrieval="titles")[0]
            shown = ctx["shown"]
            baseline = None
            if not args.batches_only:
                baseline_gen = model.run([render(claim["claim"], shown)],
                                         seed=args.seed + int(claim["id"]))[0]
                baseline = scored(claim, shown, baseline_gen)
            chunks = list(batches(shown, args.batch_size))
            chunk_gens = model.run([render(claim["claim"], part) for part in chunks],
                                   seed=args.seed + int(claim["id"])) if chunks else []
            outputs = [scored(claim, part, generation)
                       for part, generation in zip(chunks, chunk_gens)]
            union = citation_union(shown, outputs)
            final = None
            if not args.batches_only:
                final_gen = model.run([render(claim["claim"], union)],
                                      seed=args.seed + int(claim["id"]))[0]
                final = scored(claim, union, final_gen)
            row = {"id": claim["id"], "claim": claim["claim"],
                   "gold": (baseline["gold"] if baseline else FEVER_LABEL[claim["label"]]),
                   "retrieval": "titles",
                   "n_shown": len(shown), "n_union": len(union),
                   "gold_in_context": ctx["gold_in_context"],
                   "gold_page_retrieved": ctx["gold_page_retrieved"],
                   "gold_in_union": context_for(claim, union)["gold_in_context"],
                   "shown": [{"id": f"{page}:{line}", "text": text}
                             for _, page, line, text in shown],
                   "union": [{"id": f"{page}:{line}", "text": text}
                             for _, page, line, text in union],
                   "batches": outputs}
            if not args.batches_only:
                row["baseline"] = baseline
                row["final"] = final
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 10 == 0 or index == len(claims):
                print(f"{index}/{len(claims)} claims in {time.monotonic() - started:.0f}s", flush=True)
    if args.batches_only:
        def mean(values):
            return sum(values) / len(values) if values else None

        metrics = {"n_claims": len(rows),
                   "batch_verifier_calls": sum(len(r["batches"]) for r in rows),
                   "mean_batches": mean([len(r["batches"]) for r in rows]),
                   "mean_union_sentences": mean([r["n_union"] for r in rows]),
                   "empty_union_claims": sum(r["n_union"] == 0 for r in rows),
                   "gold_in_union": sum(r["gold_in_union"] is True for r in rows),
                   "batch_input_tokens": sum(s["in_tokens"] for r in rows for s in r["batches"]),
                   "batch_output_tokens": sum(s["out_tokens"] for r in rows for s in r["batches"])}
    else:
        metrics = summarize(rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=LOCAL_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt-style", choices=PROMPTS, default="direct")
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--batches-only", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = PROMPTS[args.prompt_style][2]
    run(args)


if __name__ == "__main__":
    main()
