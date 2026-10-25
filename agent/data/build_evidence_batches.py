"""Verify every evidence chunk, with optional repeated sampling per fixed chunk.

The one-draw mode preserves the historical fixed-union evaluation. Multi-draw
mode saves all draws for each chunk without making controller decisions.
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


def sample_chunk(model, claim, chunk, chunk_index, draws_per_chunk, seed):
    """Sample the exact same claim and ten-sentence prompt for one chunk."""
    prompt = render(claim["claim"], chunk)
    generations = model.run([prompt] * draws_per_chunk, seed=seed)
    if len(generations) != draws_per_chunk:
        raise ValueError("verifier returned a different number of draws than requested")
    draws = []
    for draw_index, generation in enumerate(generations):
        item = scored(claim, chunk, generation)
        item["draw_index"] = draw_index
        draws.append(item)
    return {
        "chunk_index": chunk_index,
        "sampling_seed": seed,
        "evidence": [{"id": f"{page}:{line}", "text": text}
                     for _, page, line, text in chunk],
        "draws": draws,
    }


def summarize_multi_draw(rows, draws_per_chunk):
    chunks = [chunk for row in rows for chunk in row["chunks"]]
    draws = [draw for chunk in chunks for draw in chunk["draws"]]
    return {
        "n_claims": len(rows),
        "n_chunks": len(chunks),
        "draws_per_chunk": draws_per_chunk,
        "batch_verifier_calls": len(draws),
        "mean_chunks_per_claim": len(chunks) / len(rows) if rows else None,
        "chunks_with_verdict_disagreement": sum(
            len({draw["pred"] for draw in chunk["draws"]}) > 1 for chunk in chunks),
        "chunks_with_evidence_disagreement": sum(
            len({tuple(draw["evidence"]) for draw in chunk["draws"]}) > 1
            for chunk in chunks),
        "chunks_with_any_disagreement": sum(
            len({draw["pred"] for draw in chunk["draws"]}) > 1
            or len({tuple(draw["evidence"]) for draw in chunk["draws"]}) > 1
            for chunk in chunks),
        "batch_input_tokens": sum(draw["in_tokens"] for draw in draws),
        "batch_output_tokens": sum(draw["out_tokens"] for draw in draws),
    }


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


def training_claims(ids_file):
    """Use exactly the committed 5,500 direct-SFT train IDs, never dev IDs."""
    path = Path(ids_file)
    selected = [json.loads(line)["id"] for line in path.open(encoding="utf-8") if line.strip()]
    if len(selected) != 5500 or len(set(selected)) != 5500:
        raise ValueError("train claim-ID file must contain exactly 5,500 unique IDs")
    from verifier.model import DATA
    raw = Path(DATA) / "train.jsonl"
    wanted = set(selected)
    claims = {row["id"]: row for row in (json.loads(line) for line in raw.open(encoding="utf-8"))
              if row["id"] in wanted}
    if set(claims) != wanted:
        raise ValueError("some committed SFT IDs are missing from FEVER train.jsonl")
    return [claims[claim_id] for claim_id in selected]


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

    if args.n <= 0 or args.batch_size <= 0 or args.max_new_tokens <= 0 or args.draws_per_chunk <= 0:
        raise ValueError("n, batch-size, max-new-tokens and draws-per-chunk must be positive")
    if args.draws_per_chunk > 1 and not args.batches_only:
        raise ValueError("multi-draw mode requires --batches-only; no final route is selected")
    if args.draws_per_chunk > 1 and (args.temperature <= 0 or not 0 < args.top_p <= 1):
        raise ValueError("temperature must be positive and top-p must be in (0, 1]")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"result directory exists: {output_dir}; use a new run name")
    split = getattr(args, "split", "dev")
    ids_file = getattr(args, "claim_ids_file", None)
    if split == "train":
        if not ids_file:
            raise ValueError("train split requires --claim-ids-file")
        claims = training_claims(ids_file)
        if args.n != len(claims):
            raise ValueError("--n must match the 5,500 training IDs")
    elif split == "dev" and not ids_file:
        claims = sample_claims(args.n, SEED)
    else:
        raise ValueError("dev split does not accept --claim-ids-file")
    if split == "dev" and args.n == 900:
        frozen = json.loads(Path("verifier/data/eval/fever-dev-900-ids.json").read_text())
        if [claim["id"] for claim in claims] != frozen["ids"]:
            raise ValueError("sampled claims differ from the frozen 900-ID eval manifest")
    retriever = FeverRetriever()
    system, thinking, _ = PROMPTS[args.prompt_style]
    effective_model_batch_size = max(args.model_batch_size, args.draws_per_chunk)
    model = LocalModel(args.model, batch_size=effective_model_batch_size,
                       adapter=args.adapter, system_prompt=system,
                       max_new_tokens=args.max_new_tokens,
                       enable_thinking=thinking, do_sample=args.draws_per_chunk > 1,
                       temperature=args.temperature, top_p=args.top_p)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip() or None
    manifest = {"evaluation_code": "agent.data.build_evidence_batches",
                "split": split, "claim_ids_file": str(ids_file) if ids_file else None,
                "model": args.model, "adapter": args.adapter,
                "prompt_style": args.prompt_style, "retrieval": "titles",
                "n_claims": args.n, "claim_seed": SEED, "generation_seed": args.seed,
                "batch_size": args.batch_size, "model_batch_size": args.model_batch_size,
                "effective_model_batch_size": effective_model_batch_size,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.draws_per_chunk > 1,
                "draws_per_chunk": args.draws_per_chunk,
                "temperature": args.temperature if args.draws_per_chunk > 1 else None,
                "top_p": args.top_p if args.draws_per_chunk > 1 else None,
                "multi_draw_seed_rule": "generation_seed + claim_id*1009 + chunk_index" if args.draws_per_chunk > 1 else None,
                "batches_only": args.batches_only,
                "commit": revision,
                "method": ("Run every nonempty evidence chunk; sample the identical prompt repeatedly and save every draw. No controller selection or final verification."
                           if args.draws_per_chunk > 1 else
                           "Run every nonempty evidence batch and save the citation union for later controller decisions. No final verification in this run."
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
            if args.draws_per_chunk > 1:
                chunk_records = [sample_chunk(
                    model, claim, part, chunk_index, args.draws_per_chunk,
                    args.seed + int(claim["id"]) * 1009 + chunk_index)
                    for chunk_index, part in enumerate(chunks)]
                outputs = []
                union = []
            else:
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
                   "n_shown": len(shown),
                   "gold_in_context": ctx["gold_in_context"],
                   "gold_page_retrieved": ctx["gold_page_retrieved"],
                   "shown": [{"id": f"{page}:{line}", "text": text}
                             for _, page, line, text in shown]}
            if args.draws_per_chunk > 1:
                row["chunks"] = chunk_records
            else:
                row.update({
                    "n_union": len(union),
                    "gold_in_union": context_for(claim, union)["gold_in_context"],
                    "union": [{"id": f"{page}:{line}", "text": text}
                              for _, page, line, text in union],
                    "batches": outputs,
                })
            if not args.batches_only:
                row["baseline"] = baseline
                row["final"] = final
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 10 == 0 or index == len(claims):
                print(f"{index}/{len(claims)} claims in {time.monotonic() - started:.0f}s", flush=True)
    if args.draws_per_chunk > 1:
        metrics = summarize_multi_draw(rows, args.draws_per_chunk)
    elif args.batches_only:
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
    parser.add_argument("--split", choices=("dev", "train"), default="dev")
    parser.add_argument("--claim-ids-file", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--draws-per-chunk", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--batches-only", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = PROMPTS[args.prompt_style][2]
    run(args)


if __name__ == "__main__":
    main()
