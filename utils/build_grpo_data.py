"""GRPO prompts with retrieved evidence instead of gold.

The first four GRPO runs trained on the SFT probe's prompts, where the gold
pages are always in context. In the agentic loop the verifier sees whatever
search returned: gold for ~86% of non-NEI claims with title retrieval, with
distractors, and nothing useful for the rest (where SFT accuracy is ~0.42).
This builds the same claims with the verifier evaluator's "titles" retrieval, so GRPO
practises on the evidence the loop will actually hand it.

    python -m utils.build_grpo_data                      # same 1200 claims as grpo_verifier.py
    python verifier/experiments/_shared/grpo.py --data verifier/data/train/grpo/fever-titles-1200 ...

Labels stay FEVER's even when the gold page was not retrieved: that is how
eval_verifier scores the 900 dev claims, and reward.py already prices the choice
(abstaining costs 1.0, waving a false claim through 1.5). gold_ids are the
E-ids of a complete gold set when one is shown, else [].

Writes <out>/labeled_prompts.jsonl in the SFT probe's schema (id, gold, prompt,
completion) plus gold_in_context, so grpo_verifier.load_rows reads it as is.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

from retrieval.build_bm25_index import DATA
from utils.grpo_common import DEFAULT_SFT_DATA
from verifier.model import SEED, SPLIT_FILE, build_context, gold_sets, render

OUT_ROOT = os.path.join(DATA, "..", "train", "grpo")


def same_claims(sft, n, seed):
    """The n claims grpo_verifier.load_rows picks from the SFT probe, in its order."""
    import random
    with open(os.path.join(sft, "labeled_prompts.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    random.Random(seed).shuffle(rows)
    ids = [r["id"] for r in rows[:n]]
    with open(os.path.join(DATA, SPLIT_FILE["train"]), encoding="utf-8") as f:
        by_id = {c["id"]: c for c in map(json.loads, f)}
    return [by_id[i] for i in ids], {r["id"]: r["gold"] for r in rows[:n]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=DEFAULT_SFT_DATA, help="SFT train data dir whose claims to reuse")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--retrieval", default="titles", choices=["bm25", "titles"])
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args()

    from retrieval.fever_retrieval import FeverRetriever
    retriever = FeverRetriever()
    claims, gold = same_claims(a.sft, a.n, SEED)

    out = os.path.join(OUT_ROOT, f"fever-{a.retrieval}-{a.n}")
    os.makedirs(out, exist_ok=True)
    stats = Counter()
    with open(os.path.join(out, "labeled_prompts.jsonl"), "w", encoding="utf-8") as f:
        for i in range(0, len(claims), a.chunk):
            batch = claims[i:i + a.chunk]
            for c, x in zip(batch, build_context(retriever, batch, a.retrieval)):
                g = gold[c["id"]]
                eids = []
                if x["gold_in_context"]:
                    by_key = {(p, line): eid for eid, p, line, _ in x["shown"]}
                    best = next(s for s in gold_sets(c) if s <= set(by_key))
                    eids = sorted((by_key[k] for k in best), key=lambda e: int(e[1:]))
                stats[(g, x["gold_in_context"])] += 1
                f.write(json.dumps({
                    "id": c["id"], "gold": g, "prompt": render(c["claim"], x["shown"]),
                    "completion": json.dumps({"verdict": g, "evidence": eids, "confidence": 1.0}),
                    "gold_in_context": x["gold_in_context"], "n_shown": len(x["shown"]),
                }, ensure_ascii=False) + "\n")
            print(f"  {i + len(batch)}/{len(claims)}", flush=True)

    print(f"\n{len(claims)} claims, {a.retrieval} retrieval -> {out}")
    for g in ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"):
        hit, miss = stats[(g, True)], stats[(g, False)]
        print(f"  {g:16} " + (f"gold in context {hit}, missing {miss}" if g != "NOT_ENOUGH_INFO"
                                 else f"{stats[(g, None)]} (no gold)"))
    with open(os.path.join(out, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"sft_claims": a.sft, "n": a.n, "retrieval": a.retrieval, "seed": SEED,
                   "counts": {f"{g}|{k}": v for (g, k), v in stats.items()}}, f, indent=1)
