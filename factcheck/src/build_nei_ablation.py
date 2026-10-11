"""NEI training claims made by removing the evidence from verifiable claims.

FEVER's own NEI claims are a third of every training set we have built, yet
both 1,200-claim runs are weakest on NEI (SFT 0.64, GRPO 0.51). These claims
are NEI by construction: take a SUPPORTED or REFUTED claim from FEVER train,
build its gold-evidence prompt, then take the evidence away. That is exactly
the situation a retrieval miss puts the verifier in, so the abstain answer
gets practice on the case the search loop will produce.

    python build_nei_ablation.py --mode page       # drop the gold pages (default)
    python build_nei_ablation.py --mode sentence   # drop only the annotated lines

page: every page in any gold set is removed; the prompt holds only the
    title-match distractor pages. Reliably NEI.
sentence: the gold pages stay but every annotated evidence line is removed.
    Harder and noisier: FEVER annotation is incomplete, so another line on the
    page can still settle the claim.

Claims in the SFT probe's 6,000 are excluded. Writes
<out>/train.jsonl in the SFT probe's schema (id, gold, prompt, completion)
plus orig_label, n_shown and n_removed, so sft_probe --subset-of and
grpo_verifier --data read it as is.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

from build_bm25_index import DATA
from grpo_verifier import DEFAULT_SFT
from stage0 import FEVER_LABEL, K_PAGES, MAX_SENTS_PER_PAGE, SEED, SPLIT_FILE, gold_sets, oracle_pages, render

OUT_ROOT = os.path.join(DATA, "..", "grpo_data")
NEI = "NOT_ENOUGH_INFO"


def pick_claims(n, seed, exclude):
    """n verifiable FEVER train claims, half SUPPORTS and half REFUTES."""
    with open(os.path.join(DATA, SPLIT_FILE["train"]), encoding="utf-8") as f:
        claims = [c for c in map(json.loads, f) if c["label"] != "NOT ENOUGH INFO" and c["id"] not in exclude]
    random.Random(seed).shuffle(claims)
    out = []
    for label in ("SUPPORTS", "REFUTES"):
        out += [c for c in claims if c["label"] == label][:n // 2]
    random.Random(seed).shuffle(out)
    return out


def ablate(retriever, claims, mode):
    """Per claim: the shown sentences with the evidence removed, or None if nothing is left."""
    pages = retriever.search_with_titles([c["claim"] for c in claims], k=K_PAGES + 3)
    out = []
    for c, ps in zip(claims, pages):
        gs = gold_sets(c)
        gold_keys = set().union(*gs)
        gold_pages = {p for p, _ in gold_keys}
        if mode == "page":
            # Refill to K_PAGES from the extra retrieved pages so prompt length matches.
            keep = [p for p in dict.fromkeys(ps) if p not in gold_pages][:K_PAGES]
            random.Random(f"{SEED}-{c['id']}").shuffle(keep)
        else:
            keep = oracle_pages(c, ps[:K_PAGES])
        shown, removed = [], 0
        for p in keep:
            for line, text in retriever.sentences(p)[:MAX_SENTS_PER_PAGE]:
                if (p, line) in gold_keys:
                    removed += 1
                    continue
                shown.append((f"E{len(shown) + 1}", p, line, text))
        out.append((shown, removed) if shown else None)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="page", choices=["page", "sentence"])
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--sft", default=DEFAULT_SFT, help="SFT probe dir whose claims to exclude")
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args()

    from fever_retrieval import FeverRetriever
    retriever = FeverRetriever()
    with open(os.path.join(a.sft, "train.jsonl"), encoding="utf-8") as f:
        exclude = {json.loads(line)["id"] for line in f}
    claims = pick_claims(a.n, SEED, exclude)

    out = os.path.join(OUT_ROOT, f"nei_{a.mode}_n{a.n}")
    os.makedirs(out, exist_ok=True)
    stats = Counter()
    with open(os.path.join(out, "train.jsonl"), "w", encoding="utf-8") as f:
        for i in range(0, len(claims), a.chunk):
            batch = claims[i:i + a.chunk]
            for c, x in zip(batch, ablate(retriever, batch, a.mode)):
                if x is None:
                    stats["empty"] += 1
                    continue
                shown, removed = x
                stats[c["label"]] += 1
                stats["removed"] += removed
                f.write(json.dumps({
                    "id": c["id"], "gold": NEI, "prompt": render(c["claim"], shown),
                    "completion": json.dumps({"verdict": NEI, "evidence": [], "confidence": 1.0}),
                    "orig_label": FEVER_LABEL[c["label"]], "n_shown": len(shown), "n_removed": removed,
                }, ensure_ascii=False) + "\n")
            print(f"  {i + len(batch)}/{len(claims)}", flush=True)

    kept = stats["SUPPORTS"] + stats["REFUTES"]
    print(f"\n{kept} claims ({stats['SUPPORTS']} from SUPPORTED, {stats['REFUTES']} from REFUTED), "
          f"{stats['empty']} dropped with nothing left to show; mode={a.mode} -> {out}")
    print(f"  evidence lines removed per claim: {stats['removed'] / max(kept, 1):.1f}")
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"mode": a.mode, "n": a.n, "seed": SEED, "excluded": a.sft, "counts": dict(stats)}, f, indent=1)
