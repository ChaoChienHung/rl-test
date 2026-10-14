"""What a trained verifier says on an evidence-removed NEI set.

    python probe_nei_set.py --data data/train/grpo/nei_page_n1200 --adapter <dir> --n 300

A model that answers NEI on most of them treats the set as easy; one that
still answers SUPPORTED/REFUTED is either reading leftover evidence or
guessing from the claim. Writes <data>/probe_<adapter>.jsonl.
"""

import argparse
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from stage0 import LocalModel, adapter_label, label_probs, parse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()

    with open(os.path.join(a.data, "train.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:a.n]
    model = LocalModel(adapter=a.adapter)
    outs = model.run([r["prompt"] for r in rows])
    path = os.path.join(a.data, f"probe_{adapter_label(a.adapter)}.jsonl")
    counts = collections.Counter()
    with open(path, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            obj, ok = parse(o["text"])
            pred = obj.get("verdict") if ok["verdict"] else None
            counts[(r["orig_label"], pred)] += 1
            f.write(json.dumps({"id": r["id"], "orig_label": r["orig_label"], "pred": pred,
                                "label_probs": label_probs(o["tokens"]), "raw": o["text"]}) + "\n")
    n = len(rows)
    nei = sum(v for (_, p), v in counts.items() if p == "NOT_ENOUGH_INFO")
    print(f"{n} claims: NEI {nei / n:.2f}")
    for k, v in sorted(counts.items(), key=str):
        print(f"  {k}: {v}")
    print(f"wrote {path}")
