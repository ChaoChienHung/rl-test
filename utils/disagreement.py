"""Which training claims give GRPO a signal: sample the SFT model 8 times per claim.

GRPO learns only from groups whose completions score differently. The first
full run (verifier/experiments/grpo-direct-from-sft5895-qwen3p5-2b-kl0-1200-samples/runs/run1) had ~88% of groups with zero
reward spread, so it barely moved. This pass samples each training claim the
way GRPO does (temperature 1, same prompt and reward) and records the spread,
so the next run can train only on claims where the samples disagree.

    python -m utils.disagreement                     # 1200 claims, same order as grpo_verifier.py
    python -m utils.disagreement --n-train 32        # smoke test

Writes verifier/data/train/grpo/disagreement_<adapter>_n<N>.jsonl, one row per claim, in
load_rows order (row i = prompt i), plus a summary on stdout.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from core import reward as R
from utils.grpo_common import DEFAULT_SFT, DEFAULT_SFT_DATA, VERDICT, load_rows, score_one, spread_path
from verifier.model import LOCAL_MODEL, SEED, parse


def sample(model, tok, rows, k, batch_prompts):
    import torch
    out = []
    for i in range(0, len(rows), batch_prompts):
        chunk = rows[i:i + batch_prompts]
        texts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for r in chunk]
        inp = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=96, do_sample=True, temperature=1.0,
                                 top_p=1.0, top_k=0, num_return_sequences=k)
        n_in = inp["input_ids"].shape[1]
        dec = tok.batch_decode(gen[:, n_in:], skip_special_tokens=True)
        out += [dec[j * k:(j + 1) * k] for j in range(len(chunk))]
        if (i // batch_prompts) % 25 == 0:
            print(f"  {i + len(chunk)}/{len(rows)}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--sft", default=DEFAULT_SFT)
    ap.add_argument("--data", default=None, help="prompt dir (default: the SFT probe's)")
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--k", type=int, default=8, help="samples per claim (GRPO's num_generations)")
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, a.sft).merge_and_unload().eval()
    torch.manual_seed(a.seed)

    a.data = a.data or DEFAULT_SFT_DATA
    rows = load_rows(a.data, a.n_train, a.seed)
    t0 = time.time()
    samples = sample(model, tok, rows, a.k, a.batch_prompts)
    secs = time.time() - t0

    rng = random.Random(a.seed)
    path = spread_path(a.data, a.n_train)
    stats = []
    with open(path, "w", encoding="utf-8") as f:
        for i, (r, texts) in enumerate(zip(rows, samples)):
            rewards, verdicts, confs = [], [], []
            for t in texts:
                rew, _ = score_one(t, r["gold"], r["gold_ids"], R.FULL, rng)
                obj, ok = parse(t)
                rewards.append(rew)
                verdicts.append(obj.get("verdict") if ok["verdict"] else None)
                confs.append(obj.get("confidence") if ok["confidence"] else None)
            row = {"i": i, "gold": r["gold"], "rewards": rewards, "verdicts": verdicts, "confidences": confs,
                   "reward_std": statistics.pstdev(rewards), "n_distinct_verdicts": len(set(verdicts)),
                   "acc": sum(VERDICT.get(v) == r["gold"] for v in verdicts) / a.k}
            stats.append(row)
            f.write(json.dumps(row) + "\n")

    n = len(stats)
    spread = [s for s in stats if s["reward_std"] > 1e-9]
    disagree = [s for s in stats if s["n_distinct_verdicts"] > 1]
    confs = [c for s in stats for c in s["confidences"] if c is not None]
    print(f"\n{n} claims x {a.k} samples in {secs / 60:.1f} min")
    print(f"  reward spread > 0:      {len(spread)} ({len(spread) / n:.0%})")
    print(f"  verdicts disagree:      {len(disagree)} ({len(disagree) / n:.0%})")
    print(f"  all 8 right / all wrong: {sum(s['acc'] == 1 for s in stats)} / {sum(s['acc'] == 0 for s in stats)}")
    print(f"  stated confidence: distinct values {sorted(set(confs))[:10]}, share 1.0 = "
          f"{sum(c == 1 for c in confs) / max(len(confs), 1):.2f}")
    by_gold = {}
    for s in stats:
        g = by_gold.setdefault(s["gold"], [0, 0])
        g[0] += 1
        g[1] += s["reward_std"] > 1e-9
    print("  spread by gold label:", {g: f"{b}/{t}" for g, (t, b) in by_gold.items()})
    print(f"wrote {path}")
