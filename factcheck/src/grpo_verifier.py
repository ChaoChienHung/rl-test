"""Stage 1b: GRPO on the verifier, starting from the SFT probe's adapter.

Single-turn: claim + evidence in, the stage-0 JSON out, scored by reward.py.
Same prompts as the SFT probe (FEVER train, gold pages in context, from its
train.jsonl), same system prompt and chat template as stage0, so the adapter
evaluates with stage0.py --adapter on the same 900 dev claims as every other row.

    python grpo_verifier.py --max-steps 3 --n-train 16          # smoke test
    python grpo_verifier.py                                     # full reward
    python grpo_verifier.py --reward no_calibration             # ablations: see reward.ABLATIONS
    python stage0.py --model local --adapter ../data/grpo/<run> --n 900 --retrieval titles

With --beta 0 (the first run) the LoRA being trained is the SFT adapter
itself and no reference model is held. With --beta > 0 the SFT adapter is
merged into the weights and a fresh LoRA is trained on top, so TRL's
reference (the model with the adapter off) is the SFT model, not the base.
That run sharpened the verdict probabilities until all 8 samples agreed;
the KL term is the fix (see disagreement.py).

    python grpo_verifier.py --beta 0.04 --filter --max-steps 200 --tag kl
"""

import argparse
import json
import os
import random
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import reward as R
from build_bm25_index import DATA
from stage0 import LOCAL_MODEL, MAX_CITED, SEED, SYSTEM, parse

SFT_DIR = os.path.join(DATA, "..", "sft_probe")
OUT_DIR = os.path.join(DATA, "..", "grpo")
DEFAULT_SFT = os.path.join(SFT_DIR, "Qwen_Qwen3.5-2B_n6000")
VERDICT = {"SUPPORTED": R.SUPPORTED, "REFUTED": R.REFUTED, "NOT_ENOUGH_INFO": R.NEI}
FORMAT_FAIL = -2.0  # below the worst well-formed answer (about -1.5), so breaking format never pays


def load_rows(sft_dir, n, seed):
    """Prompts and gold labels/evidence ids from the SFT probe's train.jsonl."""
    with open(os.path.join(sft_dir, "train.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        target = json.loads(r["completion"])
        out.append({
            "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["prompt"]}],
            "gold": VERDICT[r["gold"]],
            "gold_ids": target["evidence"],
        })
    return out


def score_one(text, gold, gold_ids, cfg, rng):
    """(reward, parts) for one completion; parts is None on a format failure."""
    obj, ok = parse(text)
    if not all(ok.values()):
        if cfg.mode == "format_only":
            return 0.0, None
        if cfg.mode == "random":
            return rng.random(), None
        return FORMAT_FAIL, None
    cited = obj["evidence"][:MAX_CITED]
    return R.total(gold, VERDICT[obj["verdict"]], float(obj["confidence"]), gold_ids, cited,
                   n_ops=0, format_ok=True, cfg=cfg, rng=rng)


def make_reward_funcs(cfg, seed):
    """The training reward, plus zero-weight copies of its parts so TRL logs them."""
    rng = random.Random(seed)
    cache = {}

    def scored(completions, gold, gold_ids):
        key = id(completions)
        if key not in cache:
            cache.clear()
            cache[key] = [score_one(c[-1]["content"], g, ids, cfg, rng)
                          for c, g, ids in zip(completions, gold, gold_ids)]
        return cache[key]

    def verifier_reward(completions, gold, gold_ids, **_):
        return [r for r, _ in scored(completions, gold, gold_ids)]

    def part(name):
        def f(completions, gold, gold_ids, **_):
            return [p.get(name, 0.0) if p else 0.0 for _, p in scored(completions, gold, gold_ids)]
        f.__name__ = name
        return f

    def format_ok(completions, gold, gold_ids, **_):
        return [0.0 if p is None else 1.0 for _, p in scored(completions, gold, gold_ids)]

    funcs = [verifier_reward, part("correctness"), part("calibration"), part("evidence"), format_ok]
    return funcs, [1.0, 0.0, 0.0, 0.0, 0.0]


def spread_path(data, n):
    """disagreement.py's output for this prompt set (named after the data dir)."""
    return os.path.join(OUT_DIR, f"disagreement_{os.path.basename(os.path.normpath(data))}_n{n}.jsonl")


def filter_spread(rows, data, n):
    """Keep claims whose 8 SFT samples scored differently (disagreement.py)."""
    path = spread_path(data, n)
    with open(path, encoding="utf-8") as f:
        spread = {r["i"] for r in map(json.loads, f) if r["reward_std"] > 1e-9}
    return [r for i, r in enumerate(rows) if i in spread]


def train(a, out):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    cfg_r = R.ABLATIONS[a.reward]
    # The claim set is fixed at SEED (it must match disagreement.py's order for
    # --filter); --seed changes only sampling and batch order, so seeds compare.
    rows = load_rows(a.data, a.n_train, SEED)
    if a.filter:
        rows = filter_spread(rows, a.data, a.n_train)
    ds = Dataset.from_list(rows)
    print(f"{len(ds)} prompts, reward={a.reward}")

    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    peft_cfg = None
    if a.init == "base":  # RL vs SFT from the same start: fresh LoRA on the base model
        model = base
        with open(os.path.join(a.sft, "adapter_config.json"), encoding="utf-8") as f:
            sc = json.load(f)  # same LoRA shape as SFT, so the two are matched
        peft_cfg = LoraConfig(r=sc["r"], lora_alpha=sc["lora_alpha"], lora_dropout=sc["lora_dropout"],
                              target_modules=sc["target_modules"], task_type="CAUSAL_LM")
    elif a.beta > 0:
        model = PeftModel.from_pretrained(base, a.sft).merge_and_unload()
        with open(os.path.join(a.sft, "adapter_config.json"), encoding="utf-8") as f:
            sc = json.load(f)
        peft_cfg = LoraConfig(r=sc["r"], lora_alpha=sc["lora_alpha"], lora_dropout=sc["lora_dropout"],
                              target_modules=sc["target_modules"], task_type="CAUSAL_LM")
    else:
        model = PeftModel.from_pretrained(base, a.sft, is_trainable=True)
        model.print_trainable_parameters()

    funcs, weights = make_reward_funcs(cfg_r, a.seed)
    cfg = GRPOConfig(
        output_dir=out, seed=a.seed,
        num_generations=a.num_generations,
        per_device_train_batch_size=a.num_generations,  # one prompt's group per micro-batch
        gradient_accumulation_steps=a.prompts_per_step,
        max_completion_length=96, temperature=1.0, beta=a.beta,
        chat_template_kwargs={"enable_thinking": False},
        reward_weights=weights,
        learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=10,
        num_train_epochs=1, max_steps=a.max_steps if a.max_steps else -1,
        bf16=True, gradient_checkpointing=True,
        logging_steps=1, log_completions=False,
        save_strategy="steps", save_steps=50, save_total_limit=4,
        report_to="none",
    )
    trainer = GRPOTrainer(model=model, args=cfg, train_dataset=ds,
                          reward_funcs=funcs, processing_class=tok, peft_config=peft_cfg)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trainer.save_model(out)
    tok.save_pretrained(out)
    with open(os.path.join(out, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    return secs, torch.cuda.max_memory_allocated() / 2**30, trainer.state.global_step


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LOCAL_MODEL)
    ap.add_argument("--sft", default=DEFAULT_SFT, help="SFT probe adapter dir to start from")
    ap.add_argument("--data", default=None,
                    help="dir with train.jsonl prompts; default the SFT probe's (gold in context), "
                         "or build_grpo_data.py's retrieved-evidence set")
    ap.add_argument("--reward", default="full", choices=sorted(R.ABLATIONS))
    ap.add_argument("--n-train", type=int, default=1200, help="prompts, one pass")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--init", default="sft", choices=["sft", "base"],
                    help="start from the SFT adapter (stage 1b) or the base model (GRPO vs SFT, same start)")
    ap.add_argument("--beta", type=float, default=0.0, help="KL to the SFT model; > 0 trains a fresh LoRA on merged SFT")
    ap.add_argument("--filter", action="store_true", help="only claims with reward spread (disagreement.py)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    a.data = a.data or a.sft

    name = f"{a.model.replace('/', '_')}_{'base_' if a.init == 'base' else ''}{a.reward}_n{a.n_train}" + (f"_{a.tag}" if a.tag else "")
    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    secs, peak_gb, steps = train(a, out)
    meta = {**vars(a), "steps": steps, "train_seconds": round(secs), "peak_gpu_gb": round(peak_gb, 2),
            "harm": {f"{g}->{p}": w for (g, p), w in R.HARM.items()},
            "nei_cited_penalty": R.NEI_CITED_PENALTY}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"\nadapter saved to {out}\nnext: python stage0.py --model local --adapter {out} --n 900 --retrieval titles")
