"""Sample genuine native-thinking CoT completions on the fixed train-only set.

Run on a SoC GPU node. The output is local, resumable, and ignored by Git.
"""

import argparse
import json
import random
from pathlib import Path

from verifier.model import LOCAL_MODEL, NATIVE_THINKING_SYSTEM

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "verifier/data/train/sft/fever-oracle-matched-grpo-1000/labeled_prompts.jsonl"
DEV = ROOT / "verifier/data/eval/fever-dev-900-ids.json"


def select_rows(source, tokenizer, n_claims, seed, max_prompt_tokens):
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    dev_ids = set(map(str, json.loads(DEV.read_text(encoding="utf-8"))["ids"]))
    if len(ids) != len(set(ids)) or set(ids) & dev_ids or n_claims > len(rows):
        raise ValueError("source must contain unique train-only IDs and enough claims")
    random.Random(seed).shuffle(rows)
    selected = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": NATIVE_THINKING_SYSTEM},
             {"role": "user", "content": row["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        if max_prompt_tokens == 0 or len(tokenizer(prompt, add_special_tokens=False).input_ids) <= max_prompt_tokens:
            selected.append(row)
            if len(selected) == n_claims:
                break
    if len(selected) != n_claims:
        raise ValueError(f"only {len(selected)} claims fit the prompt budget")
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="new or resumable local directory")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--n-claims", type=int, default=1000)
    ap.add_argument("--rollouts", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-prompt-tokens", type=int, default=0,
                    help="0 keeps the full fixed pool; DPO training later skips overlong pairs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if min(args.n_claims, args.rollouts, args.max_new_tokens) < 1 or args.max_prompt_tokens < 0:
        ap.error("counts must be positive and max-prompt-tokens nonnegative")
    if not (args.checkpoint / "adapter_config.json").is_file():
        ap.error("SFT checkpoint is missing adapter_config.json")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    rows = select_rows(args.source, tokenizer, args.n_claims, args.seed, args.max_prompt_tokens)
    manifest = {
        "model": LOCAL_MODEL, "checkpoint": str(args.checkpoint), "source": str(args.source),
        "claim_ids": [str(r["id"]) for r in rows], "n_claims": args.n_claims,
        "rollouts": args.rollouts, "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens, "seed": args.seed,
        "system_prompt": NATIVE_THINKING_SYSTEM, "enable_thinking": True,
        "temperature": 1.0, "top_p": 0.9,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("output manifest differs; choose a new output directory")
    else:
        if any(args.output.iterdir()):
            raise SystemExit("output exists without manifest; choose a clean directory")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output = args.output / "candidates.jsonl"
    done = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            key = (str(item["id"]), item["rollout"])
            if key in done:
                raise ValueError(f"duplicate rollout: {key}")
            done.add(key)

    base = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, str(args.checkpoint)).eval()
    with output.open("a", encoding="utf-8") as stream:
        for i, row in enumerate(rows, 1):
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": NATIVE_THINKING_SYSTEM},
                 {"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            for rollout in range(args.rollouts):
                if (str(row["id"]), rollout) in done:
                    continue
                torch.manual_seed(args.seed + i * 1000 + rollout)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, do_sample=True, temperature=1.0, top_p=0.9,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
                ended = tokenizer.eos_token_id in ids
                if ended:
                    ids = ids[:ids.index(tokenizer.eos_token_id)]
                completion = tokenizer.decode(ids, skip_special_tokens=False)
                stream.write(json.dumps({
                    "id": row["id"], "prompt": row["prompt"],
                    "system_prompt": NATIVE_THINKING_SYSTEM, "completion": completion,
                    "model": LOCAL_MODEL, "run": args.output.name, "rollout": rollout,
                    "finish_reason": "eos" if ended else "length", "out_tokens": len(ids),
                }, ensure_ascii=False) + "\n")
                stream.flush()
            print(f"{i}/{len(rows)} claims sampled", flush=True)


if __name__ == "__main__":
    main()
