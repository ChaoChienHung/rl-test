"""Fast-Slow Dual-Process Native-Thinking GRPO with Cognitive Self-Correction Reward Matrix."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.experiment_layout import artifacts_dir
from core import reward as R
from verifier.model import LABELS, LOCAL_MODEL, MAX_CITED, TAG_NATIVE_THINKING_SYSTEM, SEED, parse

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "verifier/data/train/grpo/fever-oracle-fastslow-grpo-1000"


def completion_text(completion):
    return completion[-1]["content"] if isinstance(completion, list) else completion


def score_one_fastslow(completion, gold, gold_ids):
    """
    Cognitive Self-Correction Reward Matrix:
    - guess == gold and verdict == gold: +1.5 (Full Consensus)
    - guess != gold and verdict == gold: +1.2 (Cognitive Triumph: System 2 self-correction)
    - guess == gold and verdict != gold: -0.5 (Delusion: CoT derailed intuition)
    - guess != gold and verdict != gold: -1.0 (Dual failure)
    Plus evidence precision/recall and format validation.
    """
    raw_text = completion_text(completion)
    obj, ok = parse(raw_text)
    
    # Format penalty if verdict or evidence tags are missing
    if not (ok["verdict"] and ok["evidence"]):
        return -2.0, {"status": "format_fail", "self_corrected": 0.0, "deluded": 0.0, "correctness": 0.0, "evidence": 0.0}

    verdict = obj["verdict"]
    guess = obj.get("guess")
    evidence = obj.get("evidence", [])[:MAX_CITED]
    
    guess_correct = (guess == gold) if guess else False
    verdict_correct = (verdict == gold)
    
    self_corrected = (not guess_correct and verdict_correct)
    deluded = (guess_correct and not verdict_correct)
    
    # 1. Metacognitive verdict reward
    if guess_correct and verdict_correct:
        r_verdict = 1.5
    elif self_corrected:
        r_verdict = 1.2
    elif deluded:
        r_verdict = -0.5
    else:
        r_verdict = -1.0
        
    # Harm penalty for false SUPPORTED on Gold NEI
    if gold == "NOT_ENOUGH_INFO" and verdict == "SUPPORTED":
        r_verdict -= 0.5

    # 2. Evidence reward
    if gold == "NOT_ENOUGH_INFO":
        r_evidence = 0.5 if len(evidence) == 0 else -0.5
    else:
        # Check against gold_ids
        pred_set = set(evidence)
        gold_set = set(gold_ids)
        if not gold_set:
            r_evidence = 0.0
        else:
            prec = len(pred_set & gold_set) / len(pred_set) if pred_set else 0.0
            rec = len(pred_set & gold_set) / len(gold_set)
            r_evidence = 0.5 * prec + 0.5 * rec

    total_reward = r_verdict + r_evidence
    parts = {
        "status": "ok",
        "self_corrected": float(self_corrected),
        "deluded": float(deluded),
        "correctness": float(verdict_correct),
        "evidence": float(r_evidence),
        "r_verdict": float(r_verdict),
    }
    return total_reward, parts


def make_fastslow_reward_funcs():
    cache = {}

    def scored(completions, gold, gold_ids):
        key = id(completions)
        if key not in cache:
            cache.clear()
            cache[key] = [score_one_fastslow(c, label, ids)
                          for c, label, ids in zip(completions, gold, gold_ids)]
        return cache[key]

    def fastslow_reward(completions, gold, gold_ids, **_):
        return [reward for reward, _ in scored(completions, gold, gold_ids)]

    def self_correction_rate(completions, gold, gold_ids, **_):
        return [parts["self_corrected"] if parts else 0.0 for _, parts in scored(completions, gold, gold_ids)]

    def delusion_rate(completions, gold, gold_ids, **_):
        return [parts["deluded"] if parts else 0.0 for _, parts in scored(completions, gold, gold_ids)]

    def correctness(completions, gold, gold_ids, **_):
        return [parts["correctness"] if parts else 0.0 for _, parts in scored(completions, gold, gold_ids)]

    def evidence(completions, gold, gold_ids, **_):
        return [parts["evidence"] if parts else 0.0 for _, parts in scored(completions, gold, gold_ids)]

    return [fastslow_reward, self_correction_rate, delusion_rate, correctness, evidence]


def load_train_rows(data_dir, n_train):
    path = Path(data_dir) / "labeled_prompts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    dev_ids = set(map(str, json.loads((ROOT / "verifier/data/eval/fever-dev-900-ids.json").read_text())["ids"]))
    if len(rows) not in (1000, 1200) or len(set(ids)) != len(rows) or set(ids) & dev_ids:
        raise ValueError(f"expected 1,000 or 1,200 unique train claims with no frozen dev overlap, got {len(rows)}")
    if n_train < 1 or n_train > len(rows) or any(row["gold"] not in LABELS for row in rows):
        raise ValueError("invalid n_train or gold label")
    random.Random(SEED).shuffle(rows)
    out = []
    for row in rows[:n_train]:
        obj, _ = parse(row.get("completion", ""))
        gold_ids = obj.get("evidence", []) if isinstance(obj, dict) else []
        out.append({
            "prompt": [{"role": "system", "content": TAG_NATIVE_THINKING_SYSTEM},
                       {"role": "user", "content": row["prompt"]}],
            "gold": row["gold"],
            "gold_ids": gold_ids,
        })
    return out


def main(config_name, n_train=1000, default_steps=0):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sft", required=True, help="completed Fast-Slow SFT adapter")
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root")
    parser.add_argument("--max-steps", type=int, default=default_steps)
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be nonnegative")
    checkpoint = Path(args.sft)
    if not (checkpoint / "adapter_config.json").is_file() or not any(
        (checkpoint / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        parser.error(f"incomplete SFT adapter: {checkpoint}")
    rows = load_train_rows(args.data, n_train)
    artifacts = artifacts_dir("grpo", config_name, args.seed, args.run_name,
                              args.output_root, config_name=config_name)
    if artifacts.exists():
        raise SystemExit(f"run already exists: {artifacts}; choose a new run name")

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    base = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=True)
    model.print_trainable_parameters()
    out = artifacts / "checkpoints"
    out.mkdir(parents=True)
    config = GRPOConfig(
        output_dir=str(out), seed=args.seed, num_generations=8,
        per_device_train_batch_size=8, gradient_accumulation_steps=2,
        max_completion_length=1024, temperature=1.0, beta=0.0,
        chat_template_kwargs={"enable_thinking": True},
        reward_weights=[1.0, 0.0, 0.0, 0.0, 0.0],
        learning_rate=2e-5, lr_scheduler_type="constant_with_warmup", warmup_steps=10,
        num_train_epochs=1, max_steps=args.max_steps or -1,
        bf16=True, gradient_checkpointing=True,
        logging_steps=1, log_completions=False,
        save_strategy="steps", save_steps=50, save_total_limit=2,
        report_to="none",
    )
    trainer = GRPOTrainer(model=model, args=config, train_dataset=Dataset.from_list(rows),
                          reward_funcs=make_fastslow_reward_funcs(),
                          processing_class=tokenizer)
    started = time.time()
    trainer.train()
    seconds = time.time() - started
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    logs = artifacts / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1) + "\n")
    meta = {
        "experiment": config_name, "model": LOCAL_MODEL, "sft": str(checkpoint),
        "data": str(Path(args.data)), "n_train": n_train, "seed": args.seed,
        "reward": "fastslow_metacognitive_self_correction_reward", "thinking_mode": True,
        "diagnostics": ["self_correction_rate", "delusion_rate", "correctness", "evidence"],
        "num_generations": 8, "max_completion_length": 1024, "beta": 0.0,
        "max_steps": args.max_steps, "steps": trainer.state.global_step,
        "train_seconds": round(seconds),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (artifacts / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
