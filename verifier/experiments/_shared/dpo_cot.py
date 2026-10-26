"""DPO on genuine CoT pairs from the fixed SFT4500-disjoint 1,000-claim pool."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.experiment_layout import artifacts_dir
from verifier.model import LOCAL_MODEL, NATIVE_THINKING_SYSTEM


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--sft", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True, help="pairs.jsonl from build_dpo_pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-root")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 trains for one epoch; positive values override")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--max-prompt-length", type=int, default=3072)
    ap.add_argument("--pool-claims", type=int, choices=[500, 1000], default=500)
    ap.add_argument("--report", type=Path, help="write completed run location for a pipeline")
    args = ap.parse_args()
    if args.max_steps < 0 or args.max_prompt_length < 1 or args.max_length <= args.max_prompt_length:
        ap.error("invalid step or token limits")
    if not (args.sft / "adapter_config.json").is_file() or not args.pairs.is_file():
        ap.error("missing SFT adapter or DPO pair file")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    data = []
    skipped = 0
    for line in args.pairs.read_text(encoding="utf-8").splitlines():
        pair = json.loads(line)
        if pair["system_prompt"] != NATIVE_THINKING_SYSTEM or pair["format"] != "json":
            skipped += 1
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": pair["system_prompt"]},
             {"role": "user", "content": pair["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        chosen, rejected = pair["chosen"], pair["rejected"]
        plen = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if plen > args.max_prompt_length or any(
            plen + len(tokenizer(x, add_special_tokens=False).input_ids) + 1 > args.max_length
            for x in (chosen, rejected)
        ):
            skipped += 1
            continue
        data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if not data:
        raise SystemExit("no valid pairs fit format and token limits; no run created")
    print(f"DPO pairs: {len(data)} kept, {skipped} skipped", flush=True)
    config_name = f"dpo-cot-from-sft4500-qwen3p5-2b-pool{args.pool_claims}-{len(data)}-samples"
    artifacts = artifacts_dir("dpo", config_name, args.seed, args.run_name,
                              args.output_root, config_name=config_name)
    if artifacts.exists():
        raise SystemExit(f"run already exists: {artifacts}")

    # Merge the existing SFT adapter first. With DPO's PEFT reference adapter
    # disabled, the reference is then exactly the frozen SFT checkpoint.
    base = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, dtype=torch.bfloat16, device_map="cuda")
    sft = PeftModel.from_pretrained(base, str(args.sft))
    merged = sft.merge_and_unload()
    config = LoraConfig.from_pretrained(str(args.sft))
    config.inference_mode = False
    model = get_peft_model(merged, config)
    model.print_trainable_parameters()
    out = artifacts / "checkpoints"
    out.mkdir(parents=True)
    config_args = dict(
        output_dir=str(out), seed=args.seed, beta=0.1,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=1, max_steps=args.max_steps or -1, learning_rate=1e-5,
        max_length=args.max_length,
        truncation_mode="keep_start", bf16=True, gradient_checkpointing=True,
        logging_steps=1, save_strategy="no", report_to="none",
    )
    if "max_prompt_length" in DPOConfig.__dataclass_fields__:
        config_args["max_prompt_length"] = args.max_prompt_length
    cfg = DPOConfig(**config_args)
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg,
                         train_dataset=Dataset.from_list(data),
                         processing_class=tokenizer)
    started = time.time()
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    logs = artifacts / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1) + "\n")
    meta = {
        "experiment": config_name, "model": LOCAL_MODEL,
        "sft": str(args.sft), "pairs": str(args.pairs),
        "pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "n_pairs": len(data), "skipped_pairs": skipped, "seed": args.seed,
        "pool_claims": args.pool_claims,
        "steps": trainer.state.global_step, "max_steps_override": args.max_steps,
        "num_train_epochs": 1, "beta": 0.1,
        "reference": "SFT adapter merged into base; DPO adapter disabled",
        "max_length": args.max_length, "max_prompt_length": args.max_prompt_length,
        "train_seconds": round(time.time() - started),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (artifacts / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if args.report:
        args.report.write_text(json.dumps({"config": config_name, "artifacts": str(artifacts)}) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
