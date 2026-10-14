"""Train a single-turn GRPO verifier; shared implementation for experiment runs."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import reward as R
from utils.grpo_common import (DEFAULT_SFT, DEFAULT_SFT_DATA, VERDICT, FORMAT_FAIL,
                         load_rows, make_reward_funcs, filter_spread)
from stage0 import LOCAL_MODEL, SEED
from utils.experiment_layout import artifacts_dir

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
    log_dir = os.path.join(os.path.dirname(out), "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "train_log.json"), "w", encoding="utf-8") as f:
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
    ap.add_argument("--run-name", default=None, help="repeat label under experiments/<configuration>/runs/")
    a = ap.parse_args()
    a.data = a.data or DEFAULT_SFT_DATA

    name = f"{a.model.replace('/', '_')}_{'base_' if a.init == 'base' else ''}{a.reward}_n{a.n_train}" + (f"_{a.tag}" if a.tag else "")
    artifacts = artifacts_dir("grpo", name, a.seed, a.run_name)
    if artifacts.exists():
        raise SystemExit(f"run already exists: {artifacts}; use --run-name for a new run")
    out = artifacts / "checkpoints"
    out.mkdir(parents=True)
    secs, peak_gb, steps = train(a, str(out))
    meta = {**vars(a), "steps": steps, "train_seconds": round(secs), "peak_gpu_gb": round(peak_gb, 2),
            "harm": {f"{g}->{p}": w for (g, p), w in R.HARM.items()},
            "nei_cited_penalty": R.NEI_CITED_PENALTY}
    with open(artifacts / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"\nadapter saved to {out}\nnext: python stage0.py --model local --adapter {out} --n 900 --retrieval titles")
