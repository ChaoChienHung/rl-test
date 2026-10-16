#!/usr/bin/env python3
"""
Generate Fast-Slow Dual-Process datasets for SFT and GRPO.
Format:
  <guess>VERDICT</guess>
  <think>REASONING</think>
  <evidence>EID...</evidence>
  <verdict>VERDICT</verdict>
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TAGGED_COT_4500 = REPO_ROOT / "verifier/data/train/sft/fever-oracle-tagged-cot-4500/labeled_prompts.jsonl"
TAGGED_GRPO_1000 = REPO_ROOT / "verifier/data/train/sft/fever-oracle-tagged-grpo-1000/labeled_prompts.jsonl"

OUT_SFT_DIR = REPO_ROOT / "verifier/data/train/sft/fever-oracle-fastslow-4500"
OUT_GRPO_DIR = REPO_ROOT / "verifier/data/train/grpo/fever-oracle-fastslow-grpo-1000"


def transform_sft_row(row: dict) -> dict:
    """
    Transforms tagged CoT completion:
      <think>...</think>\n<evidence>...</evidence>\n<verdict>...</verdict>
    into fast-slow:
      <guess>{gold}</guess>\n<think>...</think>\n<evidence>...</evidence>\n<verdict>{gold}</verdict>
    """
    gold = row["gold"]
    raw_prompt = row["prompt"]
    completion = row["completion"]
    
    # Fast-Slow prompt instructs the model to emit guess first, then think, evidence, verdict
    base_prompt = raw_prompt.split("Respond in the following format")[0].strip()
    if "<think>" in base_prompt:
        base_prompt = base_prompt.split("<think>")[0].strip()
        
    fastslow_prompt = (
        f"{base_prompt}\n\n"
        "Respond in the following format:\n"
        "<guess>SUPPORTED | REFUTED | NOT_ENOUGH_INFO</guess>\n"
        "<think>\n"
        "Your step-by-step reasoning...\n"
        "</think>\n"
        "<evidence>E1, E2</evidence> (or <evidence></evidence> if insufficient)\n"
        "<verdict>SUPPORTED | REFUTED | NOT_ENOUGH_INFO</verdict>\n\n"
        "<guess>"
    )
    
    # Strip leading <think> if present in completion
    comp = completion.strip()
    if comp.startswith("<think>"):
        comp = comp[len("<think>"):].strip()
        
    # Since prompt ends with <guess>, completion starts with {gold}</guess>\n<think>\n...
    # Or prompt ends with prompt text and completion starts with <guess>
    # In our repository convention: prompt ends with opening tag or newline.
    # If prompt ends with <guess>, completion is:
    fastslow_completion = f"{gold}</guess>\n<think>\n{comp}"
    
    return {
        "id": row["id"],
        "gold": gold,
        "prompt": fastslow_prompt,
        "completion": fastslow_completion
    }


def transform_grpo_row(row: dict) -> dict:
    gold = row["gold"]
    raw_prompt = row["prompt"]
    
    base_prompt = raw_prompt.split("Respond in the following format")[0].strip()
    if "<think>" in base_prompt:
        base_prompt = base_prompt.split("<think>")[0].strip()
        
    fastslow_prompt = (
        f"{base_prompt}\n\n"
        "Respond in the following format:\n"
        "<guess>SUPPORTED | REFUTED | NOT_ENOUGH_INFO</guess>\n"
        "<think>\n"
        "Your step-by-step reasoning...\n"
        "</think>\n"
        "<evidence>E1, E2</evidence> (or <evidence></evidence> if insufficient)\n"
        "<verdict>SUPPORTED | REFUTED | NOT_ENOUGH_INFO</verdict>\n\n"
        "<guess>"
    )
    
    return {
        "id": row["id"],
        "gold": gold,
        "prompt": fastslow_prompt,
        "completion": f"{gold}</guess>"
    }


def main():
    OUT_SFT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_GRPO_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading SFT sources from {TAGGED_COT_4500}...")
    sft_rows = []
    with open(TAGGED_COT_4500, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sft_rows.append(transform_sft_row(json.loads(line)))
                
    sft_out = OUT_SFT_DIR / "labeled_prompts.jsonl"
    with open(sft_out, "w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Write manifest
    sft_manifest = {
        "name": "fever-oracle-fastslow-4500",
        "n_samples": len(sft_rows),
        "source": str(TAGGED_COT_4500),
        "format": "fastslow_dual_process"
    }
    with open(OUT_SFT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(sft_manifest, f, indent=2)
    print(f"Wrote {len(sft_rows)} SFT rows to {sft_out}")

    print(f"Reading GRPO sources from {TAGGED_GRPO_1000}...")
    grpo_rows = []
    with open(TAGGED_GRPO_1000, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                grpo_rows.append(transform_grpo_row(json.loads(line)))
                
    grpo_out = OUT_GRPO_DIR / "labeled_prompts.jsonl"
    with open(grpo_out, "w", encoding="utf-8") as f:
        for r in grpo_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    grpo_manifest = {
        "name": "fever-oracle-fastslow-grpo-1000",
        "n_samples": len(grpo_rows),
        "source": str(TAGGED_GRPO_1000),
        "format": "fastslow_dual_process"
    }
    with open(OUT_GRPO_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(grpo_manifest, f, indent=2)
    print(f"Wrote {len(grpo_rows)} GRPO rows to {grpo_out}")


if __name__ == "__main__":
    main()
