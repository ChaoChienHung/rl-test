"""Train a Qwen3.5-2B LoRA adapter on 5,500 Native Thinking CoT SFT data."""

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.experiment_layout import ROOT, artifacts_dir
from verifier.model import NATIVE_THINKING_SYSTEM

DEFAULT_DATA = ROOT / "verifier/data/train/sft/fever-oracle-cot-soclaas/cot_native_sft_labeled_prompts.jsonl"
MODEL = "Qwen/Qwen3.5-2B"


def load_cot_data(path):
    """Require completed, validated native thinking data and no dev-claim overlap."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"CoT SFT data must exist: {path}")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 5500:
        raise ValueError(f"Expected exactly 5,500 rows, got {len(rows)}")

    ids = [str(row["id"]) for row in rows]
    dev_ids = set(map(str, json.loads((ROOT / "verifier/data/eval/fever-dev-900-ids.json").read_text())["ids"]))
    if len(ids) != len(set(ids)) or set(ids) & dev_ids:
        raise ValueError("duplicate CoT train IDs or overlap with frozen FEVER dev IDs")

    for row in rows:
        comp = row["completion"]
        if "</think>" not in comp:
            raise ValueError(f"missing </think> in completion for claim {row['id']}")
        reasoning_part, json_part = comp.split("</think>", 1)
        if not reasoning_part.strip():
            raise ValueError(f"empty reasoning in completion for claim {row['id']}")
        m = re.search(r"\{.*\}", json_part, re.S)
        if not m:
            raise ValueError(f"missing JSON object after </think> for claim {row['id']}")
        target = json.loads(m.group(0))
        if target.get("verdict") != row["gold"] or not isinstance(target.get("evidence"), list) or "confidence" in target:
            raise ValueError(f"invalid JSON payload in completion for claim {row['id']}")

    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-root")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="cot_native_sft_labeled_prompts.jsonl")
    ap.add_argument("--max-steps", type=int, default=0, help="0 trains one epoch")
    args = ap.parse_args()

    rows, data_hash = load_cot_data(args.data)
    artifacts = artifacts_dir("cot_sft", "qwen_3.5-2b_oracle_5500", args.seed, args.run_name,
                              args.output_root, config_name=Path(__file__).parent.name)
    if artifacts.exists():
        raise SystemExit(f"run already exists: {artifacts}; choose a new run name")

    from verifier.experiments._shared.sft import train
    checkpoint_dir = artifacts / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    print(f"Native Thinking CoT SFT data={Path(args.data).resolve()} rows={len(rows)} sha256={data_hash}", flush=True)
    seconds, peak_gb = train(MODEL, rows, str(checkpoint_dir), False, args.max_steps,
                             1e-4, 16, args.seed, system_prompt=NATIVE_THINKING_SYSTEM,
                             enable_thinking=True)

    meta = {
        "experiment": Path(__file__).parent.name,
        "model": MODEL,
        "mode": "native_thinking",
        "data_file": str(Path(args.data).resolve()),
        "data_sha256": data_hash,
        "n_train": len(rows),
        "label_counts": dict(collections.Counter(row["gold"] for row in rows)),
        "prompt_style": "cot_native",
        "max_steps": args.max_steps,
        "lr": 1e-4,
        "rank": 16,
        "seed": args.seed,
        "train_seconds": round(seconds),
        "peak_gpu_gb": round(peak_gb, 2),
        "checkpoint_dir": str(checkpoint_dir),
    }
    (artifacts / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
