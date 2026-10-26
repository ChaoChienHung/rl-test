"""Supervise only RSI-Jev's option scorer on high-margin citation decisions."""

import argparse
import json
import math
import random
import subprocess
from collections import Counter
from pathlib import Path

from agent.evaluation.pooled_retry import FINAL_QUESTION
from agent.evaluation.results.rsi_jev_evidence_selection_baseline.eval_rsi_jev_evidence_selection_baseline import ACTION_KEEP_ALL
from agent.train.jev_head import JevHeadPolicy, choice_question

QUESTIONS = {"chunk": ACTION_KEEP_ALL, "pooled": FINAL_QUESTION["action"]}
LABELS = ("STOP", "RETRY")


def load_examples(data_dir, split_manifest):
    data = Path(data_dir)
    manifest = json.loads((data / "manifest.json").read_text())
    split = json.loads(Path(split_manifest).read_text())
    if manifest["split_manifest"] != str(split_manifest) or manifest["n_claims"] != 3000:
        raise ValueError("SFT data does not match the fixed 3,000-claim split")
    expected = set(split["sft_ids"])
    episodes = [json.loads(line) for line in (data / "episodes.jsonl").open(encoding="utf-8")
                if line.strip()]
    if len(episodes) != 3000 or {e["claim_id"] for e in episodes} != expected:
        raise ValueError("SFT episodes are incomplete or contain wrong IDs")
    examples = [example for episode in episodes for example in episode["examples"]]
    for example in examples:
        if example["claim_id"] not in expected or example["stage"] not in QUESTIONS:
            raise ValueError("invalid SFT example stage or claim ID")
        if example["label"] not in LABELS or "gold" in json.loads(example["state"]):
            raise ValueError("invalid SFT label or gold leaked into policy state")
    counts = Counter((e["stage"], e["label"]) for e in examples)
    if any(counts[stage, label] == 0 for stage in QUESTIONS for label in LABELS):
        raise ValueError("SFT data must include STOP and RETRY for both stages")
    return manifest, examples, counts


def balance_weights(counts):
    result = {}
    for stage in QUESTIONS:
        inverse = {label: 1 / math.sqrt(counts[stage, label]) for label in LABELS}
        mean = sum(inverse.values()) / 2
        result[stage] = {label: min(4.0, inverse[label] / mean) for label in LABELS}
    return result


def run(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Jev SFT requires a CUDA GPU allocation")
    data_manifest, examples, counts = load_examples(args.data_dir, args.split_manifest)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"training output exists: {output}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = JevHeadPolicy(args.jev_model, device="cuda")
    optimizer = torch.optim.AdamW(policy.model.scorer.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    weights = balance_weights(counts)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip() or None
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps({
        "training_code": __name__, "method": "high-margin outcome-derived SFT",
        "variant": "citation", "release": args.jev_model,
        "data_dir": args.data_dir, "data_manifest": data_manifest,
        "split_manifest": args.split_manifest, "n_claims": 3000,
        "n_examples": len(examples), "label_counts":
            {f"{stage}:{label}": count for (stage, label), count in counts.items()},
        "class_weights": weights, "epochs": args.epochs, "seed": args.seed,
        "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "updated_parameters": "RSI-Jev option scorer only; tower frozen",
        "commit": revision, "validation_claims_used": False}, indent=2) + "\n")
    steps = 0
    with (output / "train_metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for epoch in range(args.epochs):
            order = list(range(len(examples)))
            random.Random(args.seed + epoch).shuffle(order)
            loss_sum, correct = 0.0, 0
            for index in order:
                example = examples[index]
                question_spec = QUESTIONS[example["stage"]]
                question = choice_question("action", question_spec["instructions"],
                                           question_spec["criteria"])
                logits = policy.logits(example["state"], question)
                target = question.options.index(example["label"])
                class_weight = weights[example["stage"]][example["label"]]
                loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                    torch.tensor([target], device=logits.device)) * class_weight
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite Jev SFT loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.model.scorer.parameters(), 1.0)
                optimizer.step()
                steps += 1
                loss_sum += float(loss.detach())
                correct += int(int(logits.argmax()) == target)
            record = {"epoch": epoch + 1, "steps": steps,
                      "train_loss": loss_sum / len(examples),
                      "train_accuracy": correct / len(examples)}
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            extra = {"episode": steps, "release": args.jev_model,
                     "variant": "citation", "training_method": "sft",
                     "epoch": epoch + 1, "split_manifest": args.split_manifest}
            policy.save(output / "artifacts" / "checkpoints" /
                        f"epoch-{epoch + 1}" / "scorer.pt", extra)
            policy.save(output / "artifacts" / "checkpoints" / "scorer.pt", extra)
            print(json.dumps(record), flush=True)
    print(f"SFT completed: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jev-model", default="shgao/rsi-jev-v2.1-qwen3.5-2b")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error("epochs and lr must be positive")
    run(args)


if __name__ == "__main__":
    main()
