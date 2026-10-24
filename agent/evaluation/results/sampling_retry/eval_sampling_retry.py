"""Three stochastic verifier draws per FEVER claim, then an offline retry policy.

Every draw sees the same claim and retrieved evidence. All three are generated
even when the simulated policy would stop early, so disagreement can be measured
without changing the policy's simulated cost.
"""

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from verifier.model import (
    COT_BRIEF_SYSTEM, LOCAL_MODEL, NATIVE_THINKING_SYSTEM, SEED, SYSTEM,
    TAG_DIRECT_SYSTEM, TAG_NATIVE_THINKING_SYSTEM, LocalModel, build_context,
    render, sample_claims, score,
)


PROMPTS = {
    "direct": (SYSTEM, False, 160),
    "direct_tag": (TAG_DIRECT_SYSTEM, False, 160),
    "cot_brief": (COT_BRIEF_SYSTEM, False, 512),
    "cot_native": (NATIVE_THINKING_SYSTEM, True, 1024),
    "cot_native_tag": (TAG_NATIVE_THINKING_SYSTEM, True, 1024),
}
NEI = "NOT_ENOUGH_INFO"
LABELS = ("SUPPORTED", "REFUTED", NEI)
N_DRAWS = 3


def stopping_index(samples, threshold):
    """Select the first valid, non-NEI draw above threshold, else draw 3."""
    if len(samples) != N_DRAWS:
        raise ValueError(f"expected exactly {N_DRAWS} samples")
    for index, sample in enumerate(samples):
        confidence = sample.get("token_conf")
        if (sample.get("format_ok") and sample.get("pred") in LABELS[:2]
                and isinstance(confidence, (int, float))
                and math.isfinite(confidence) and confidence >= threshold):
            return index
    return N_DRAWS - 1


def summarize(rows):
    def mean(values):
        return sum(values) / len(values) if values else None

    def quality(samples):
        return {
            "accuracy": mean([int(s["correct"]) for s in samples]),
            "fever_score": mean([int(s["fever_score"]) for s in samples]),
            "evidence_f1_non_nei": mean([s["evidence_f1"] for s in samples
                                         if s["evidence_f1"] is not None]),
            "format_rate": mean([int(s["format_ok"]) for s in samples]),
            "recall": {label: mean([int(s["pred"] == label) for s in samples
                                    if s["gold"] == label]) for label in LABELS},
        }

    def comparison(part):
        return {
            "n": len(part),
            "first_draw": quality([row["samples"][0] for row in part]),
            "all_draws": quality([sample for row in part for sample in row["samples"]]),
            "heuristic": quality([row["samples"][row["heuristic"]["selected_index"]]
                                  for row in part]),
            "third_draw": quality([row["samples"][2] for row in part]),
        }

    return {
        "n_claims": len(rows),
        "draws_per_claim": N_DRAWS,
        "overall": comparison(rows),
        "gold_in_context": comparison([r for r in rows if r["gold_in_context"] is True]),
        "gold_not_in_context": comparison([r for r in rows if r["gold_in_context"] is False]),
        "gold_nei": comparison([r for r in rows if r["gold"] == NEI]),
        "stop_at": {str(i): sum(row["heuristic"]["n_verify"] == i for row in rows)
                    for i in range(1, N_DRAWS + 1)},
        "mean_simulated_verifier_calls": mean([row["heuristic"]["n_verify"] for row in rows]),
        "actual_verifier_calls": len(rows) * N_DRAWS,
        "verdict_disagreement_rate": mean([
            int(len({s["pred"] for s in row["samples"]}) > 1) for row in rows]),
        "evidence_disagreement_rate": mean([
            int(len({tuple(s["evidence"]) for s in row["samples"]}) > 1) for row in rows]),
        "mean_token_conf_std": mean([
            row["token_conf_std"] for row in rows if row["token_conf_std"] is not None]),
        "early_wrong_with_later_correct": sum(
            row["heuristic"]["n_verify"] < N_DRAWS
            and not row["samples"][row["heuristic"]["selected_index"]]["correct"]
            and any(s["correct"] for s in row["samples"][row["heuristic"]["n_verify"]:])
            for row in rows),
        "actual_tokens_in": sum(s["in_tokens"] for row in rows for s in row["samples"]),
        "actual_tokens_out": sum(s["out_tokens"] for row in rows for s in row["samples"]),
        "simulated_tokens_in": sum(s["in_tokens"] for row in rows
                                   for s in row["samples"][:row["heuristic"]["n_verify"]]),
        "simulated_tokens_out": sum(s["out_tokens"] for row in rows
                                    for s in row["samples"][:row["heuristic"]["n_verify"]]),
    }


def run(args):
    from retrieval.fever_retrieval import FeverRetriever

    if not 0 < args.threshold <= 1 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("threshold and top_p must be in (0, 1]; temperature must be positive")
    if args.n <= 0 or args.max_new_tokens <= 0:
        raise ValueError("n and max-new-tokens must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"result directory exists: {output_dir}; use a new run name")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip() or None
    manifest = {
        "evaluation_code": "agent.evaluation.results.sampling_retry.eval_sampling_retry",
        "model": args.model, "adapter": args.adapter, "prompt_style": args.prompt_style,
        "retrieval": "titles", "n_claims": args.n, "claim_seed": SEED,
        "sampling_seed": args.seed, "draws_per_claim": N_DRAWS,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens, "threshold": args.threshold,
        "confidence_for_gate": "token_conf", "commit": commit,
        "note": "All three draws use identical evidence; stopping is simulated after generation.",
    }
    claims = sample_claims(args.n, SEED)
    if args.n == 900:
        frozen = json.loads(Path("verifier/data/eval/fever-dev-900-ids.json").read_text())
        if [claim["id"] for claim in claims] != frozen["ids"]:
            raise ValueError("sampled claims differ from the frozen 900-ID eval manifest")
    retriever = FeverRetriever()
    system, thinking, _ = PROMPTS[args.prompt_style]
    model = LocalModel(args.model, batch_size=N_DRAWS, adapter=args.adapter,
                       system_prompt=system, max_new_tokens=args.max_new_tokens,
                       enable_thinking=thinking, do_sample=True,
                       temperature=args.temperature, top_p=args.top_p)
    output_dir.mkdir(parents=True)
    predictions = output_dir / "predictions.jsonl"
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    started = time.monotonic()
    with predictions.open("w", encoding="utf-8") as stream:
        for index, claim in enumerate(claims, 1):
            ctx = build_context(retriever, [claim], retrieval="titles")[0]
            prompt = render(claim["claim"], ctx["shown"])
            draws = model.run([prompt] * N_DRAWS, seed=args.seed + int(claim["id"]))
            samples = []
            for draw_index, generation in enumerate(draws, 1):
                item = score(claim, ctx, generation, require_confidence=False)
                # score() retains the parser's optional guess flag. It is not
                # part of the final-answer format checked by this policy.
                fmt = item["format"]
                item["format_ok"] = all(fmt.get(key) for key in ("json", "verdict", "evidence"))
                item["evidence"] = [f"{page}:{line}" for page, line in item["cited"]]
                item["draw"] = draw_index
                samples.append(item)
            selected_index = stopping_index(samples, args.threshold)
            confs = [s["token_conf"] for s in samples if s["token_conf"] is not None]
            row = {
                "id": claim["id"], "claim": claim["claim"], "gold": samples[0]["gold"],
                "retrieval": "titles", "n_shown": len(ctx["shown"]),
                "shown": [{"id": f"{page}:{line}", "text": text}
                          for _, page, line, text in ctx["shown"]],
                "gold_in_context": ctx["gold_in_context"],
                "gold_page_retrieved": ctx["gold_page_retrieved"],
                "samples": samples,
                "token_conf_std": statistics.pstdev(confs) if len(confs) >= 2 else None,
                "heuristic": {
                    "selected_index": selected_index, "n_verify": selected_index + 1,
                    "stop_reason": "threshold" if selected_index < N_DRAWS - 1
                    and samples[selected_index]["pred"] != NEI
                    and samples[selected_index]["token_conf"] is not None
                    and samples[selected_index]["token_conf"] >= args.threshold
                    else "budget",
                },
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 10 == 0 or index == len(claims):
                print(f"{index}/{len(claims)} claims in {time.monotonic() - started:.0f}s", flush=True)

    metrics = summarize(rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=LOCAL_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt-style", choices=PROMPTS, default="direct")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = PROMPTS[args.prompt_style][2]
    run(args)


if __name__ == "__main__":
    main()
