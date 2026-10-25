"""Evaluate a Jev-style controller over a completed Qwen evidence-batch run.

RSI-Jev and a fixed heuristic see the same completed Qwen batch summaries.
They independently choose the evidence set for a new final Qwen verification.
Only after both choices are fixed do we call Qwen. Identical routes share one
final call. No gold labels, evidence text, or final Qwen outputs enter either
controller decision.
"""

import argparse
import gc
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

from agent.data.build_evidence_batches import quality, scored
from agent.evaluation.results.sampling_retry.eval_sampling_retry import PROMPTS
from verifier.model import LocalModel, render, sample_claims


RSI_MODEL = "shgao/rsi-jev-v2.1-qwen3.5-2b"
OPENJEV_MODEL = "alanhuangya/OpenJev-1.7B"
UNION = "CITED_UNION"
FULL = "FULL_CONTEXT"
HEURISTIC_CONFIDENCE_THRESHOLD = 0.5
QUESTION = {
    "route": {
        "type": "choice",
        "instructions": "Choose which evidence set the fact verifier should read for its final decision. Batch summaries are noisy; favor retaining all evidence when the cited union may omit needed facts. Do not decide the claim's verdict yourself.",
        "criteria": {
            UNION: "Use the union of sentences cited by the batch verifiers. This is shorter but may omit necessary evidence.",
            FULL: "Use every retrieved sentence again. This preserves coverage but uses a longer input.",
        },
    }
}


class RSIAdapter:
    """Expose RSI-Jev's Decider with the same answer shape as OpenJev."""

    def __init__(self, decider):
        self.decider = decider

    def predict(self, *, state, questions):
        return {"answers": self.decider.decide(state, questions)}


def load_controller(backend, model_name):
    if backend == "rsi-jev":
        from rsijev import Decider
        return RSIAdapter(Decider(model_name))
    if backend == "openjev":
        import wev
        return wev.load(model_name)
    raise ValueError(f"unsupported controller backend: {backend}")


def controller_state(row):
    """Serialize only information available before the final verification."""
    batches = []
    for index, output in enumerate(row["batches"], 1):
        batches.append({
            "batch": index,
            "verdict": output["pred"],
            "format_ok": output["format_ok"],
            "verdict_token_confidence": output["token_conf"],
            "stated_confidence": output["stated_conf"],
            "cited_sentence_ids": output["evidence"],
        })
    public = {"claim": row["claim"], "retrieved_sentence_count": row["n_shown"],
              "cited_union_sentence_count": row["n_union"],
              "batch_results": batches}
    return json.dumps(public, ensure_ascii=False, separators=(",", ":"))


def choose_route(model, row):
    if row["n_union"] == 0:
        return {"route": FULL, "probabilities": None, "confidence": None,
                "reason": "empty_union_fallback", "seconds": 0.0, "model_calls": 0}
    started = time.monotonic()
    answer = model.predict(state=controller_state(row), questions=QUESTION)["answers"]["route"]
    route = answer["choice"]
    if route not in (UNION, FULL):
        raise ValueError(f"unexpected controller action: {route!r}")
    probabilities = answer.get("probabilities", {})
    return {"route": route, "probabilities": probabilities,
            "selected_probability": probabilities.get(route),
            "confidence": answer.get("confidence"), "reason": "model_choice",
            "seconds": time.monotonic() - started, "model_calls": 1}


def heuristic_route(row, threshold=HEURISTIC_CONFIDENCE_THRESHOLD):
    """Predeclared route rule using the same batch summaries as Jev."""
    if row["n_union"] == 0:
        route, reason = FULL, "empty_union"
    elif not row["batches"] or any(not b["format_ok"] for b in row["batches"]):
        route, reason = FULL, "missing_or_invalid_batch"
    elif any(b["pred"] == "NOT_ENOUGH_INFO" for b in row["batches"]):
        route, reason = FULL, "nei_batch"
    elif len({b["pred"] for b in row["batches"]}) > 1:
        route, reason = FULL, "batch_verdict_disagreement"
    elif any(b["token_conf"] is None or b["token_conf"] < threshold
             for b in row["batches"]):
        route, reason = FULL, "low_or_missing_token_confidence"
    else:
        route, reason = UNION, "consistent_confident_batches"
    return {"route": route, "reason": reason, "threshold": threshold,
            "model_calls": 0}


def route_evidence(row, route):
    """Rebuild prompt-local E ids from saved retrieval-order evidence."""
    source = row["union"] if route == UNION else row["shown"]
    evidence = []
    for index, item in enumerate(source, 1):
        page, line = item["id"].rsplit(":", 1)
        evidence.append((f"E{index}", page, int(line), item["text"]))
    return evidence


def verify_routes(verifier, claim, row, routes, seed):
    """Run Qwen after both choices; reuse one result when routes agree."""
    results = {}
    for route in dict.fromkeys(routes.values()):
        evidence = route_evidence(row, route)
        generation = verifier.run([render(row["claim"], evidence)], seed=seed)[0]
        results[route] = scored(claim, evidence, generation)
    return {policy: results[route] for policy, route in routes.items()}, len(results)


def summarize(rows):
    def group(part):
        heuristic = [r["heuristic_final"] for r in part]
        controlled = [r["controller_final"] for r in part]
        return {"n": len(part), "heuristic_controller": quality(heuristic),
                "jev_controller": quality(controlled),
                "jev_vs_heuristic_accuracy_gained": sum(
                    not a["correct"] and b["correct"] for a, b in zip(heuristic, controlled)),
                "jev_vs_heuristic_accuracy_lost": sum(
                    a["correct"] and not b["correct"] for a, b in zip(heuristic, controlled)),
                "jev_vs_heuristic_fever_gained": sum(
                    not a["fever_score"] and b["fever_score"]
                    for a, b in zip(heuristic, controlled)),
                "jev_vs_heuristic_fever_lost": sum(
                    a["fever_score"] and not b["fever_score"]
                    for a, b in zip(heuristic, controlled))}

    def mean(values):
        return sum(values) / len(values) if values else None

    return {"overall": group(rows),
            "gold_in_full_context": group([r for r in rows if r["gold_in_context"] is True]),
            "gold_not_in_full_context": group([r for r in rows if r["gold_in_context"] is False]),
            "gold_nei": group([r for r in rows if r["gold"] == "NOT_ENOUGH_INFO"]),
            "routes": {
                policy: {route: sum(r[policy]["route"] == route for r in rows)
                         for route in (UNION, FULL)}
                for policy in ("heuristic", "controller")},
            "route_agreement": sum(r["heuristic"]["route"] == r["controller"]["route"]
                                   for r in rows),
            "empty_union_fallbacks": sum(r["controller"]["reason"] == "empty_union_fallback"
                                         for r in rows),
            "jev_model_calls": sum(r["controller"]["model_calls"] for r in rows),
            "mean_jev_seconds": mean([r["controller"]["seconds"] for r in rows]),
            "shared_batch_qwen_calls": sum(len(r["batches"]) for r in rows),
            "new_final_qwen_calls": sum(r["n_final_qwen_calls"] for r in rows),
            "mean_qwen_calls_per_policy": mean([len(r["batches"]) + 1 for r in rows]),
            "note": "Both controllers chose from the same saved Qwen batch outputs before any final Qwen call. A single new final call is shared when routes agree; otherwise each route gets one. The source run's historical baseline/final outputs are not used. No gold or final Qwen output enters either route decision."}


def run(args):
    source = Path(args.source_run)
    manifest_path = source / "manifest.json"
    predictions_path = source / "predictions.jsonl"
    metrics_path = source / "metrics.json"
    if not all(path.is_file() for path in (manifest_path, predictions_path, metrics_path)):
        raise FileNotFoundError(f"source batch run is incomplete in {source}")
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("evaluation_code") not in {
        "agent.data.build_evidence_batches",
        "agent.evaluation.results.evidence_batches.eval_evidence_batches",
    }:
        raise ValueError("source-run must be an evidence_batches result")
    if source_manifest.get("batches_only") is not True:
        raise ValueError("source-run must use --batches-only; historical final outputs are not reused")
    if source_manifest.get("do_sample") is not False:
        raise ValueError("controller comparison requires deterministic verifier decoding")
    with predictions_path.open(encoding="utf-8") as source_stream:
        source_rows = [json.loads(line) for line in source_stream]
    n_claims = source_manifest["n_claims"]
    if len(source_rows) != n_claims or json.loads(metrics_path.read_text())["n_claims"] != n_claims:
        raise ValueError("source predictions do not match the completed batch count")
    claims = {claim["id"]: claim for claim in
              sample_claims(n_claims, source_manifest["claim_seed"])}
    if len(claims) != n_claims or {row["id"] for row in source_rows} != set(claims):
        raise ValueError("source claim IDs differ from the sampled FEVER dev claims")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"result directory exists: {output_dir}; use a new run name")
    controller_model = load_controller(args.backend, args.model)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip() or None
    manifest = {"evaluation_code": "agent.evaluation.results.rsi_jev_controller_baseline.eval_rsi_jev_controller_baseline",
                "model": args.model, "backend": args.backend, "source_run": str(source),
                "source_commit": source_manifest.get("commit"), "commit": revision,
                "verifier": {key: source_manifest[key] for key in
                             ("model", "adapter", "prompt_style", "max_new_tokens",
                              "model_batch_size", "generation_seed", "do_sample")},
                "controller_package_version": importlib.metadata.version(
                    "rsi-jev" if args.backend == "rsi-jev" else "wev-ai"),
                "question": QUESTION, "empty_union_action": FULL,
                "heuristic_rule": {
                    "route_to_full_context_if": ["empty citation union", "missing or invalid batch",
                                                 "any NEI batch", "batch verdict disagreement",
                                                 "missing or low verdict-token confidence"],
                    "token_confidence_threshold": HEURISTIC_CONFIDENCE_THRESHOLD,
                    "otherwise": UNION},
                "state_fields": ["claim", "retrieved_sentence_count", "cited_union_sentence_count",
                                 "batch_results: verdict, format, token/stated confidence, cited IDs"],
                "note": "Both routes are selected from the same saved batch outputs before new final verifier calls. Historical source final outputs are ignored. FEVER dev claims are for evaluation only."}
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    decisions = []
    with (output_dir / "decisions.jsonl").open("w", encoding="utf-8") as decision_stream:
        for index, source_row in enumerate(source_rows, 1):
            controller = choose_route(controller_model, source_row)
            heuristic = heuristic_route(source_row)
            decision = {"id": source_row["id"], "controller": controller,
                        "heuristic": heuristic, "controller_state": controller_state(source_row)}
            decisions.append(decision)
            decision_stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
            decision_stream.flush()
            if index % 10 == 0:
                print(f"{index}/{len(source_rows)} controller decisions", flush=True)
    del controller_model
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    system, thinking, _ = PROMPTS[source_manifest["prompt_style"]]
    verifier = LocalModel(source_manifest["model"],
                          batch_size=source_manifest["model_batch_size"],
                          adapter=source_manifest["adapter"],
                          system_prompt=system,
                          max_new_tokens=source_manifest["max_new_tokens"],
                          enable_thinking=thinking)
    rows = []
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as out_stream:
        for index, (source_row, decision) in enumerate(zip(source_rows, decisions), 1):
            controller = decision["controller"]
            heuristic = decision["heuristic"]
            routes = {"controller": controller["route"], "heuristic": heuristic["route"]}
            final_outputs, n_final_calls = verify_routes(
                verifier, claims[source_row["id"]], source_row, routes,
                source_manifest["generation_seed"] + int(source_row["id"]))
            row = {"id": source_row["id"], "gold": source_row["gold"],
                   "gold_in_context": source_row["gold_in_context"],
                   "n_shown": source_row["n_shown"], "n_union": source_row["n_union"],
                   "controller_state": decision["controller_state"],
                   "controller": controller,
                   "heuristic": heuristic,
                   "controller_final": final_outputs["controller"],
                   "heuristic_final": final_outputs["heuristic"],
                   "n_final_qwen_calls": n_final_calls,
                   "batches": source_row["batches"]}
            rows.append(row)
            out_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_stream.flush()
            if index % 10 == 0:
                print(f"{index}/{len(source_rows)} final Qwen verifications", flush=True)
    if not rows:
        raise ValueError("source run has no predictions")
    metrics = summarize(rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rsi-jev", "openjev"), default="rsi-jev")
    parser.add_argument("--model", default=None)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.model is None:
        args.model = RSI_MODEL if args.backend == "rsi-jev" else OPENJEV_MODEL
    run(args)


if __name__ == "__main__":
    main()
