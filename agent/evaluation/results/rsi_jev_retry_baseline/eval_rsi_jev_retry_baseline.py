"""Compare zero-shot Jev and a fixed rule on three-draw evidence chunks.

Every chunk is visited. A controller can retry only the same chunk, at most
twice. Selected citations are pooled for one fresh final Qwen verification.
Gold and future draws are never included in controller state.
"""

import argparse
import gc
import importlib.metadata
import json
import math
import subprocess
import time
from pathlib import Path

from agent.data.build_evidence_batches import quality, scored
from agent.evaluation.results.rsi_jev_controller_baseline.eval_rsi_jev_controller_baseline import (
    RSI_MODEL, load_controller,
)
from agent.evaluation.results.sampling_retry.eval_sampling_retry import PROMPTS
from agent.train.replay_env import MAX_DRAWS, RETRY, STOP
from verifier.model import LocalModel, gold_sets, render, sample_claims


QUESTION = {"action": {
    "type": "choice",
    "instructions": (
        "Decide only whether the CURRENT verifier output for this fixed group is "
        "reasonable to keep. You see only the claim, verifier status and the sentences "
        "it actually cited; uncited retrieved sentences are hidden. STOP saves only the sentences "
        "explicitly cited in this output; saved citations from every group are combined "
        "later for a separate final claim verification. Every remaining group will still "
        "be processed. Do not automatically trust every cited sentence: citations must be "
        "relevant, justified, and no broader than necessary. RETRY asks the same verifier "
        "to sample the exact same claim and sentences again; it does not retrieve new "
        "information and costs one call. Retry if the verdict or citations are malformed, "
        "unsupported, missing, excessive, or otherwise unreliable. Judge the output, "
        "not the claim's final truth. Use only observed draws; at most two retries are "
        "available for this group."
    ),
    "criteria": {
        STOP: "Keep the latest verifier output and its valid, necessary citations for the final evidence pool; move to the next group.",
        RETRY: "Discard the latest output for this group and sample the same fixed input once more.",
    },
}}
THRESHOLD = 0.5
HEURISTIC_THRESHOLDS = (0.3, 0.5, 0.7)


def state_for(row, chunk_index, draw_index):
    chunk = row["chunks"][chunk_index]
    global_numbers = {item["id"]: f"E{i + 1}" for i, item in enumerate(row["shown"])}
    by_id = {item["id"]: item["text"] for item in chunk["evidence"]}
    observed = []
    for draw in chunk["draws"][:draw_index + 1]:
        cited = [{"id": item_id, "retrieval_eid": global_numbers[item_id],
                  "text": by_id[item_id]}
                 for item_id in dict.fromkeys(draw.get("evidence", [])) if item_id in by_id]
        observed.append({"draw_index": draw.get("draw_index"),
                         "verdict": draw.get("pred"), "format_ok": draw.get("format_ok"),
                         "token_conf": draw.get("token_conf"),
                         "stated_conf": draw.get("stated_conf"),
                         "cited_evidence": cited})
    public = {
        "claim": row["claim"],
        "current_group_index": chunk_index + 1,
        "total_groups": len(row["chunks"]),
        "current_group_sentence_count": len(chunk["evidence"]),
        "observed_verifier_outputs": observed,
        "remaining_retries_for_this_group": MAX_DRAWS - draw_index - 1,
    }
    return json.dumps(public, ensure_ascii=False, separators=(",", ":"))


def heuristic_action(draw, chunk_size=None, threshold=THRESHOLD):
    conf = draw.get("token_conf")
    good_conf = isinstance(conf, (int, float)) and math.isfinite(conf) and conf >= threshold
    excessive = chunk_size is not None and chunk_size >= 4 and len(set(draw.get("evidence", []))) >= chunk_size
    if draw.get("format_ok") and draw.get("pred") in ("SUPPORTED", "REFUTED") and good_conf and not excessive:
        return STOP
    return RETRY


def choose(model, state):
    started = time.monotonic()
    answer = model.predict(state=state, questions=QUESTION)["answers"]["action"]
    action = answer["choice"]
    if action not in (STOP, RETRY):
        raise ValueError(f"unexpected Jev action: {action!r}")
    return {"action": action, "reason": "model_choice", "confidence": answer.get("confidence"),
            "probabilities": answer.get("probabilities"), "seconds": time.monotonic() - started}


def select(row, controller=None, threshold=THRESHOLD):
    """Return a policy trace and the valid citations of the chosen draws."""
    saved = set()
    trace = []
    calls = 0
    forced_bad = 0
    forced_stops = 0
    for ci, chunk in enumerate(row["chunks"]):
        allowed = {x["id"] for x in chunk["evidence"]}
        decisions = []
        for di, draw in enumerate(chunk["draws"]):
            if di == MAX_DRAWS - 1:
                action = STOP
                meta = {"action": STOP, "reason": "retry_limit", "seconds": 0.0}
                forced_stops += 1
                if heuristic_action(draw, len(chunk["evidence"]), threshold) == RETRY:
                    forced_bad += 1
            elif controller is None:
                action = heuristic_action(draw, len(chunk["evidence"]), threshold)
                meta = {"action": action, "reason": "fixed_format_label_token_conf_and_no_whole_chunk_citation_rule",
                        "seconds": 0.0}
            else:
                meta = choose(controller, state_for(row, ci, di))
                action = meta["action"]
                calls += 1
            decisions.append({"draw_index": di, **meta})
            if action == STOP:
                if draw.get("format_ok"):
                    cited = set(draw.get("evidence", []))
                    if not cited <= allowed:
                        raise ValueError(f"claim {row['id']}: citation outside current chunk")
                    saved.update(cited)
                trace.append({"chunk_index": ci, "selected_draw_index": di,
                              "selected_pred": draw.get("pred"),
                              "selected_format_ok": draw.get("format_ok"),
                              "selected_citations": draw.get("evidence", []) if draw.get("format_ok") else [],
                              "cited_all_shown": bool(allowed) and allowed <= set(draw.get("evidence", [])),
                              "decisions": decisions})
                break
    selected = [item for item in row["shown"] if item["id"] in saved]
    used = sum(part["selected_draw_index"] + 1 for part in trace)
    return {"chunks": trace, "selected_evidence": selected,
            "chunk_verifier_calls": used, "extra_draws": used - len(trace),
            "controller_calls": calls, "forced_stops_after_retry_limit": forced_bad,
            "forced_stop_chunks": forced_stops,
            "cited_all_shown_chunks": sum(x["cited_all_shown"] for x in trace)}


def validate_source(source):
    files = [source / name for name in ("manifest.json", "metrics.json", "predictions.jsonl")]
    if any(not p.is_file() for p in files):
        raise FileNotFoundError(f"incomplete three-draw source: {source}")
    manifest = json.loads(files[0].read_text())
    metrics = json.loads(files[1].read_text())
    if (manifest.get("evaluation_code") != "agent.data.build_evidence_batches"
            or manifest.get("split", "dev") != "dev"
            or manifest.get("batches_only") is not True
            or manifest.get("draws_per_chunk") != MAX_DRAWS
            or manifest.get("do_sample") is not True
            or manifest.get("batch_size") != 10):
        raise ValueError("source must be the stochastic three-draw, 10-sentence batch-only run")
    with files[2].open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    n = manifest["n_claims"]
    if len(rows) != n or metrics.get("n_claims") != n or len({r["id"] for r in rows}) != n:
        raise ValueError("source claim count/IDs disagree")
    claims = {c["id"]: c for c in sample_claims(n, manifest["claim_seed"])}
    if set(claims) != {r["id"] for r in rows}:
        raise ValueError("source IDs differ from sampled FEVER dev claims")
    for row in rows:
        if row["claim"] != claims[row["id"]]["claim"]:
            raise ValueError("claim text differs from FEVER source")
        seen = [x for chunk in row["chunks"] for x in chunk["evidence"]]
        if seen != row["shown"]:
            raise ValueError(f"claim {row['id']}: chunk evidence differs from shown order")
        if any(not 1 <= len(c["evidence"]) <= 10 or len(c["draws"]) != MAX_DRAWS
               for c in row["chunks"]):
            raise ValueError(f"claim {row['id']}: invalid chunk/draw count")
    return manifest, rows, claims


def evidence_for(items):
    result = []
    for index, item in enumerate(items, 1):
        page, line = item["id"].rsplit(":", 1)
        result.append((f"E{index}", page, int(line), item["text"]))
    return result


def controller_diagnostics(row, selection, claim):
    """Hindsight-only diagnostics; never passed to either policy."""
    saved = {x["id"] for x in selection["selected_evidence"]}
    shown = {x["id"] for x in row["shown"]}
    gold = [{f"{page}:{line}" for page, line in group} for group in gold_sets(claim)]
    reachable = [group for group in gold if group and group <= shown]
    selected_wrong = 0
    avoidable_wrong = 0
    no_correct_candidate = 0
    retry_after_correct = 0
    early_stop_on_wrong = 0
    correct_retry_then_wrong_selection = 0
    for picked, chunk in zip(selection["chunks"], row["chunks"]):
        chosen = chunk["draws"][picked["selected_draw_index"]]
        candidates = [bool(draw.get("correct")) for draw in chunk["draws"]]
        earlier = chunk["draws"][:picked["selected_draw_index"]]
        retry_after_correct += sum(bool(draw.get("correct")) for draw in earlier)
        correct_retry_then_wrong_selection += int(
            not chosen.get("correct") and any(draw.get("correct") for draw in earlier))
        early_stop_on_wrong += int(
            picked["selected_draw_index"] < MAX_DRAWS - 1 and not chosen.get("correct"))
        if not chosen.get("correct"):
            selected_wrong += 1
            avoidable_wrong += int(any(candidates))
            no_correct_candidate += int(not any(candidates))
    return {"chunks": len(selection["chunks"]),
            "stop_at_draw_1": sum(x["selected_draw_index"] == 0 for x in selection["chunks"]),
            "stop_at_draw_2": sum(x["selected_draw_index"] == 1 for x in selection["chunks"]),
            "forced_stop_at_draw_3": sum(x["selected_draw_index"] == 2 for x in selection["chunks"]),
            "selected_wrong_global_label_chunks": selected_wrong,
            "retry_after_correct_draws": retry_after_correct,
            "early_stop_on_wrong_draws": early_stop_on_wrong,
            "correct_retry_then_wrong_selection_chunks": correct_retry_then_wrong_selection,
            "avoidable_wrong_selection_chunks": avoidable_wrong,
            "no_correct_candidate_chunks": no_correct_candidate,
            "gold_reachable": bool(reachable) if gold else None,
            "complete_gold_saved": any(group <= saved for group in reachable) if reachable else None,
            "saved_evidence_count": len(saved),
            "note": "Gold/future draws are used only after policy decisions. A per-chunk global FEVER label may be impossible to justify from that chunk; these are hindsight diagnostics, not online controller accuracy."}


def summarize(rows):
    def group(part):
        a = [r["heuristic_final"] for r in part]
        b = [r["jev_final"] for r in part]
        return {"n": len(part), "heuristic": quality(a), "jev": quality(b),
                "jev_accuracy_gained": sum(not x["correct"] and y["correct"] for x, y in zip(a, b)),
                "jev_accuracy_lost": sum(x["correct"] and not y["correct"] for x, y in zip(a, b)),
                "jev_fever_gained": sum(not x["fever_score"] and y["fever_score"] for x, y in zip(a, b)),
                "jev_fever_lost": sum(x["fever_score"] and not y["fever_score"] for x, y in zip(a, b))}
    def policy(name):
        selected = [r[name] for r in rows]
        return {"simulated_chunk_verifier_calls": sum(x["chunk_verifier_calls"] for x in selected),
                "simulated_total_qwen_calls_including_final": sum(x["chunk_verifier_calls"] + 1 for x in selected),
                "controller_calls": sum(x["controller_calls"] for x in selected),
                "extra_draws": sum(x["extra_draws"] for x in selected),
                "selected_evidence_total": sum(len(x["selected_evidence"]) for x in selected),
                "empty_evidence_claims": sum(not x["selected_evidence"] for x in selected),
                "cited_all_shown_chunks": sum(x["cited_all_shown_chunks"] for x in selected),
                "forced_stops_after_retry_limit": sum(x["forced_stops_after_retry_limit"] for x in selected)}
    return {"overall": group(rows),
            "gold_in_full_context": group([r for r in rows if r["gold_in_context"] is True]),
            "gold_not_in_full_context": group([r for r in rows if r["gold_in_context"] is False]),
            "gold_nei": group([r for r in rows if r["gold"] == "NOT_ENOUGH_INFO"]),
            "policies": {"heuristic": policy("heuristic"), "jev": policy("jev")},
            "source_precomputed_qwen_calls": sum(3 * len(r["heuristic"]["chunks"]) for r in rows),
            "new_final_qwen_calls": sum(r["new_final_qwen_calls"] for r in rows),
            "note": "Both policies read the same precomputed draws. Controller decisions use no gold or future draws. All chunks are processed. Final Qwen receives only selected citations; identical evidence sets share a final call. Simulated call counts differ from actual precomputation cost."}


def single_policy_metrics(rows, policy):
    outcomes = [row[f"{policy}_final"] for row in rows]
    traces = [row[policy] for row in rows]
    diagnostics = [row[f"{policy}_controller_diagnostics"] for row in rows]
    return {"quality": quality(outcomes),
            "gold_in_full_context": quality([row[f"{policy}_final"] for row in rows
                                            if row["gold_in_context"] is True]),
            "gold_not_in_full_context": quality([row[f"{policy}_final"] for row in rows
                                                if row["gold_in_context"] is False]),
            "gold_nei": quality([row[f"{policy}_final"] for row in rows
                                if row["gold"] == "NOT_ENOUGH_INFO"]),
            "simulated_chunk_verifier_calls": sum(x["chunk_verifier_calls"] for x in traces),
            "simulated_total_qwen_calls_including_final": sum(x["chunk_verifier_calls"] + 1 for x in traces),
            "controller_calls": sum(x["controller_calls"] for x in traces),
            "extra_draws": sum(x["extra_draws"] for x in traces),
            "selected_evidence_total": sum(len(x["selected_evidence"]) for x in traces),
            "empty_evidence_claims": sum(not x["selected_evidence"] for x in traces),
            "cited_all_shown_chunks": sum(x["cited_all_shown_chunks"] for x in traces),
            "forced_stop_chunks": sum(x["forced_stop_chunks"] for x in traces),
            "forced_stops_failing_fixed_quality_rule": sum(x["forced_stops_after_retry_limit"] for x in traces),
            "controller_hindsight": {
                "chunks": sum(x["chunks"] for x in diagnostics),
                "stop_at_draw_1": sum(x["stop_at_draw_1"] for x in diagnostics),
                "stop_at_draw_2": sum(x["stop_at_draw_2"] for x in diagnostics),
                "forced_stop_at_draw_3": sum(x["forced_stop_at_draw_3"] for x in diagnostics),
                "retry_after_correct_draws": sum(x["retry_after_correct_draws"] for x in diagnostics),
                "early_stop_on_wrong_draws": sum(x["early_stop_on_wrong_draws"] for x in diagnostics),
                "correct_retry_then_wrong_selection_chunks": sum(x["correct_retry_then_wrong_selection_chunks"] for x in diagnostics),
                "selected_wrong_global_label_chunks": sum(x["selected_wrong_global_label_chunks"] for x in diagnostics),
                "avoidable_wrong_selection_chunks": sum(x["avoidable_wrong_selection_chunks"] for x in diagnostics),
                "no_correct_candidate_chunks": sum(x["no_correct_candidate_chunks"] for x in diagnostics),
                "gold_reachable_claims": sum(x["gold_reachable"] is True for x in diagnostics),
                "complete_gold_saved_claims": sum(x["complete_gold_saved"] is True for x in diagnostics),
                "note": "Uses gold and all three draws only after decisions; not online controller accuracy. Chunk-level global labels can be unsupported by the chunk's limited evidence."
            }}


def threshold_label(threshold):
    return f"heuristic-retry-threshold-{str(threshold).replace('.', 'p')}"


def run(args):
    source = Path(args.source_run)
    source_manifest, source_rows, claims = validate_source(source)
    output = Path(args.output_dir)
    heuristic_dirs = {
        threshold: (Path(args.heuristic_results_root) / threshold_label(threshold) / "runs"
                    / output.parent.parent.name / output.parent.name / output.name)
        for threshold in HEURISTIC_THRESHOLDS
    }
    for directory in (output, *heuristic_dirs.values()):
        if directory.exists():
            raise FileExistsError(f"result directory exists: {directory}")
    trained_head = getattr(args, "trained_head", None)
    if trained_head:
        if args.backend != "rsi-jev":
            raise ValueError("a trained scorer head requires the RSI-Jev backend")
        from agent.train.jev_head import JevHeadPolicy
        controller = JevHeadPolicy(args.model, head_path=trained_head)
    else:
        controller = load_controller(args.backend, args.model)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip() or None
    manifest = {"evaluation_code": __name__, "backend": args.backend, "jev_model": args.model,
                "jev_package_version": importlib.metadata.version("rsi-jev" if args.backend == "rsi-jev" else "wev-ai"),
                "source_run": str(source), "source_commit": source_manifest.get("commit"),
                "trained_head": trained_head,
                "commit": revision, "question": QUESTION,
                "heuristic_thresholds": HEURISTIC_THRESHOLDS,
                "heuristic_rule": "STOP on valid SUPPORTED/REFUTED with finite verdict-token confidence >= threshold and without citing every sentence in a chunk of at least four; otherwise RETRY, at most twice per chunk",
                "verifier": {k: source_manifest[k] for k in ("model", "adapter", "prompt_style", "max_new_tokens", "generation_seed")},
                "final_verifier_do_sample": False,
                "split": "FEVER shared_task_dev; frozen validation claim IDs",
                "purpose": "evaluation_only_no_controller_training",
                "dev_only": True}
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for threshold, directory in heuristic_dirs.items():
        directory.mkdir(parents=True)
        item = {**manifest, "method": "fixed_heuristic", "threshold": threshold,
                "paired_jev_run": str(output)}
        (directory / "manifest.json").write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n")
    decisions = []
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(source_rows, 1):
            choice = {"id": row["id"],
                      "heuristics": {str(t): select(row, threshold=t) for t in HEURISTIC_THRESHOLDS},
                      "jev": select(row, controller)}
            choice["heuristic"] = choice["heuristics"][str(THRESHOLD)]
            decisions.append(choice)
            stream.write(json.dumps(choice, ensure_ascii=False) + "\n")
            stream.flush()
            if i % 10 == 0:
                print(f"{i}/{len(source_rows)} controller decisions", flush=True)
    del controller
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    system, thinking, _ = PROMPTS[source_manifest["prompt_style"]]
    verifier = LocalModel(source_manifest["model"],
                          batch_size=source_manifest["model_batch_size"],
                          adapter=source_manifest["adapter"], system_prompt=system,
                          max_new_tokens=source_manifest["max_new_tokens"],
                          enable_thinking=thinking, do_sample=False)
    results = []
    heuristic_streams = {t: (directory / "predictions.jsonl").open("w", encoding="utf-8")
                         for t, directory in heuristic_dirs.items()}
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for i, (row, decision) in enumerate(zip(source_rows, decisions), 1):
                finals = {}
                cache = {}
                policies = {"jev": decision["jev"],
                            **{threshold_label(t): decision["heuristics"][str(t)]
                               for t in HEURISTIC_THRESHOLDS}}
                for policy, policy_decision in policies.items():
                    items = policy_decision["selected_evidence"]
                    key = tuple(x["id"] for x in items)
                    if key not in cache:
                        evidence = evidence_for(items)
                        generation = verifier.run([render(row["claim"], evidence)],
                                                  seed=source_manifest["generation_seed"] + int(row["id"]))[0]
                        cache[key] = scored(claims[row["id"]], evidence, generation)
                    finals[policy] = cache[key]
                result = {"id": row["id"], "gold": row["gold"],
                          "gold_in_context": row["gold_in_context"],
                          "heuristic": decision["heuristic"], "jev": decision["jev"],
                          "heuristic_controller_diagnostics": controller_diagnostics(
                              row, decision["heuristic"], claims[row["id"]]),
                          "jev_controller_diagnostics": controller_diagnostics(
                              row, decision["jev"], claims[row["id"]]),
                          "heuristic_final": finals[threshold_label(THRESHOLD)],
                          "jev_final": finals["jev"],
                          "new_final_qwen_calls": len(cache)}
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                for threshold, hstream in heuristic_streams.items():
                    label = threshold_label(threshold)
                    hrow = {"id": row["id"], "gold": row["gold"],
                            "gold_in_context": row["gold_in_context"],
                            "heuristic": decision["heuristics"][str(threshold)],
                            "jev": decision["jev"],
                            "heuristic_controller_diagnostics": controller_diagnostics(
                                row, decision["heuristics"][str(threshold)], claims[row["id"]]),
                            "jev_controller_diagnostics": result["jev_controller_diagnostics"],
                            "heuristic_final": finals[label], "jev_final": finals["jev"]}
                    hstream.write(json.dumps(hrow, ensure_ascii=False) + "\n")
                    hstream.flush()
                if i % 10 == 0:
                    print(f"{i}/{len(source_rows)} final Qwen verifications", flush=True)
    finally:
        for hstream in heuristic_streams.values():
            hstream.close()
    metrics = summarize(results)
    metrics["controller_hindsight"] = {
        "jev": single_policy_metrics(results, "jev")["controller_hindsight"],
        "heuristic_threshold_0p5": single_policy_metrics(results, "heuristic")["controller_hindsight"],
    }
    chunks = [chunk for row in source_rows for chunk in row["chunks"]]
    metrics["source_chunk_disagreement"] = {
        "n_chunks": len(chunks),
        "verdict": sum(len({d["pred"] for d in c["draws"]}) > 1 for c in chunks),
        "evidence": sum(len({tuple(d["evidence"]) for d in c["draws"]}) > 1 for c in chunks),
        "either": sum(len({d["pred"] for d in c["draws"]}) > 1
                      or len({tuple(d["evidence"]) for d in c["draws"]}) > 1 for c in chunks),
    }
    metrics["threshold_runs"] = {str(t): str(directory) for t, directory in heuristic_dirs.items()}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    for threshold, directory in heuristic_dirs.items():
        with (directory / "predictions.jsonl").open(encoding="utf-8") as source_stream:
            hrows = [json.loads(line) for line in source_stream]
        hmetrics = single_policy_metrics(hrows, "heuristic")
        hmetrics["threshold"] = threshold
        hmetrics["paired_jev"] = single_policy_metrics(hrows, "jev")["quality"]
        hmetrics["jev_vs_heuristic"] = {
            "accuracy_gained": sum(not r["heuristic_final"]["correct"] and r["jev_final"]["correct"] for r in hrows),
            "accuracy_lost": sum(r["heuristic_final"]["correct"] and not r["jev_final"]["correct"] for r in hrows),
            "fever_gained": sum(not r["heuristic_final"]["fever_score"] and r["jev_final"]["fever_score"] for r in hrows),
            "fever_lost": sum(r["heuristic_final"]["fever_score"] and not r["jev_final"]["fever_score"] for r in hrows),
        }
        (directory / "metrics.json").write_text(json.dumps(hmetrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rsi-jev", "openjev"), default="rsi-jev")
    parser.add_argument("--model", default=RSI_MODEL)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heuristic-results-root", default="agent/evaluation/results")
    parser.add_argument("--trained-head", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
