"""Evaluate STOP/RETRY plus per-sentence KEEP/DROP on fixed three-draw chunks.

KEEP decisions accumulate across RETRY and STOP. All chunks are visited, and
the selected set is sent to one fresh final Qwen verifier. No gold reaches Jev.
"""

import argparse
import gc
import importlib.metadata
import json
import subprocess
from pathlib import Path

from agent.data.build_evidence_batches import quality, scored
from agent.evaluation.pooled_retry import FINAL_QUESTION, choose_pooled
from agent.evaluation.results.rsi_jev_controller_baseline.eval_rsi_jev_controller_baseline import (
    RSI_MODEL, load_controller,
)
from agent.evaluation.results.rsi_jev_retry_baseline.eval_rsi_jev_retry_baseline import (
    THRESHOLD, evidence_for, heuristic_action, validate_source,
)
from agent.evaluation.results.sampling_retry.eval_sampling_retry import PROMPTS
from agent.train.replay_env import MAX_DRAWS, RETRY, STOP
from verifier.model import LocalModel, gold_sets, render

KEEP = "KEEP"
DROP = "DROP"
TERMINAL_LABELS = {"ACCEPT": "SUPPORTED", "REJECT": "REFUTED",
                   "NEI": "NOT_ENOUGH_INFO"}
TERMINAL_QUESTION = {"verdict": {"type": "choice",
    "instructions": ("After all evidence groups and pooled verification have been processed, "
                     "record your own final judgment of the claim using only the saved "
                     "sentences and observed verifier outputs. ACCEPT means supported, REJECT means refuted, "
                     "and NEI means the saved evidence is insufficient. Do not assume a "
                     "chunk verdict alone proves the full claim. This is a diagnostic output; "
                     "a separate Qwen verifier still makes the main final prediction."),
    "criteria": {"ACCEPT": "Saved evidence supports the claim.",
                 "REJECT": "Saved evidence refutes the claim.",
                 "NEI": "Saved evidence does not justify either conclusion."}}}
ACTION = {"type": "choice", "instructions": (
    "For this fixed evidence group, decide whether to STOP after this verifier draw or "
    "RETRY the exact same input. Independently decide which individual sentences to KEEP. "
    "You can KEEP or DROP only sentences actually cited by this verifier draw; uncited "
    "retrieved sentences are hidden. KEEP decisions from this draw are saved even if "
    "you RETRY; they accumulate as a SET "
    "with saved sentences from all previous draws and groups. Every group will still be "
    "processed. RETRY reveals no new retrieved sentences and costs one verifier call. "
    "Use RETRY when another sample may help; avoid accumulating irrelevant evidence. "
    "You have at most two retries per group. Do not decide the final claim verdict."),
    "criteria": {STOP: "Keep selected sentences from this draw and move to the next group.",
                 RETRY: "Keep selected sentences from this draw and sample the same group again."}}


def cited_candidates(chunk, draw):
    by_id = {item["id"]: item for item in chunk["evidence"]}
    return [by_id[item_id] for item_id in dict.fromkeys(draw.get("evidence", []))
            if item_id in by_id]


def questions_for(chunk, draw, global_numbers, allow_retry=True):
    questions = {"action": ACTION} if allow_retry else {}
    for index, item in enumerate(cited_candidates(chunk, draw)):
        eid = global_numbers[item["id"]]
        questions[f"keep_{index}"] = {
            "type": "choice",
            "instructions": (f"Should the current verifier-cited sentence {eid} ({item['id']}) "
                             "be saved for the final claim verifier? Its text is in the current "
                             "verifier output. "
                             "Select KEEP only if it is relevant, factually useful, and not redundant. "
                             "Avoid keeping every shown sentence merely for safety. "
                             "This KEEP decision persists even when the group is retried."),
            "criteria": {KEEP: "Save this exact sentence in the final evidence set.",
                         DROP: "Do not save this sentence from this draw."},
        }
    return questions


def selection_state_for(row, chunk_index, draw_index):
    """Only the current verifier output; no full chunk or earlier draw text."""
    chunk = row["chunks"][chunk_index]
    draw = chunk["draws"][draw_index]
    numbers = {item["id"]: f"E{i + 1}" for i, item in enumerate(row["shown"])}
    cited = [{"id": item["id"], "retrieval_eid": numbers[item["id"]],
              "text": item["text"]} for item in cited_candidates(chunk, draw)]
    return json.dumps({"claim": row["claim"], "current_group_index": chunk_index + 1,
                       "total_groups": len(row["chunks"]),
                       "current_draw_index": draw_index,
                       "remaining_retries": MAX_DRAWS - draw_index - 1,
                       "verifier_output": {"verdict": draw.get("pred"),
                                           "format_ok": draw.get("format_ok"),
                                           "token_conf": draw.get("token_conf"),
                                           "stated_conf": draw.get("stated_conf"),
                                           "cited_evidence": cited}},
                      ensure_ascii=False, separators=(",", ":"))


def select(row, model=None, threshold=THRESHOLD):
    numbers = {item["id"]: f"E{i + 1}" for i, item in enumerate(row["shown"])}
    kept = set()
    chunks = []
    controller_calls = 0
    for ci, chunk in enumerate(row["chunks"]):
        attempts = []
        chunk_kept = set()
        for di, draw in enumerate(chunk["draws"]):
            allow_retry = di < MAX_DRAWS - 1
            if model is None:
                chosen = set(draw.get("evidence", [])) if draw.get("format_ok") else set()
                action = heuristic_action(draw, len(chunk["evidence"]), threshold) if allow_retry else STOP
                probabilities = None
            else:
                questions = questions_for(chunk, draw, numbers, allow_retry)
                answers = model.predict(state=selection_state_for(row, ci, di), questions=questions)["answers"] if questions else {}
                controller_calls += int(bool(questions))
                action = answers["action"]["choice"] if allow_retry else STOP
                if action not in (STOP, RETRY):
                    raise ValueError(f"invalid controller action {action!r}")
                chosen = set()
                for index, item in enumerate(cited_candidates(chunk, draw)):
                    choice = answers[f"keep_{index}"]["choice"]
                    if choice not in (KEEP, DROP):
                        raise ValueError(f"invalid evidence choice {choice!r}")
                    if choice == KEEP:
                        chosen.add(item["id"])
                probabilities = {key: answer.get("probabilities") for key, answer in answers.items()}
            allowed = {item["id"] for item in chunk["evidence"]}
            if not chosen <= allowed:
                raise ValueError(f"claim {row['id']}: selected evidence outside chunk")
            chunk_kept.update(chosen)
            attempts.append({"draw_index": di, "action": action,
                             "kept_ids": sorted(chosen), "kept_retrieval_eids":
                             [numbers[item["id"]] for item in chunk["evidence"] if item["id"] in chosen],
                             "probabilities": probabilities})
            if action == STOP:
                break
        kept.update(chunk_kept)
        chunks.append({"chunk_index": ci, "attempts": attempts,
                       "kept_ids": sorted(chunk_kept)})
    selected = [item for item in row["shown"] if item["id"] in kept]
    calls = sum(len(chunk["attempts"]) for chunk in chunks)
    return {"chunks": chunks, "selected_evidence": selected,
            "chunk_verifier_calls": calls, "extra_draws": calls - len(chunks),
            "controller_calls": controller_calls,
            "evidence_keep_decisions": sum(len(a["kept_ids"]) for c in chunks for a in c["attempts"]),
            "unique_evidence_kept": len(selected),
            "duplicate_keep_decisions_removed_by_set":
                sum(len(a["kept_ids"]) for c in chunks for a in c["attempts"]) - len(selected)}


def terminal_verdict(row, policy, model, pooled_selected=None):
    """One record-only choice after evidence selection and pooled verification."""
    numbers = {item["id"]: f"E{i + 1}" for i, item in enumerate(row["shown"])}
    state = {"claim": row["claim"],
             "saved_evidence": [{"id": item["id"], "retrieval_eid": numbers[item["id"]],
                                 "text": item["text"]}
                                for item in policy["selected_evidence"]],
             "observed_chunk_summaries": []}
    for chunk, decision in zip(row["chunks"], policy["chunks"]):
        last = decision["attempts"][-1]
        draw = chunk["draws"][last["draw_index"]]
        state["observed_chunk_summaries"].append({
            "chunk_index": decision["chunk_index"],
            "draws_used": len(decision["attempts"]), "verdict": draw.get("pred"),
            "format_ok": draw.get("format_ok"), "token_conf": draw.get("token_conf"),
            "saved_ids": decision["kept_ids"]})
    if pooled_selected is not None:
        state["selected_pooled_verifier_output"] = {
            "verdict": pooled_selected.get("pred"),
            "format_ok": pooled_selected.get("format_ok"),
            "token_conf": pooled_selected.get("token_conf"),
            "cited_ids": pooled_selected.get("evidence", [])}
    answer = model.predict(state=json.dumps(state, ensure_ascii=False, separators=(",", ":")),
                           questions=TERMINAL_QUESTION)["answers"]["verdict"]
    choice = answer["choice"]
    if choice not in TERMINAL_LABELS:
        raise ValueError(f"invalid terminal controller verdict: {choice!r}")
    return {"choice": choice, "fever_label": TERMINAL_LABELS[choice],
            "confidence": answer.get("confidence"),
            "probabilities": answer.get("probabilities")}


def evidence_diagnostic(row, policy, claim):
    shown = {item["id"] for item in row["shown"]}
    saved = {item["id"] for item in policy["selected_evidence"]}
    gold = [{f"{p}:{line}" for p, line in group} for group in gold_sets(claim)]
    reachable = [group for group in gold if group and group <= shown]
    return {"gold_reachable": bool(reachable) if gold else None,
            "complete_gold_saved": any(group <= saved for group in reachable) if reachable else None,
            "saved_count": len(saved),
            "gold_used_only_after_decisions": True}


def action_diagnostic(row, policy):
    """Hindsight comparison to global claim labels, never exposed to a policy."""
    result = {"chunks": len(policy["chunks"]), "retry_after_correct_draws": 0,
              "early_stop_on_wrong_draws": 0, "forced_stop_on_wrong_draws": 0,
              "correct_retry_then_wrong_selected_chunks": 0,
              "no_correct_candidate_chunks": 0,
              "stop_at_draw_1": 0, "stop_at_draw_2": 0, "forced_stop_at_draw_3": 0}
    for chunk, decision in zip(row["chunks"], policy["chunks"]):
        attempts = decision["attempts"]
        last = attempts[-1]["draw_index"]
        result[f"{'forced_stop' if last == 2 else 'stop'}_at_draw_{last + 1}"] += 1
        result["retry_after_correct_draws"] += sum(
            bool(chunk["draws"][a["draw_index"]].get("correct"))
            for a in attempts[:-1] if a["action"] == RETRY)
        selected_correct = bool(chunk["draws"][last].get("correct"))
        if not selected_correct:
            if last == 2:
                result["forced_stop_on_wrong_draws"] += 1
            else:
                result["early_stop_on_wrong_draws"] += 1
            result["correct_retry_then_wrong_selected_chunks"] += int(any(
                chunk["draws"][a["draw_index"]].get("correct") for a in attempts[:-1]))
        result["no_correct_candidate_chunks"] += int(not any(
            draw.get("correct") for draw in chunk["draws"]))
    return result


def summarize(rows):
    def group(part):
        h = [r["heuristic_final"] for r in part]
        j = [r["jev_final"] for r in part]
        return {"n": len(part), "heuristic": quality(h), "jev": quality(j),
                "jev_fever_gained": sum(not a["fever_score"] and b["fever_score"] for a, b in zip(h, j)),
                "jev_fever_lost": sum(a["fever_score"] and not b["fever_score"] for a, b in zip(h, j))}
    def policy(name):
        traces = [r[name] for r in rows]
        diagnostics = [r[f"{name}_evidence_diagnostic"] for r in rows]
        actions = [r[f"{name}_action_diagnostic"] for r in rows]
        pools = [r[f"{name}_pooled_retry"] for r in rows]
        return {"simulated_qwen_calls_including_final": sum(
                    x["chunk_verifier_calls"] + p["simulated_qwen_calls"]
                    for x, p in zip(traces, pools)),
                "simulated_pooled_qwen_calls": sum(p["simulated_qwen_calls"] for p in pools),
                "pooled_retry_after_correct_draws": sum(p["retry_after_correct_draws"] for p in pools),
                "pooled_early_stop_on_wrong": sum(p["early_stop_on_wrong"] for p in pools),
                "pooled_forced_stop_on_wrong": sum(p["forced_stop_on_wrong"] for p in pools),
                "controller_calls": sum(x["controller_calls"] + p["controller_calls"]
                                        for x, p in zip(traces, pools)) + (len(rows) if name == "jev" else 0),
                "extra_draws": sum(x["extra_draws"] for x in traces),
                "keep_decisions": sum(x["evidence_keep_decisions"] for x in traces),
                "unique_evidence_kept": sum(x["unique_evidence_kept"] for x in traces),
                "duplicate_keep_decisions_removed_by_set": sum(x["duplicate_keep_decisions_removed_by_set"] for x in traces),
                "empty_evidence_claims": sum(not x["selected_evidence"] for x in traces),
                "gold_reachable_claims": sum(x["gold_reachable"] is True for x in diagnostics),
                "complete_gold_saved_claims": sum(x["complete_gold_saved"] is True for x in diagnostics),
                "action_hindsight": {key: sum(x[key] for x in actions)
                                      for key in actions[0] if key != "chunks"} if actions else {}}
    return {"overall": group(rows),
            "gold_in_full_context": group([r for r in rows if r["gold_in_context"] is True]),
            "gold_not_in_full_context": group([r for r in rows if r["gold_in_context"] is False]),
            "gold_nei": group([r for r in rows if r["gold"] == "NOT_ENOUGH_INFO"]),
            "policies": {"heuristic": policy("heuristic"), "jev": policy("jev")},
            "jev_terminal_verdict": {
                "n": len(rows),
                "accuracy": (sum(r["jev_terminal_verdict"]["fever_label"] == r["gold"]
                                 for r in rows) / len(rows)) if rows else None,
                "confusion": {gold: {pred: sum(r["gold"] == gold and
                                               r["jev_terminal_verdict"]["fever_label"] == pred
                                               for r in rows)
                                     for pred in TERMINAL_LABELS.values()}
                              for gold in TERMINAL_LABELS.values()},
                "status": "record_only_not_used_for_final_qwen_or_training_reward"},
            "terminal_vs_pooled_verifier_disagreement": sum(
                r["jev_terminal_verdict"]["fever_label"] != r["jev_final"]["pred"]
                for r in rows),
            "actual_new_final_qwen_calls": sum(r["new_final_qwen_calls"] for r in rows),
            "note": "Each chunk has its own KEEP set. Three new pooled-verifier draws use unchanged pooled evidence; STOP/RETRY selects one. Actual precomputation costs three calls per unique pool even when simulated stopping is earlier. Gold and future draws are used only in post-hoc diagnostics."}


def run(args):
    source = Path(args.source_run)
    manifest, source_rows, claims = validate_source(source)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"result directory exists: {output}")
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
    record = {"evaluation_code": __name__, "split": "FEVER shared_task_dev; frozen validation IDs",
              "purpose": "evaluation_only_no_controller_training", "source_run": str(source),
              "source_commit": manifest.get("commit"), "commit": revision,
              "jev_backend": args.backend, "jev_model": args.model,
              "trained_head": trained_head,
              "jev_package_version": importlib.metadata.version("rsi-jev" if args.backend == "rsi-jev" else "wev-ai"),
              "action_question": ACTION, "keep_question_builder": "questions_for",
              "terminal_question": TERMINAL_QUESTION,
              "pooled_retry_question": FINAL_QUESTION,
              "pooled_draws": MAX_DRAWS, "pooled_do_sample": True,
              "pooled_temperature": manifest.get("temperature", 0.7),
              "pooled_top_p": manifest.get("top_p", 0.9),
              "pooled_seed_offset": 100000,
              "heuristic_threshold": args.threshold,
              "method": "Each chunk draw yields STOP/RETRY plus cited-sentence KEEP/DROP; chunk sets are pooled; three fresh final-verifier samples are generated from the unchanged pooled input and a second STOP/RETRY policy selects one."}
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    decisions = []
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(source_rows, 1):
            decision = {"id": row["id"], "heuristic": select(row, threshold=args.threshold),
                        "jev": select(row, controller)}
            decisions.append(decision)
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 10 == 0:
                print(f"{index}/{len(source_rows)} selection decisions", flush=True)
    del controller
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    system, thinking, _ = PROMPTS[manifest["prompt_style"]]
    verifier = LocalModel(manifest["model"], batch_size=manifest["model_batch_size"],
                          adapter=manifest["adapter"], system_prompt=system,
                          max_new_tokens=manifest["max_new_tokens"],
                          enable_thinking=thinking, do_sample=True,
                          temperature=manifest.get("temperature", 0.7),
                          top_p=manifest.get("top_p", 0.9))
    pooled_rows = []
    with (output / "pooled_samples.jsonl").open("w", encoding="utf-8") as pooled_stream:
        for index, (row, decision) in enumerate(zip(source_rows, decisions), 1):
            cache = {}
            for policy in ("heuristic", "jev"):
                items = decision[policy]["selected_evidence"]
                key = tuple(item["id"] for item in items)
                if key not in cache:
                    evidence = evidence_for(items)
                    prompt = render(row["claim"], evidence)
                    generations = verifier.run([prompt] * MAX_DRAWS,
                                               seed=manifest["generation_seed"] + 100000 + int(row["id"]))
                    if len(generations) != MAX_DRAWS:
                        raise ValueError("final verifier did not return three pooled draws")
                    cache[key] = [scored(claims[row["id"]], evidence, g) for g in generations]
            record_row = {"id": row["id"], "pools": {policy: {
                "evidence_ids": [item["id"] for item in decision[policy]["selected_evidence"]],
                "draws": cache[tuple(item["id"] for item in decision[policy]["selected_evidence"])]}
                for policy in ("heuristic", "jev")},
                "actual_qwen_calls": MAX_DRAWS * len(cache)}
            pooled_rows.append(record_row)
            pooled_stream.write(json.dumps(record_row, ensure_ascii=False) + "\n")
            pooled_stream.flush()
            if index % 10 == 0:
                print(f"{index}/{len(source_rows)} pooled Qwen samples", flush=True)
    del verifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if trained_head:
        from agent.train.jev_head import JevHeadPolicy
        pooled_controller = JevHeadPolicy(args.model, head_path=trained_head)
    else:
        pooled_controller = load_controller(args.backend, args.model)
    rows = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, (row, decision, pooled) in enumerate(zip(source_rows, decisions, pooled_rows), 1):
            final = {}
            pool_decisions = {}
            for policy in ("heuristic", "jev"):
                pool_decisions[policy] = choose_pooled(
                    row["claim"], decision[policy]["selected_evidence"],
                    pooled["pools"][policy]["draws"],
                    model=pooled_controller if policy == "jev" else None,
                    threshold=args.threshold)
                final[policy] = pool_decisions[policy]["selected"]
            own_verdict = terminal_verdict(row, decision["jev"], pooled_controller,
                                           pooled_selected=final["jev"])
            result = {"id": row["id"], "gold": row["gold"],
                      "gold_in_context": row["gold_in_context"],
                      "heuristic": decision["heuristic"], "jev": decision["jev"],
                      "jev_terminal_verdict": own_verdict,
                      "heuristic_pooled_retry": pool_decisions["heuristic"],
                      "jev_pooled_retry": pool_decisions["jev"],
                      "heuristic_final": final["heuristic"], "jev_final": final["jev"],
                      "heuristic_evidence_diagnostic": evidence_diagnostic(row, decision["heuristic"], claims[row["id"]]),
                      "jev_evidence_diagnostic": evidence_diagnostic(row, decision["jev"], claims[row["id"]]),
                      "heuristic_action_diagnostic": action_diagnostic(row, decision["heuristic"]),
                      "jev_action_diagnostic": action_diagnostic(row, decision["jev"]),
                      "new_final_qwen_calls": pooled["actual_qwen_calls"]}
            rows.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 10 == 0:
                print(f"{index}/{len(source_rows)} final Qwen verifications", flush=True)
    metrics = summarize(rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rsi-jev", "openjev"), default="rsi-jev")
    parser.add_argument("--model", default=RSI_MODEL)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--trained-head", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
