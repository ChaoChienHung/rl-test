"""Offline STOP/RETRY on three fresh final-verifier draws over pooled evidence.

The chunk source is never changed. A caller generates all three pooled draws
before controller decisions so heuristic and Jev see matched candidates.
"""

import json
import math
from collections import Counter

from agent.train.replay_env import MAX_DRAWS, RETRY, STOP

FINAL_QUESTION = {"action": {"type": "choice",
    "instructions": ("The current verifier output was generated from the final pooled evidence "
                     "saved across all chunks. Decide whether to STOP and use this output, "
                     "or RETRY the exact same claim and pooled evidence for another sample. "
                     "RETRY does not retrieve or add evidence and costs another verifier call. "
                     "Judge the current verdict and its citations, using only the current "
                     "output and cited sentence text. At most two retries are available."),
    "criteria": {STOP: "Use the current verifier output as the final prediction.",
                 RETRY: "Sample the same pooled evidence again before deciding."}}}


def state_for_pool(claim, evidence, draw, draw_index):
    by_id = {item["id"]: item for item in evidence}
    cited = [{"id": item_id, "text": by_id[item_id]["text"]}
             for item_id in dict.fromkeys(draw.get("evidence", [])) if item_id in by_id]
    return json.dumps({"claim": claim, "pooled_evidence_count": len(evidence),
                       "draw_index": draw_index,
                       "remaining_retries": MAX_DRAWS - draw_index - 1,
                       "verifier_output": {"verdict": draw.get("pred"),
                                           "format_ok": draw.get("format_ok"),
                                           "token_conf": draw.get("token_conf"),
                                           "stated_conf": draw.get("stated_conf"),
                                           "cited_evidence": cited}},
                      ensure_ascii=False, separators=(",", ":"))


def selected_draw_index(draws, observed_count):
    """Early STOP uses the current draw; the third draw triggers voting."""
    if observed_count not in (1, 2, MAX_DRAWS) or len(draws) < observed_count:
        raise ValueError("invalid pooled draw count")
    if observed_count < MAX_DRAWS:
        return observed_count - 1
    labels = [draw.get("pred") for draw in draws[:observed_count]]
    votes = Counter(label for label in labels if label in
                    ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"))
    def confidence(index):
        value = draws[index].get("token_conf")
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else float("-inf")
    if votes:
        winning = max(votes, key=lambda label: (votes[label],
                                                 max(confidence(i) for i in range(observed_count)
                                                     if labels[i] == label)))
        return max((i for i in range(observed_count) if labels[i] == winning), key=confidence)
    return max(range(observed_count), key=confidence)


def choose_pooled(claim, evidence, draws, model=None, threshold=0.5):
    if len(draws) != MAX_DRAWS:
        raise ValueError("pooled evidence requires exactly three fresh draws")
    decisions = []
    for index, draw in enumerate(draws):
        if index == MAX_DRAWS - 1:
            action = STOP
            reason = "retry_limit"
            answer = None
        elif model is None:
            from agent.evaluation.results.rsi_jev_retry_baseline.eval_rsi_jev_retry_baseline import heuristic_action
            action = heuristic_action(draw, len(evidence), threshold)
            reason = "fixed_confidence_rule"
            answer = None
        else:
            answer = model.predict(state=state_for_pool(claim, evidence, draw, index),
                                   questions=FINAL_QUESTION)["answers"]["action"]
            action = answer["choice"]
            reason = "model_choice"
            if action not in (STOP, RETRY):
                raise ValueError(f"invalid pooled-evidence action: {action!r}")
        decisions.append({"draw_index": index, "action": action, "reason": reason,
                          "confidence": answer.get("confidence") if answer else None,
                          "probabilities": answer.get("probabilities") if answer else None})
        if action == STOP:
            break
    observed_count = len(decisions)
    selected = selected_draw_index(draws, observed_count)
    return {"decisions": decisions, "selected_draw_index": selected,
            "stopped_at_draw_index": observed_count - 1,
            "selected": draws[selected], "simulated_qwen_calls": len(decisions),
            "controller_calls": sum(x["reason"] == "model_choice" for x in decisions),
            "retry_after_correct_draws": sum(bool(draws[x["draw_index"]].get("correct"))
                                             for x in decisions[:-1]),
            "early_stop_on_wrong": int(observed_count < MAX_DRAWS and
                                       not draws[selected].get("correct")),
            "forced_stop_on_wrong": int(observed_count == MAX_DRAWS and
                                        not draws[selected].get("correct")),
            "any_correct_candidate": any(draw.get("correct") for draw in draws),
            "aggregation_rule": ("three_draw_majority_then_confidence"
                                 if observed_count == MAX_DRAWS
                                 else "current_output_on_early_stop")}
