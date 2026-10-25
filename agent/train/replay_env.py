"""Three-draw STOP/RETRY replay environment for train-only controller rollouts.

Each stop point must have a separately generated final verifier result using
only evidence observed through that point. The controller finishes or retries
the *same* input; all other chunks remain scheduled. The policy never sees
later draws, gold labels or counterfactual final outputs in its state.
"""

from dataclasses import dataclass


STOP = "STOP"
RETRY = "RETRY"
MAX_DRAWS = 3
POLICY_DRAW_FIELDS = (
    "draw_index", "pred", "format_ok", "format", "evidence",
    "token_conf", "stated_conf", "label_probs", "raw", "reasoning",
)


def policy_draw(draw):
    """Expose verifier observations without gold-derived scoring fields."""
    return {key: draw[key] for key in POLICY_DRAW_FIELDS if key in draw}


@dataclass(frozen=True)
class RewardWeights:
    fever: float = 1.0
    label: float = 0.0
    extra_draw: float = 0.05
    missing_gold: float = 0.25
    excess_evidence: float = 0.05
    evidence_slack: int = 3


def verifier_result_good(draw, min_evidence_f1=2 / 3):
    """Whether a verifier output warrants STOP under the current reward rule."""
    if not draw.get("correct"):
        return False
    if draw.get("gold") == "NOT_ENOUGH_INFO":
        return True
    f1 = draw.get("evidence_f1")
    return isinstance(f1, (int, float)) and f1 > min_evidence_f1


def stop_retry_feedback(action, draw):
    """Reward a good STOP or a retry of a bad output; penalize the reverse."""
    if action not in (STOP, RETRY):
        raise ValueError("action must be STOP or RETRY")
    good = verifier_result_good(draw)
    return 1.0 if (action == STOP) == good else -1.0


def best_gold_set_evidence_credit(gold, cited_ids, gold_groups,
                                  off_gold_weight=0.25, complete_bonus=0.25):
    """Partial evidence credit against one alternative FEVER gold set.

    Off-gold IDs receive a small penalty because FEVER annotations are not
    exhaustive; they are not asserted to be semantically false.
    """
    cited = set(cited_ids)
    if gold == "NOT_ENOUGH_INFO":
        return (complete_bonus if not cited else -off_gold_weight), {
            "scorable": True, "nei": True, "cited_count": len(cited)}
    groups = [set(group) for group in gold_groups if group]
    if not groups:
        return 0.0, {"scorable": False, "reason": "no_gold_evidence_set"}
    candidates = []
    for group in groups:
        overlap = len(cited & group)
        off_gold = len(cited - group)
        complete = group <= cited
        score = (overlap / len(group) + complete_bonus * complete
                 - off_gold_weight * off_gold / max(1, len(cited)))
        candidates.append((score, overlap, -off_gold, complete, len(group)))
    score, overlap, negative_off_gold, complete, gold_size = max(candidates)
    return score, {"scorable": True, "nei": False, "overlap": overlap,
                   "gold_size": gold_size, "off_gold_count": -negative_off_gold,
                   "complete_gold_set": complete, "cited_count": len(cited)}


def verdict_retry_feedback(action, current_draw, selected_draw,
                           useful_retry_credit=0.25, correct_retry_penalty=1.5):
    """Asymmetric verdict-only feedback; a futile RETRY earns no credit."""
    if action == STOP:
        return 1.0 if current_draw.get("correct") else -1.0
    if action != RETRY:
        raise ValueError("action must be STOP or RETRY")
    if current_draw.get("correct"):
        return -correct_retry_penalty
    return useful_retry_credit if selected_draw.get("correct") else -1.0


def evidence_penalties(saved, retrieved, gold_sets, weights=RewardWeights()):
    """Score saved evidence against any fully retrievable FEVER gold set.

    Gold evidence is a disjunction of complete sets. Retrieval misses and NEI
    rows have no controller-attributable evidence penalty.
    """
    saved_ids = set(saved)
    retrieved_ids = set(retrieved)
    reachable = [set(group) for group in gold_sets if group and set(group) <= retrieved_ids]
    if not reachable:
        return {"scorable": False, "missing": None, "excess": None,
                "penalty": 0.0, "reason": "no_complete_gold_set_retrieved_or_nei"}
    candidates = []
    for group in reachable:
        missing = len(group - saved_ids)
        excess = max(0, len(saved_ids) - len(group) - weights.evidence_slack)
        penalty = weights.missing_gold * missing + weights.excess_evidence * excess
        candidates.append((penalty, missing, excess, group))
    penalty, missing, excess, group = min(candidates,
                                          key=lambda item: (item[0], item[1], item[2], len(item[3])))
    return {"scorable": True, "missing": missing, "excess": excess,
            "penalty": penalty, "matched_gold_size": len(group),
            "saved_count": len(saved_ids), "reason": "best_retrievable_gold_set"}


class ClaimReplayEpisode:
    """Visit every chunk; STOP saves its latest citations, RETRY resamples it."""

    def __init__(self, claim, chunks, gold_sets, finalize, weights=RewardWeights()):
        if not claim or not chunks:
            raise ValueError("claim and at least one evidence chunk required")
        if any(len(chunk["draws"]) != MAX_DRAWS for chunk in chunks):
            raise ValueError("every chunk needs three precomputed draws")
        self.claim = claim
        self.chunks = chunks
        self.gold_sets = gold_sets
        self.finalize = finalize
        self.weights = weights
        self.chunk_index = 0
        self.draw_index = 0
        self.extra_draws = 0
        self.selected = []
        self.done = False

    def state(self):
        if self.done:
            raise RuntimeError("episode already finished")
        chunk = self.chunks[self.chunk_index]
        return {"claim": self.claim, "chunk_index": self.chunk_index,
                "chunk_count": len(self.chunks), "evidence": chunk["evidence"],
                "observed_draws": [policy_draw(draw)
                                   for draw in chunk["draws"][:self.draw_index + 1]],
                "draws_used_for_chunk": self.draw_index + 1,
                "remaining_retries": MAX_DRAWS - self.draw_index - 1,
                "actions": [STOP, RETRY] if self.draw_index < MAX_DRAWS - 1 else [STOP]}

    def step(self, action):
        if action not in self.state()["actions"]:
            raise ValueError(f"unavailable action {action!r}")
        if action == RETRY:
            self.draw_index += 1
            self.extra_draws += 1
            return {"done": False, "state": self.state(), "reward": 0.0}
        chunk = self.chunks[self.chunk_index]
        allowed = {item["id"] for item in chunk["evidence"]}
        cited = chunk["draws"][self.draw_index]["evidence"]
        if not set(cited) <= allowed:
            raise ValueError("selected draw cited evidence outside the current chunk")
        self.selected.extend(cited)
        if self.chunk_index + 1 < len(self.chunks):
            self.chunk_index += 1
            self.draw_index = 0
            return {"done": False, "state": self.state(), "reward": 0.0}
        self.done = True
        saved = list(dict.fromkeys(self.selected))
        retrieved = [item["id"] for part in self.chunks for item in part["evidence"]]
        final = self.finalize(saved)
        evidence = evidence_penalties(saved, retrieved, self.gold_sets, self.weights)
        reward = (self.weights.fever * float(final["fever_score"])
                  + self.weights.label * float(final["correct"])
                  - self.weights.extra_draw * self.extra_draws - evidence["penalty"])
        return {"done": True, "reward": reward, "final": final,
                "saved_evidence": saved, "extra_draws": self.extra_draws,
                "evidence_penalties": evidence}


class ReplayEpisode:
    def __init__(self, context, draws, final_at_stop, weights=RewardWeights()):
        if len(draws) != MAX_DRAWS or len(final_at_stop) != MAX_DRAWS:
            raise ValueError("exactly three draws and three counterfactual final results required")
        if not context.get("claim") or not context.get("evidence"):
            raise ValueError("context must include the claim and current fixed evidence input")
        self.context = context
        self.draws = draws
        self.final_at_stop = final_at_stop
        self.weights = weights
        self.index = 0
        self.done = False

    def state(self):
        if self.done:
            raise RuntimeError("episode already finished")
        return {"claim": self.context["claim"],
                "evidence": self.context["evidence"],
                "observed_draws": self.draws[:self.index + 1],
                "draws_used": self.index + 1,
                "remaining_retries": MAX_DRAWS - self.index - 1,
                "actions": [STOP, RETRY] if self.index < MAX_DRAWS - 1 else [STOP]}

    def step(self, action):
        if self.done:
            raise RuntimeError("episode already finished")
        if action not in self.state()["actions"]:
            raise ValueError(f"unavailable action {action!r} at draw {self.index + 1}")
        if action == RETRY:
            self.index += 1
            return {"done": False, "state": self.state(), "reward": 0.0}
        final = self.final_at_stop[self.index]
        reward = (self.weights.fever * float(final["fever_score"])
                  + self.weights.label * float(final["correct"])
                  - self.weights.extra_draw * self.index)
        self.done = True
        return {"done": True, "reward": reward,
                "final": final, "draws_used": self.index + 1}
