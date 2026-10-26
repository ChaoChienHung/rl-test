"""Build high-margin, outcome-derived citation-controller SFT examples.

Gold and future draws are used only to derive training labels. Saved states contain
only the currently visible verifier output. Ambiguous decisions are omitted.
"""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from agent.data.build_evidence_batches import scored
from agent.evaluation.pooled_retry import selected_draw_index, state_for_pool
from agent.evaluation.results.rsi_jev_evidence_selection_baseline.eval_rsi_jev_evidence_selection_baseline import selection_state_for
from agent.evaluation.results.rsi_jev_retry_baseline.eval_rsi_jev_retry_baseline import evidence_for, heuristic_action
from agent.evaluation.results.sampling_retry.eval_sampling_retry import PROMPTS
from agent.train.replay_env import best_gold_set_evidence_credit, verdict_retry_feedback
from agent.train.train_jev_rl import validate_train_source
from verifier.model import LocalModel, gold_sets, render

DEFAULT_SPLIT = Path("agent/data/controller_split/sft3000-rl2500/manifest.json")


def citation_score(draw, gold, groups, chunk_ids):
    """Partial gold coverage within a chunk; no claim-level verdict label here."""
    if gold == "NOT_ENOUGH_INFO" or not any(group & chunk_ids for group in groups):
        return None
    if not draw.get("format_ok"):
        return -0.5
    cited = set(draw.get("evidence", [])) & chunk_ids
    return best_gold_set_evidence_credit(gold, cited, groups)[0]


def label_from_scores(current, future, draw_index, margin=0.2, retry_cost=0.05,
                      allow_bad_stop=False):
    """Hindsight comparison; skip close calls and unreliable bad STOP labels."""
    if current is None or not future:
        return None
    reachable = [(score - retry_cost * (offset + 1), offset + draw_index + 1)
                 for offset, score in enumerate(future) if score is not None]
    if not reachable:
        return None
    best_future = max(score for score, _ in reachable)
    if best_future >= current + margin:
        return "RETRY", best_future - current
    if current >= best_future + margin and (allow_bad_stop or current > 0):
        return "STOP", current - best_future
    return None


def chunk_examples(row, claim, margin):
    groups = [{f"{page}:{line}" for page, line in group} for group in gold_sets(claim)]
    output = []
    for ci, chunk in enumerate(row["chunks"]):
        chunk_ids = {item["id"] for item in chunk["evidence"]}
        scores = [citation_score(draw, row["gold"], groups, chunk_ids)
                  for draw in chunk["draws"]]
        for di in range(2):
            label = label_from_scores(scores[di], scores[di + 1:], di, margin)
            if label:
                output.append({"claim_id": row["id"], "stage": "chunk",
                               "chunk_index": ci, "draw_index": di,
                               "state": selection_state_for(row, ci, di),
                               "label": label[0], "margin": round(label[1], 6)})
    return output


def heuristic_pool(row):
    saved = set()
    for chunk in row["chunks"]:
        for di, draw in enumerate(chunk["draws"]):
            action = heuristic_action(draw, len(chunk["evidence"])) if di < 2 else "STOP"
            if action == "STOP":
                if draw.get("format_ok"):
                    saved.update(draw.get("evidence", []))
                break
    return [item for item in row["shown"] if item["id"] in saved]


def pooled_score(draw, gold_groups):
    credit, _ = best_gold_set_evidence_credit(
        draw["gold"], draw.get("evidence", []), gold_groups)
    return (1.0 if draw["correct"] else -1.0) + 0.5 * credit


def pooled_action_label(draws, scores, draw_index, margin):
    """Compare observed STOP with hindsight useful/futile RETRY outcomes."""
    current = draws[draw_index]
    if current["fever_score"]:
        return "STOP", margin
    stop_value = scores[draw_index] + verdict_retry_feedback("STOP", current, current)
    continuations = []
    if draw_index == 0:
        continuations.append((1, 1))  # RETRY once, then STOP on draw 2.
    continuations.append((selected_draw_index(draws, 3), 2 - draw_index))
    retry_value = max(
        scores[selected] + verdict_retry_feedback(
            "RETRY", current, draws[selected], useful_retry_credit=1.5,
            correct_retry_penalty=1.0) - 0.05 * extra_calls
        for selected, extra_calls in continuations)
    if retry_value >= stop_value + margin:
        return "RETRY", retry_value - stop_value
    if stop_value >= retry_value + margin and current["correct"]:
        return "STOP", stop_value - retry_value
    return None


def pooled_examples(row, claim, evidence, draws, margin):
    groups = [{f"{page}:{line}" for page, line in group} for group in gold_sets(claim)]
    scores = [pooled_score(draw, groups) for draw in draws]
    output = []
    for di in range(2):
        label = pooled_action_label(draws, scores, di, margin)
        if label:
            output.append({"claim_id": row["id"], "stage": "pooled",
                           "draw_index": di,
                           "state": state_for_pool(row["claim"], evidence, draws[di], di),
                           "label": label[0], "margin": round(label[1], 6)})
    return output


def run(args):
    split = json.loads(Path(args.split_manifest).read_text())
    allowed = set(split["sft_ids"])
    source_manifest, rows, claims = validate_train_source(Path(args.source_run), allowed)
    if len(rows) != 3000 or set(claims) != allowed:
        raise ValueError("SFT source must contain the fixed 3,000 IDs")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "episodes.jsonl"
    config_path = output / "config.json"
    config = {"source_run": args.source_run, "split_manifest": args.split_manifest,
              "margin": args.margin, "useful_retry_credit": 1.5,
              "correct_retry_penalty": 1.0,
              "source_commit": source_manifest.get("commit"),
              "data_builder": __name__}
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("existing output uses another SFT data configuration")
    else:
        if data_path.exists():
            raise ValueError("episodes exist without config")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    if (output / "manifest.json").exists():
        raise FileExistsError(f"SFT data already complete: {output}")
    done = set()
    if data_path.exists():
        for line in data_path.open(encoding="utf-8"):
            if line.strip():
                claim_id = json.loads(line)["claim_id"]
                if claim_id in done:
                    raise ValueError("duplicate claim in resumed SFT data")
                done.add(claim_id)
        if not done <= allowed:
            raise ValueError("resumed SFT data has an unexpected claim ID")
    system, thinking, _ = PROMPTS[source_manifest["prompt_style"]]
    verifier = LocalModel(source_manifest["model"],
                          adapter=source_manifest["adapter"], system_prompt=system,
                          max_new_tokens=source_manifest["max_new_tokens"],
                          enable_thinking=thinking, do_sample=True,
                          temperature=source_manifest.get("temperature", 0.7),
                          top_p=source_manifest.get("top_p", 0.9))
    with data_path.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            claim = claims[row["id"]]
            evidence = heuristic_pool(row)
            generations = verifier.run([render(row["claim"], evidence_for(evidence))] * 3,
                                       seed=source_manifest["generation_seed"] + 200000 + int(row["id"]))
            if len(generations) != 3:
                raise ValueError("expected three pooled Qwen draws")
            draws = [scored(claim, evidence_for(evidence), generation)
                     for generation in generations]
            examples = chunk_examples(row, claim, args.margin)
            examples.extend(pooled_examples(row, claim, evidence, draws, args.margin))
            stream.write(json.dumps({"claim_id": row["id"], "examples": examples,
                                     "pool_ids": [item["id"] for item in evidence],
                                     "pooled_draws": [{key: draw.get(key) for key in
                                        ("pred", "correct", "fever_score", "evidence_f1",
                                         "evidence", "token_conf", "format_ok")}
                                        for draw in draws]}, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 50 == 0:
                print(f"{index}/{len(rows)} SFT claim labels", flush=True)
    episodes = [json.loads(line) for line in data_path.open(encoding="utf-8") if line.strip()]
    if len(episodes) != 3000 or {episode["claim_id"] for episode in episodes} != allowed:
        raise ValueError("incomplete SFT label output")
    counts = Counter((example["stage"], example["label"])
                     for episode in episodes for example in episode["examples"])
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip() or None
    manifest = {**config, "commit": revision, "n_claims": 3000,
                "n_labeled_actions": sum(counts.values()),
                "label_counts": {f"{stage}:{label}": count for (stage, label), count in counts.items()},
                "method": "hindsight high-margin labels; heuristic only fixes the pooled citation set; future draws and FEVER gold never enter policy state"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT))
    parser.add_argument("--margin", type=float, default=0.2)
    args = parser.parse_args()
    if args.margin <= 0:
        parser.error("margin must be positive")
    run(args)


if __name__ == "__main__":
    main()
