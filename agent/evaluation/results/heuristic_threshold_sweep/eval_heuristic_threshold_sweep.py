"""Evaluate only 0.3 and 0.7 heuristic thresholds on fixed FEVER chunk draws."""

import argparse
import json
import subprocess
from pathlib import Path

from agent.data.build_evidence_batches import quality, scored
from agent.evaluation.pooled_retry import choose_pooled
from agent.evaluation.results.rsi_jev_evidence_selection_baseline.eval_rsi_jev_evidence_selection_baseline import (
    action_diagnostic, evidence_diagnostic, select,
)
from agent.evaluation.results.rsi_jev_retry_baseline.eval_rsi_jev_retry_baseline import (
    evidence_for, validate_source,
)
from agent.evaluation.results.sampling_retry.eval_sampling_retry import PROMPTS
from agent.train.replay_env import MAX_DRAWS
from verifier.model import LocalModel, render

THRESHOLDS = {"heuristic_0p3": 0.3, "heuristic_0p7": 0.7}


def generate_pools(row, decisions, claim, verifier, source_manifest):
    """Reuse Qwen draws only for identical evidence pools within one claim."""
    cache = {}
    pools = {}
    for name, decision in decisions.items():
        items = decision["selected_evidence"]
        key = tuple(item["id"] for item in items)
        if key not in cache:
            evidence = evidence_for(items)
            prompt = render(row["claim"], evidence)
            generations = verifier.run(
                [prompt] * MAX_DRAWS,
                seed=source_manifest["generation_seed"] + 100000 + int(row["id"]),
            )
            if len(generations) != MAX_DRAWS:
                raise ValueError("final verifier did not return three pooled draws")
            cache[key] = [scored(claim, evidence, generation) for generation in generations]
        pools[name] = {"evidence_ids": list(key), "draws": cache[key]}
    return {"id": row["id"], "pools": pools, "actual_qwen_calls": MAX_DRAWS * len(cache)}


def summarize(rows):
    policies = {}
    for name, threshold in THRESHOLDS.items():
        traces = [row[name] for row in rows]
        pooled = [row[f"{name}_pooled_retry"] for row in rows]
        predictions = [row[f"{name}_final"] for row in rows]
        policies[name] = {
            "threshold": threshold,
            "quality": quality(predictions),
            "gold_in_full_context": quality([row[f"{name}_final"] for row in rows
                                             if row["gold_in_context"] is True]),
            "gold_not_in_full_context": quality([row[f"{name}_final"] for row in rows
                                                 if row["gold_in_context"] is False]),
            "gold_nei": quality([row[f"{name}_final"] for row in rows
                                 if row["gold"] == "NOT_ENOUGH_INFO"]),
            "simulated_chunk_qwen_calls": sum(trace["chunk_verifier_calls"] for trace in traces),
            "simulated_pooled_qwen_calls": sum(item["simulated_qwen_calls"] for item in pooled),
            "simulated_total_qwen_calls": sum(trace["chunk_verifier_calls"] + item["simulated_qwen_calls"]
                                               for trace, item in zip(traces, pooled)),
            "selected_evidence_total": sum(len(trace["selected_evidence"]) for trace in traces),
            "empty_evidence_claims": sum(not trace["selected_evidence"] for trace in traces),
            "forced_chunk_stops": sum(sum(len(chunk["attempts"]) == MAX_DRAWS
                                          for chunk in trace["chunks"]) for trace in traces),
            "forced_pooled_stops": sum(item["simulated_qwen_calls"] == MAX_DRAWS for item in pooled),
            "pooled_early_stop_on_wrong": sum(item["early_stop_on_wrong"] for item in pooled),
            "pooled_forced_stop_on_wrong": sum(item["forced_stop_on_wrong"] for item in pooled),
        }
    return {
        "n_claims": len(rows),
        "policies": policies,
        "actual_new_pooled_qwen_calls": sum(row["new_final_qwen_calls"] for row in rows),
        "note": "Both thresholds read identical precomputed chunk draws. Distinct evidence pools receive fresh pooled Qwen draws; identical pools within a claim reuse draws. Simulated early stopping does not reduce the actual precomputation count. Gold is used only for scoring and post-hoc diagnostics.",
    }


def run(args):
    source = Path(args.source_run)
    source_manifest, source_rows, claims = validate_source(source)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"result directory exists: {output}")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip() or None
    output.mkdir(parents=True)
    manifest = {
        "evaluation_code": __name__,
        "split": "FEVER shared_task_dev; frozen validation IDs",
        "purpose": "evaluation_only_no_controller_training",
        "source_run": str(source),
        "source_commit": source_manifest.get("commit"),
        "commit": revision,
        "n_claims": len(source_rows),
        "thresholds": THRESHOLDS,
        "threshold_applies_to": ["chunk_stop_retry", "pooled_stop_retry"],
        "model": source_manifest["model"],
        "adapter": source_manifest["adapter"],
        "prompt_style": source_manifest["prompt_style"],
        "max_new_tokens": source_manifest["max_new_tokens"],
        "pooled_draws": MAX_DRAWS,
        "pooled_do_sample": True,
        "pooled_seed_offset": 100000,
        "pooled_temperature": source_manifest.get("temperature", 0.7),
        "pooled_top_p": source_manifest.get("top_p", 0.9),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    decisions = []
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(source_rows, 1):
            item = {"id": row["id"], **{name: select(row, threshold=threshold)
                                        for name, threshold in THRESHOLDS.items()}}
            decisions.append(item)
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            if index % 10 == 0:
                print(f"{index}/{len(source_rows)} heuristic selections", flush=True)

    system, thinking, _ = PROMPTS[source_manifest["prompt_style"]]
    verifier = LocalModel(source_manifest["model"],
                          batch_size=source_manifest["model_batch_size"],
                          adapter=source_manifest["adapter"], system_prompt=system,
                          max_new_tokens=source_manifest["max_new_tokens"],
                          enable_thinking=thinking, do_sample=True,
                          temperature=source_manifest.get("temperature", 0.7),
                          top_p=source_manifest.get("top_p", 0.9))
    rows = []
    with (output / "pooled_samples.jsonl").open("w", encoding="utf-8") as sample_stream, \
         (output / "predictions.jsonl").open("w", encoding="utf-8") as prediction_stream:
        for index, (row, decision) in enumerate(zip(source_rows, decisions), 1):
            pooled = generate_pools(row, {name: decision[name] for name in THRESHOLDS},
                                    claims[row["id"]], verifier, source_manifest)
            sample_stream.write(json.dumps(pooled, ensure_ascii=False) + "\n")
            result = {"id": row["id"], "gold": row["gold"],
                      "gold_in_context": row["gold_in_context"],
                      "new_final_qwen_calls": pooled["actual_qwen_calls"]}
            for name, threshold in THRESHOLDS.items():
                selection = decision[name]
                chosen = choose_pooled(row["claim"], selection["selected_evidence"],
                                       pooled["pools"][name]["draws"], threshold=threshold)
                result[name] = selection
                result[f"{name}_pooled_retry"] = chosen
                result[f"{name}_final"] = chosen["selected"]
                result[f"{name}_evidence_diagnostic"] = evidence_diagnostic(
                    row, selection, claims[row["id"]])
                result[f"{name}_action_diagnostic"] = action_diagnostic(row, selection)
            rows.append(result)
            prediction_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            sample_stream.flush()
            prediction_stream.flush()
            if index % 10 == 0:
                print(f"{index}/{len(source_rows)} pooled Qwen samples", flush=True)
    metrics = summarize(rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
