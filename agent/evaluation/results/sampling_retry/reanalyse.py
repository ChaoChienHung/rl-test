"""Add post-hoc controller diagnostics without changing archived predictions or metrics."""

import argparse
import json
from pathlib import Path

from .eval_sampling_retry import controller_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    predictions = args.run_dir / "predictions.jsonl"
    if not predictions.is_file() or not (args.run_dir / "metrics.json").is_file():
        parser.error("run_dir must contain completed predictions.jsonl and metrics.json")
    with predictions.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    result = {"source_run": str(args.run_dir), "n_claims": len(rows),
              "overall": controller_diagnostics(rows),
              "gold_in_context": controller_diagnostics(
                  [r for r in rows if r["gold_in_context"] is True]),
              "gold_not_in_context": controller_diagnostics(
                  [r for r in rows if r["gold_in_context"] is False]),
              "gold_nei": controller_diagnostics(
                  [r for r in rows if r["gold"] == "NOT_ENOUGH_INFO"])}
    output = args.run_dir / "controller_diagnostics.json"
    if output.exists():
        parser.error(f"{output} exists; refusing to overwrite historical analysis")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
