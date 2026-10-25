# Heuristic threshold sweep on pooled evidence

This validation-only evaluator compares confidence thresholds **0.3** and **0.7** without loading RSI-Jev. Both arms use the same completed 900-claim, three-draw-per-10-sentence source as the evidence-selection evaluation. A threshold applies to STOP/RETRY at both the chunk and final pooled-verifier stages. STOP saves all valid citations from that draw; RETRY discards them. Every chunk is visited. After the saved citations are pooled, Qwen produces three fresh stochastic draws per distinct evidence set. The controller can stop early on draw 1 or 2; a forced third draw uses majority verdict, then confidence to break ties. No gold labels are shown to the policy.

The two arms can produce different pools, so their pooled Qwen outputs are generated separately. When a claim has identical ordered evidence IDs for both thresholds, the same three Qwen outputs are reused. The per-claim seed and verifier configuration match the main evidence-selection evaluator; `manifest.json` records the source and current code revision. The source chunk data is never modified.

From the SoC repository root, after pulling the updated `rl` branch and checking current GPU partition limits:

```bash
JEV_SOURCE_RUN=agent/data/evidence_batches/sft-direct-from-base-qwen3p5-2b-5500-samples/direct-cap160/stochastic-3-draws-per-chunk-900-claims \
EVAL_METHOD=heuristic_threshold_sweep RUN_NAME=thresholds-0p3-0p7-run1 \
  sbatch --time=1-00:00:00 --export=ALL agent/eval.sbatch
```

The default output is `agent/evaluation/results/heuristic-threshold-sweep/runs/<source-model>/<source-condition>/<run>/`. Use a new `RUN_NAME` or set `RESULT_DIR` to a new directory. `decisions.jsonl` records chunk selections; `pooled_samples.jsonl` has the fresh Qwen candidates for each threshold; `predictions.jsonl` contains selected outputs and post-hoc diagnostics; `metrics.json` compares quality, simulated calls and actual pooled Qwen calls. Compare the 0.5 heuristic from the full evidence-selection run only when its source manifest, model, adapter, prompt, seed and evaluator settings match.
