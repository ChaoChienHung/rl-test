# Exhaustive evidence batches

This method uses the same title-plus-BM25 evidence as the matched single-pass verifier. It splits the retrieved sentences into batches of 10, verifies **every** batch, collects valid citations, then verifies once more against the union of those cited sentences. The final union is **not capped at 10**; it can contain citations from every batch. NEI and confidence never stop the scan or contribute a vote. All prompts use deterministic decoding.

From the repository root on SoC, after preparing FEVER data, indexes and the default SFT 5500 checkpoint, submit the frozen 900-claim evaluation. The longer partition/time overrides are for this multi-pass run; inspect current SoC resource limits before submission:

```bash
EVAL_METHOD=evidence_batches RUN_NAME=full900-run1 \
  sbatch --partition=gpu-long --time=12:00:00 --export=ALL agent/evaluation/eval.sbatch
```

The default run uses all 900 frozen dev claims, a 10-sentence batch, direct prompt and 160 output tokens. Set `EVIDENCE_BATCH_SIZE`, `CHECKPOINTS`, `EVAL_MODEL`, `PROMPT_STYLE`, `MAX_NEW_TOKENS`, or `RESULT_DIR` to change those conditions. Use a new `RUN_NAME` each time. The job may run for several hours; compare Slurm elapsed time with the 12-hour limit and inspect the log as it progresses.

Results are saved under `runs/<eval-model>/<prompt>-cap<tokens>/<run>/`. `predictions.jsonl` contains the full-context baseline, every batch's parsed result and citations, the final union and final verdict. `metrics.json` compares paired accuracy, FEVER score, evidence F1, format rate, label recall, gains/losses and token/call cost. `manifest.json` records settings and revision. Slurm writes `slurm-agent-eval-<job-id>.out` to the submission directory.

This is an evaluation of a fixed heuristic on dev claims. If a batch cites nothing, the union may be empty; if a complete evidence set spans batches and individual batches cannot recognize it, this method may lose it. Inspect `empty_union_claims` and `gold_in_union` before interpreting score changes. No controller is trained on these claims.
