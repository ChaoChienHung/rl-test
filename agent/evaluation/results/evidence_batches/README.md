# Exhaustive evidence batches

This method uses the same title-plus-BM25 evidence as the verifier. It splits the retrieved sentences into batches of 10, verifies **every** batch, and collects valid citations in retrieval order. The union is **not capped at 10**; it can contain citations from every batch. NEI and confidence never stop the scan or contribute a vote. All prompts use deterministic decoding. With `EVIDENCE_BATCHES_ONLY=1`, it saves intermediate data for the Jev-versus-heuristic controller comparison; each controller then chooses a route before a new final verifier call. Historical full-evaluation runs without this flag remain supported and retain their saved results.

From the repository root on SoC, after preparing FEVER data, indexes and the default SFT 5500 checkpoint, submit the frozen 900-claim evaluation. The longer partition/time overrides are for this multi-pass run; inspect current SoC resource limits before submission:

```bash
RUN_NAME=full900-batches1 sbatch --export=ALL agent/eval.sbatch
```

The default `agent/eval.sbatch` run uses batch-only mode on all 900 frozen dev claims, 10-sentence batches, the direct prompt and 160 output tokens. Plain `sbatch agent/eval.sbatch` also works and names the run `job-<Slurm job ID>`. Set `EVIDENCE_BATCH_SIZE`, `CHECKPOINTS`, `EVAL_MODEL`, `PROMPT_STYLE`, `MAX_NEW_TOKENS`, or `RESULT_DIR` to change those conditions. Set `EVIDENCE_BATCHES_ONLY=0` only to reproduce the historical full evaluation. Use a new `RUN_NAME` each time. The job may run for several hours; compare Slurm elapsed time with the 12-hour limit and inspect the log as it progresses.

Batch-only data is saved under `agent/data/evidence_batches/runs/<eval-model>/<prompt>-cap<tokens>/<run>/` and is Git-ignored. `predictions.jsonl` contains the full retrieved context, every batch's parsed result and citations, and the citation union; it contains **no final verifier output**. `metrics.json` records batch counts, token cost and union coverage. `manifest.json` records settings and revision. The [controller evaluator](../jev_controller/README.md) consumes this directory. Historical full-evaluation runs still write under `agent/evaluation/results/evidence_batches/runs/`. Slurm writes `slurm-agent-eval-<job-id>.out` to the submission directory.

If a batch cites nothing, the union may be empty; if a complete evidence set spans batches and individual batches cannot recognize it, the union may omit necessary evidence. Inspect `empty_union_claims` and `gold_in_union` before interpreting controller scores. No controller is trained on these dev claims.
