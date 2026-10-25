# Exhaustive evidence batches

This method uses the same title-plus-BM25 evidence as the verifier. It splits the retrieved sentences into batches of 10, verifies **every** batch, and collects valid citations in retrieval order. The union is **not capped at 10**; it can contain citations from every batch. NEI and confidence never stop the scan or contribute a vote. All prompts use deterministic decoding. With `EVIDENCE_BATCHES_ONLY=1`, it saves intermediate data for the Jev-versus-heuristic controller comparison; each controller then chooses a route before a new final verifier call. Historical full-evaluation runs without this flag remain supported and retain their saved results.

From the repository root on SoC, after preparing FEVER data, indexes and the default SFT 5500 checkpoint, submit the frozen 900-claim evaluation. The longer partition/time overrides are for this multi-pass run; inspect current SoC resource limits before submission:

```bash
RUN_NAME=deterministic-1-draw-per-chunk-900-claims-run2 sbatch --export=ALL agent/eval.sbatch
```

The default `agent/eval.sbatch` run uses batch-only mode on all 900 frozen dev claims, 10-sentence batches, the direct prompt and 160 output tokens. Plain `sbatch agent/eval.sbatch` also works and names the run `job-<Slurm job ID>`. Set `EVIDENCE_BATCH_SIZE`, `CHECKPOINTS`, `EVAL_MODEL`, `PROMPT_STYLE`, `MAX_NEW_TOKENS`, or `RESULT_DIR` to change those conditions. Set `EVIDENCE_BATCHES_ONLY=0` only to reproduce the historical full evaluation. Use a new `RUN_NAME` each time. The job may run for several hours; compare Slurm elapsed time with the 12-hour limit and inspect the log as it progresses.

Batch-only data is saved under `agent/data/evidence_batches/<eval-model>/<prompt>-cap<tokens>/<run>/` and is Git-ignored. The completed `full900-batches1` run used an older launcher and was originally under `agent/data/evidence_batches/runs/`; its organized name is `deterministic-1-draw-per-chunk-900-claims` in the current layout. `git pull` does not transfer it. `predictions.jsonl` contains the full retrieved context, every batch's parsed result and citations, and the citation union; it contains **no final verifier output**. `metrics.json` records batch counts, token cost and union coverage. `manifest.json` records settings and revision. The [controller evaluator](../evaluation/results/rsi_jev_controller_baseline/README.md) consumes this directory. Historical full-evaluation runs still write under `agent/evaluation/results/evidence_batches/runs/`. Slurm writes `slurm-agent-eval-<job-id>.out` to the submission directory.

If a batch cites nothing, the union may be empty; if a complete evidence set spans batches and individual batches cannot recognize it, the union may omit necessary evidence. Inspect `empty_union_claims` and `gold_in_union` before interpreting controller scores. No controller is trained on these dev claims.

## Repeated samples on each fixed chunk

For the STOP/RETRY comparison, generate three samples from the **same prompt for each 10-sentence chunk**. All retrieval chunks are processed regardless of any sampled verdict. Run from the repository root on SoC after checking partition limits:

```bash
EVAL_METHOD=evidence_batches EVIDENCE_BATCHES_ONLY=1 EVIDENCE_DRAWS=3 \
EVIDENCE_BATCH_SIZE=10 RUN_NAME=stochastic-3-draws-per-chunk-900-claims \
  sbatch --time=1-00:00:00 --export=ALL agent/eval.sbatch
```

`EVIDENCE_DRAWS` defaults to 1 to preserve the historical deterministic producer. Values above 1 turn on sampling, with `TEMPERATURE=0.7` and `TOP_P=0.9` by default. `RUN_NAME` must be new. Multi-draw `predictions.jsonl` uses `chunks[].evidence` for the 10 fixed global sentence IDs and texts, and `chunks[].draws` for the independently generated scored outputs. The same rendered prompt is repeated within a chunk, with one recorded seed per chunk. `chunks[].draws` are candidates only; there is no chosen sample, citation union, controller decision or final Qwen verification in this file. The separate `sampling_retry` dataset samples the full context rather than each chunk and cannot stand in for this one.

`metrics.json` records the number of chunks and actual verifier calls, total input/output tokens, and per-chunk verdict/evidence disagreement. On the earlier 900-claim retrieval, three draws would produce 3,169 × 3 = 9,507 generated outputs; verify the actual count after the new job. Increase to `EVIDENCE_DRAWS=5` only in a separate run if a five-attempt policy is being studied. This frozen dev dataset must not be used to train the controller.

Once the three-draw run finishes, compare fixed heuristic and zero-shot RSI-Jev STOP/RETRY decisions with the [matched controller evaluator](../evaluation/results/rsi_jev_retry_baseline/README.md). STOP saves the latest draw's valid citations for a final evidence pool; RETRY resamples only this unchanged chunk. Every chunk is still read. The Jev prompt, actions, final Qwen verification and result layout are documented there.

The [evidence-selection variant](../evaluation/results/rsi_jev_evidence_selection_baseline/README.md) reads the **same** completed chunk source. Its controller chooses STOP/RETRY for the current verifier output. RETRY discards that output and citations; on STOP, it can choose KEEP/DROP for cited sentences. It then creates a **separate** `pooled_samples.jsonl` with three new stochastic verifier draws on the saved evidence from each chunk's STOP draw. Early pooled STOP uses the current output; at the forced third draw both policies choose the majority verdict, using token confidence to break ties. Jev's own terminal verdict is diagnostic only. The old `chunks[].draws` remain untouched.
