# OpenJev verdict evaluation

This follow-up reads a completed [`evidence_batches`](../evidence_batches/) run for the **same claims and retrieved sentences**. [OpenJev-1.7B](https://huggingface.co/alanhuangya/OpenJev-1.7B) first makes a full-context verdict and a separate yes/no citation decision for every sentence. It then judges every 10-sentence batch, unions **its own** positive citation decisions without a 10-sentence cap, and makes one final verdict and citation pass. It saves per-option verdict probabilities and per-sentence citation probabilities.

OpenJev is a typed decision model. It does not **generate** a citation list, but one binary question per candidate sentence lets it select citations. The final top five positive sentences are scored with the same FEVER criterion as Qwen. This is an exploratory adaptation: its citation questions were not trained specifically for FEVER, and the `0.5` selection threshold is not calibrated here. Qwen's output supplies only the frozen claim/evidence context and a comparison result; Jev never reads Qwen's citations.

Install its official Python package in the SoC virtual environment once: `python -m pip install wev-ai`. The model weights download on first use; keep them outside Git. Then, from the repository root, after the 900-claim batch run finishes, inspect current SoC limits and submit:

```bash
JEV_SOURCE_RUN=agent/evaluation/results/evidence_batches/runs/sft-direct-from-base-qwen3p5-2b-5500-samples/direct-cap160/full900-run1 \
EVAL_METHOD=jev_decision RUN_NAME=full900-run1 \
  sbatch --partition=gpu-long --time=12:00:00 --export=ALL agent/evaluation/eval.sbatch
```

Set `JEV_SOURCE_RUN` to the actual completed result directory. The output goes to `agent/evaluation/results/jev_decision/runs/openjev-1p7b/full-vs-union/<run>/` by default. Use `RESULT_DIR` for a custom path and a fresh `RUN_NAME` for every attempt. `JEV_CITATION_THRESHOLD` defaults to `0.5`; `JEV_QUESTION_BATCH_SIZE` defaults to `20` and limits how many citation questions share an inference call. Read `metrics.json` for paired accuracy, FEVER score, evidence F1 and call counts; `predictions.jsonl` has per-claim probabilities. New Slurm logs use `slurm-agent-eval-<job-id>.out` in the submission directory.
