# Jev-style controller evaluation

This evaluator uses [RSI-Jev v2.1 2B](https://huggingface.co/shgao/rsi-jev-v2.1-qwen3.5-2b) **as a controller** by default, with Qwen kept as the verifier. [OpenJev-1.7B](https://huggingface.co/alanhuangya/OpenJev-1.7B) remains an optional controller comparison. It reads a completed batch-only Qwen run from `agent/data/evidence_batches/`. For each claim, the controller sees the claim, the number of retrieved sentences, each batch verifier's verdict, format status, confidence and cited sentence IDs, plus the size of their citation union. It does **not** see the gold label, raw evidence sentences, or Qwen final outputs.

Jev chooses one route for the **final Qwen verifier**: `CITED_UNION` (the union of sentences cited by the batches) or `FULL_CONTEXT` (all originally retrieved sentences). The main comparison is **RSI-Jev controller versus a fixed heuristic controller** on the same 900 claims and saved Qwen batch outputs. The heuristic chooses `FULL_CONTEXT` if the citation union is empty, any batch output is invalid or NEI, batch verdicts disagree, or any verdict-token confidence is missing or below `0.5`; otherwise it chooses `CITED_UNION`. This threshold is fixed before inspecting this controller evaluation and is not a calibrated probability claim. Both policies have the same two actions and access to the same batch summaries; only Jev uses a model call. **Both routes are chosen before new final Qwen inference.** If they agree, one final Qwen call is shared; if they disagree, each route gets its own final call. The historical batch run's baseline and union-final outputs are never reused. Report paired accuracy, FEVER score, evidence F1, per-label recall, route agreement and call cost. Qwen still supplies verdicts and cited evidence. RSI-Jev 2B matches the verifier's nominal model size, but they have different training and roles; this is not a matched-training comparison.

Install the optional RSI-Jev runtime in the SoC `.venv` once, from the repository root:

```bash
source .venv/bin/activate
mkdir -p "$HOME/.pip-tmp"
export TMPDIR="$HOME/.pip-tmp"
python -m pip install --no-cache-dir -r agent/requirements.txt
python -c 'import rsijev; print("RSI-Jev: OK")'
```

After the 900-claim Qwen batch run completes, submit from the repository root:

```bash
JEV_SOURCE_RUN=agent/data/evidence_batches/sft-direct-from-base-qwen3p5-2b-5500-samples/direct-cap160/deterministic-1-draw-per-chunk-900-claims \
EVAL_METHOD=rsi_jev_controller_baseline RUN_NAME=full900-run1 \
  sbatch --partition=gpu-long --time=12:00:00 --export=ALL agent/eval.sbatch
```

Read `agent/evaluation/results/rsi-jev-controller-baseline/runs/rsi-jev-v2p1-qwen3p5-2b/route-full-vs-union/<run>/metrics.json` and `predictions.jsonl`. The main result is `overall.jev_controller` versus `overall.heuristic_controller`, with paired gain/loss counts beside them. Each prediction records both route choices and **new** final Qwen results. `new_final_qwen_calls` counts actual final calls in this job; each policy incurs one final call per claim in deployment, after the shared batch calls. Use a fresh `RUN_NAME` or `RESULT_DIR` for another run. Set `JEV_MODEL` to override the model ID within the selected runtime. To run the OpenJev comparison instead, set `JEV_BACKEND=openjev` and install `wev-ai`; it writes under `runs/openjev-1p7b/`. This zero-shot controller evaluation uses frozen FEVER dev claims only; train any controller on separate train-claim rollouts. It does not establish a learning or RL gain. The package is not installed or model loaded by the repository tests, so test one controller request on SoC before interpreting a full run.
