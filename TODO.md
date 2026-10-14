# TODO

Current work is verifier evaluation on the `rl` branch. These tasks replace the old proposal's dated week-by-week schedule; completed SFT and GRPO runs are documented in [factcheck/docs/sft-probe.md](factcheck/docs/sft-probe.md) and [factcheck/docs/grpo-results.md](factcheck/docs/grpo-results.md).

## P0 — comparable verifier evaluation (RL-1, AG-3)

- [ ] Freeze disjoint train, prompt-development, calibration and holdout claim IDs, then publish the split manifest and leakage check. The committed 5,895-row SFT train set and matched 1,200-row subset have zero ID overlap with the archived 900 dev claims; the remaining splits are not frozen.
- [ ] Run the direct-JSON and zero-shot evidence-chain scripts on the same FEVER dev IDs and title retrieval: 30-claim smoke tests, then 900 claims. Store predictions, metrics, logs and manifests under `factcheck/results/eval/`.
- [ ] Record per claim: gold, verdict, evidence IDs, retrieval condition, reasoning if present, stated confidence, verdict-token probability, tokens and latency. Check evidence citations by hand; a valid E-ID alone does not prove support.
- [ ] Report accuracy, FEVER score, evidence F1, label recall, Brier, ECE, risk–coverage and evidence-in-context strata with one evaluator. `factcheck/src/eval_saved_runs.py` already replays the archived outputs.
- [ ] Reproduce the SFT baseline with the original adapter or retrain it; the LoRA weights are not committed.

## P1 — prompt and inference controls (RL-1)

- [ ] Add a few-shot evidence-chain condition using examples outside evaluation IDs.
- [ ] At a fixed total token budget, compare one response with multiple samples / voting.
- [ ] Compare direct, short evidence reasoning, and explicit evidence chain on quality, token use, and citation faithfulness.

## P2 — confidence and reward (RL-2 with RL-1)

- [ ] Measure stated confidence and verdict-token probability separately; both predict whether the chosen verdict is correct. Inspect NEI and retrieval-miss subsets.
- [ ] Fit temperature scaling and a simple correctness predictor on a separate calibration split; evaluate once on holdout.
- [ ] Audit reward components, wrong-`SUPPORTED` harm, NEI behaviour, and confidence collapse. Compare reward ablations only with matched SFT and multiple seeds.

## P3 — train and integrate only after controls (RL-1/2/3, agent group)

- [ ] If evidence-chain prompting helps and rollouts show reward spread, run CoT + GRPO against matched SFT / CoT controls; report accuracy, evidence, calibration, variance and compute cost.
- [ ] Connect the verifier to the agent loop's `verify()` interface. Compare one search, fixed NEI retry and confidence-guided retry before training a controller.
- [ ] Train or evaluate a real-data controller only if simple fixed policies leave a measurable accuracy–cost gap. Check retrieval misses and transfer to SciFact / AVeriTeC.
- [ ] Prepare the team failure audit, shortcut checks, transfer results and final demo from the same evaluation definitions.
