# TODO

Current work is verifier evaluation on the `rl` branch. These tasks replace the old proposal's dated week-by-week schedule; completed SFT and GRPO runs are documented in [docs/sft-probe.md](docs/sft-probe.md) and [docs/grpo-results.md](docs/grpo-results.md).

## Next experiment sequence

1. [ ] Once the no-CoT 2B SFT run completes, use **the same checkpoint** for direct (`THINKING=off`) and evidence-chain (`THINKING=on`) evaluation on the same 900 dev IDs, retrieval setup and evaluator. Compare both with the base-model runs, including accuracy, FEVER score, label recall, citation quality, confidence and token cost.
2. [ ] Start a **new GRPO run from that SFT checkpoint**, retaining the checkpoint path and matched no-GRPO SFT results as the control. First inspect sampled rollouts and reward spread; for CoT GRPO, update the current 96-token completion cap, prompt format and reward checks before training, then compare on the same held-out evaluation.

CoT SFT data preparation can proceed in parallel with step 1; it is a separate experiment and should not change the existing no-CoT checkpoint or its evaluation.

## P0 — comparable verifier evaluation (RL-1, AG-3)

- [ ] Train the no-CoT Qwen3.5-2B SFT adapter from the committed 5,895 examples (seed 0); the historical LoRA weights are not committed. Record job ID, runtime, peak GPU memory, and checkpoint path.
- [ ] Freeze disjoint train, prompt-development, calibration and holdout claim IDs, then publish the split manifest and leakage check. The committed 5,895-row SFT train set and matched 1,200-row subset have zero ID overlap with the archived 900 dev claims; the remaining splits are not frozen.
- [ ] Run direct JSON (`THINKING=off`) and zero-shot evidence-chain (`THINKING=on`) evaluation on the same FEVER dev IDs and title retrieval: 30-claim smoke tests, then 900 claims. Submit `verifier/evaluation/run.sbatch`; each run stores predictions, metrics, manifest and logs under `verifier/evaluation/results/<condition>/<run>/`.
- [ ] Record per claim: gold, verdict, evidence IDs, retrieval condition, reasoning if present, stated confidence, verdict-token probability, tokens and latency. Check evidence citations by hand; a valid E-ID alone does not prove support.
- [ ] Report accuracy, FEVER score, evidence F1, label recall, Brier, ECE, risk–coverage and evidence-in-context strata with one evaluator. `verifier/evaluation/score_predictions.py` already replays the archived outputs.

## P1 — prompt and inference controls (RL-1)

- [ ] Add a few-shot evidence-chain condition using examples outside evaluation IDs.
- [x] Implement SoCLaaS/Gemini CoT SFT teacher-data preparation from the existing 5,895 FEVER **train** oracle prompts, with a 100-per-class pilot, resumable requests and automated validation (`python -m utils.build_cot_sft_data`). The `qwen3.8:27b` pilot received all 300 responses; 227 passed and 73 were rejected (46 citation mismatches, 18 gold not justified by context, 8 wrong relations, 1 duplicate/too-many citations). Token usage: 442,495 prompt + 29,377 completion. Never send dev/evaluation claims to the teacher for training data.
- [ ] Use the pilot to assess explanation quality without per-row manual review: inspect automated rejection categories, add an independent evidence-faithfulness judge or targeted challenge set, and record its limits. Valid E IDs and a teacher self-check do not establish semantic faithfulness. The accepted CoT target omits `confidence`; decide how to train and evaluate this output format before using it as a checkpoint.
- [x] Add a separate CoT SFT training configuration and matching confidence-free evaluation prompt/format check. The 227 accepted teacher examples pass local preflight; training remains pending until the dataset and code are synced to SoC.
- [ ] Run the 227-row CoT SFT pilot and matched evaluation; track the NEI imbalance (51 NEI versus 90 SUPPORTED and 86 REFUTED). Do not interpret a comparison against the 5,895-row no-CoT SFT as an isolated CoT effect; make a no-CoT control on the same 227 claim IDs before claiming a training-method gain.
- [ ] After the oracle-prompt pilot, add **train-only** title-retrieval contexts so CoT SFT sees the evidence distribution used at evaluation. For non-NEI claims whose gold evidence is absent, do not teach the original FEVER label as though it were supported by the shown text; exclude them from ordinary CoT SFT or make a separately labeled contextual-abstention set. Keep the original FEVER gold label for evaluation and provenance.
- [ ] Once teacher responses pass validation, run the separate CoT SFT pilot adapter and evaluate it on the same held-out IDs as the no-CoT SFT checkpoint. Record accuracy, FEVER score, citation quality, output length and token probability; stated confidence is absent in this pilot.
- [ ] At a fixed total token budget, compare one response with multiple samples / voting.
- [ ] Compare direct, short evidence reasoning, and explicit evidence chain on quality, token use, and citation faithfulness.

## P2 — confidence and reward (RL-2 with RL-1)

- [ ] Measure stated confidence and verdict-token probability separately; both predict whether the chosen verdict is correct. Inspect NEI and retrieval-miss subsets.
- [ ] Fit temperature scaling and a simple correctness predictor on a separate calibration split; evaluate once on holdout.
- [ ] Audit reward components, wrong-`SUPPORTED` harm, NEI behaviour, and confidence collapse. Compare reward ablations only with matched SFT and multiple seeds.

## P3 — train and integrate only after controls (RL-1/2/3, agent group)

- [ ] After the SFT-started GRPO pilot, compare it against the same SFT checkpoint with matched prompts, IDs and evaluator. For CoT GRPO, measure reasoning/citation quality, reward spread, accuracy, evidence, calibration, variance and compute cost; expand only if the pilot is informative.
- [ ] Connect the verifier to the agent loop's `verify()` interface. Compare one search, fixed NEI retry and confidence-guided retry before training a controller.
- [ ] Train or evaluate a real-data controller only if simple fixed policies leave a measurable accuracy–cost gap. Check retrieval misses and transfer to SciFact / AVeriTeC.
- [ ] Prepare the team failure audit, shortcut checks, transfer results and final demo from the same evaluation definitions.
