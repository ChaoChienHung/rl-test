# TODO

Current work is verifier evaluation on the `rl` branch. These tasks replace the old proposal's dated week-by-week schedule; completed SFT and GRPO runs are documented in [docs/sft-probe.md](docs/sft-probe.md) and [docs/grpo-results.md](docs/grpo-results.md).

## Next experiment sequence

1. [x] Compare direct (`THINKING=off`) and evidence-chain (`THINKING=on`) prompts using the same no-CoT 2B SFT `run2` checkpoint, 900 dev IDs, title retrieval and evaluator. Direct: 667/900 accuracy, 593/900 FEVER; CoT: 643/900 accuracy, 562/900 FEVER. CoT has 48 format failures, 44 at its 512-token cap; see `INSIGHT.md` for label and confidence details.

2. [x] Retest **both the base 2B model and the no-CoT SFT `run2` checkpoint** on the same 900 IDs with `PROMPT_STYLE=cot_brief` and the same 512-token cap. Base: 556/900 accuracy, 437/900 FEVER, 898/900 valid format, 1 output at 512 tokens. SFT: 671/900 accuracy, 575/900 FEVER, 891/900 valid format, 9 outputs at 512 tokens. Brief CoT largely fixes truncation but both FEVER scores are below the corresponding best prompt; SFT NEI recall remains 57.0% versus 70.3% for direct. The prompt was tuned after inspecting these 900 dev results, so validate promising gains on an untouched holdout.

3. [x] Complete the six Base/SFT × direct/long CoT/brief CoT evaluations with the same 900 IDs, titles retrieval and 1024-token output cap. Base accuracy: 47.6% / 62.8% / 61.8%; SFT accuracy: 74.6% / 71.4% / 74.6%. Base long CoT has the best Base FEVER score (52.8%); SFT direct has the best SFT FEVER score (66.3%). SFT long CoT still has 44 capped outputs at 1024. See `INSIGHT.md`; validate on an untouched holdout because brief CoT was tuned on these dev claims.

4. [x] Complete the fixed 4,695/1,200 split comparison on SoC. Both SFT starting checkpoints and both Native-Thinking CoT GRPO jobs (Seen Arm from 5895 and Unseen Arm from 4695) have completed training and evaluation on SoC.
   - **Seen Arm CoT GRPO** (`cot-grpo1-seen-cot-grpo`): **74.1% accuracy**, **67.1% FEVER score**, **72.2% non-NEI evidence F1** under `cot_native` (cap 1024), beating the continued SFT control (60.2% evidence F1) by +12.0%.
   - **Unseen Arm CoT GRPO** (`cot-grpo1-unseen-cot-grpo`): **76.2% accuracy**, **67.2% FEVER score**, **73.7% non-NEI evidence F1** (all-time repository records!), establishing that CoT GRPO generalizes out-of-distribution without reliance on seen claims.

5. [ ] Audit the saved 1024-token **fail cases before changing prompts, decoding, or training**. Group Base direct's 130 invalid outputs by JSON parse, verdict, evidence-ID and confidence failure, and separate its 3 capped cases. Inspect SFT long CoT's 48 invalid outputs (44 capped) and brief CoT's 9 capped failures for repetition, missing final JSON and excessive reasoning. Compare matched claim IDs where direct versus long/brief CoT flip verdict or FEVER outcome; split by gold label and gold evidence in context. Save counts plus a small ID-based example list, then choose a targeted fix and re-evaluate on a fresh holdout. Do not tune again on these 900 dev claims and call the result independent validation.

6. [x] Complete the **5,500 Direct vs Native-Thinking CoT SFT comparison**. Both models have completed training and evaluation on SoC.
   - **Direct SFT 5500**: **76.1% accuracy**, **66.4% FEVER score** (direct-cap1024), and **69.9% evidence F1** (cot-brief-cap1024), decisively outperforming the legacy 5,895 baseline (+1.6% accuracy, +7.3% REFUTED recall).
   - **Native-Thinking CoT SFT 5500**: **73.4% accuracy**, **65.0% FEVER score**, **61.9% evidence F1** (cot-native-cap1024). Demonstrates that supervised teacher imitation alone underperforms RL-rewarded reasoning, highlighting the necessity of Stage 2 GRPO.

7. [x] Build **Dual-Path 4,500 SFT / 1,000 GRPO datasets** from clean 5,500 claims (zero dev overlap):
   - **Path 1 (Standard JSON)**: 4,500 balanced SFT (1,500 per class) + 1,000 disjoint GRPO holdout (`fever-oracle-disjoint-4500` & `fever-oracle-matched-grpo-1000`). Preserves backward compatibility and traceability.
   - **Path 2 (Native Tag Format)**: Parallel datasets with `<evidence>...</evidence>\n<verdict>...</verdict>` (evidence-first causal attention) and `<think>...</think>` tags (`fever-oracle-tagged-4500`, `fever-oracle-tagged-cot-4500` & `fever-oracle-tagged-grpo-1000`).
   - Both splits generated via `utils/build_4500_1000_splits.py` with signed split manifests and 0 dev overlap.

8. [x] **Universal Parsing, RLVR Rewards & DPO data audit**:
   - Upgraded `parse()` in `verifier/model.py` to seamlessly handle both JSON objects and native XML tags, with dual `VERDICT_PREFIX` support for token probabilities.
   - Formulated RLVR process rewards (index grounding $+0.1$, non-existent index penalty $-0.5$, parsimony penalties) and advanced RL algorithms (**Dr. GRPO** advantage without $\sigma_R$ division and **DAPO** dynamic sampling / decoupled clip-higher) in `INSIGHT.md` (Discussions 7 & 10).
   - The earlier 395 teacher-rejection DPO pairs were invalid: their chosen and rejected responses reused the same rejected reasoning while changing only the final answer. Removed those generated files and replaced `utils/build_dpo_pairs.py` with a builder that requires two real CoT rollouts for the same training claim and exact prompt. No valid DPO training set has been produced yet.

9. [x] **Train Dual-Path 4,500 SFT Checkpoints & Base Model Tag Baselines**:
   - Evaluated Base Qwen3.5-2B under Native Tag formats: `direct_tag` (Acc 40.7%, Evidence F1 0.513), `cot_brief_tag` (Acc 55.3%, Evidence F1 0.581), `cot_chain_tag` (Acc 55.8%, Evidence F1 0.511). Establishes format rate surge (98.0%) alongside confirmation-bias analysis in `INSIGHT.md` (Discussion 14).
   - Completed Path 1 (JSON) 4,500 SFT training (`run1-json-sft`, 5,268s, 5.57 GB, loss $\to$ 0.024).
   - Completed Path 2 (Native Tag) 4,500 SFT training (`run1-tag-sft`, 6,636s, 5.57 GB, loss $\to$ 0.029).
   - Created multi-condition validation array `verifier/evaluation/run_eval_4500_array.sbatch` and launcher `verifier/evaluation/submit_eval_4500.sh`.

10. [x] **Complete Dual-Path 4,500 SFT & 1,000-sample CoT GRPO Benchmarks (Tasks 0-6)**:
    - **JSON SFT 4500 Direct**: **76.7% Acc** (All-time SFT Record!), **67.6% FEVER**, 64.9% Ev F1, **Token ECE 0.0503**.
    - **JSON SFT 4500 Brief CoT**: 73.9% Acc, 64.3% FEVER, 65.7% Ev F1, Token ECE 0.2238.
    - **Tag SFT 4500 Direct**: 75.6% Acc, **68.1% FEVER** (All-time SFT Record!), 63.6% Ev F1, **75.7% NEI Recall**.
    - **Tag SFT 4500 Brief CoT**: 74.2% Acc, 67.6% FEVER, 60.6% Ev F1, **78.0% NEI Recall**.
    - **JSON GRPO 1000 CoT** (`cot-native`): **74.7% Acc**, **67.3% FEVER**, **74.7% Non-NEI Ev F1** (All-time Repository Record!), 100% format rate, 67.7% NEI Recall.
    - **Tag GRPO 1000 CoT** (`cot-native-tag`): **90.1% Acc, 81.4% FEVER, 81.3% Ev F1** on Gold-in-Context ($n=517$); 67.6% Acc, 58.0% FEVER overall due to unmasked evidence confirmation bias on NEI (33.7% NEI Recall). Validates the Fast/Slow Thinking architecture in `INSIGHT.md` (Discussions 15 & 16).

11. [ ] **Present comparable model results and collect preference feedback**:
    - Show Base, direct SFT, CoT SFT, JSON CoT GRPO and Tag CoT GRPO on the same frozen claim IDs, retrieval context, prompt budget and evaluator. Report accuracy, FEVER, evidence F1, per-label recall, format rate, tokens and runtime; do not rank models using the gold-in-context subset as if it were the full evaluation.
    - Build a claim-level side-by-side review with the claim and identical shown evidence, two anonymized model outputs, citations and reasoning. Add buttons for **A better**, **B better**, **tie**, and **both wrong**, plus an optional reason. Save the exact prompt/context, model/run IDs, raw outputs, choice, reviewer and timestamp. Keep evaluation feedback separate from training data until a training set is explicitly curated.
    - For DPO, collect multiple CoT rollouts on **train-only** claims from different checkpoints or seeds with the same exact prompt. Pair a genuinely correct response with an incorrect one, audit reasoning faithfulness, and hold out review/evaluation claims. The existing 29 evaluation prediction files have zero ID overlap with the 5,895 training claims, so they cannot form training pairs.


## P0 — comparable verifier evaluation (RL-1, AG-3)

- [x] Train the no-CoT Qwen3.5-2B SFT adapter from the committed 5,895 examples (seed 0). The local run is `sft-direct-from-base-qwen3p5-2b-5895-samples/runs/run2` with checkpoint under `artifacts/checkpoints/`; the LoRA weights are not committed. The run metadata reports 6,720 seconds and 5.62 GB peak GPU memory.

- [ ] Freeze disjoint train, prompt-development, calibration and holdout claim IDs, then publish the split manifest and leakage check. The committed 5,895-row SFT train set and matched 1,200-row subset have zero ID overlap with the archived 900 dev claims; the remaining splits are not frozen.

- [ ] Run direct JSON (`THINKING=off`) and zero-shot evidence-chain (`THINKING=on`) evaluation on the same FEVER dev IDs and title retrieval: 30-claim smoke tests, then 900 claims. Submit `verifier/evaluation/evaluate.sbatch`; each run stores predictions, metrics, manifest and logs under `verifier/evaluation/results/<condition>/<run>/`.

- [ ] Record per claim: gold, verdict, evidence IDs, retrieval condition, reasoning if present, stated confidence, verdict-token probability, tokens and latency. Check evidence citations by hand; a valid E-ID alone does not prove support.

- [ ] Report accuracy, FEVER score, evidence F1, label recall, Brier, ECE, risk–coverage and evidence-in-context strata with one evaluator. `verifier/evaluation/score_predictions.py` already replays the archived outputs.


## P1 — prompt and inference controls (RL-1)

- [ ] Add a few-shot evidence-chain condition using examples outside evaluation IDs.

- [x] Implement SoCLaaS/Gemini CoT SFT teacher-data preparation from the existing 5,895 FEVER **train** oracle prompts, with a 100-per-class pilot, resumable requests and automated validation (`python -m utils.build_cot_sft_data`). The `qwen3.8:27b` pilot received all 300 responses; 227 passed and 73 were rejected (46 citation mismatches, 18 gold not justified by context, 8 wrong relations, 1 duplicate/too-many citations). Token usage: 442,495 prompt + 29,377 completion. Never send dev/evaluation claims to the teacher for training data.

- [ ] Use the pilot to assess explanation quality without per-row manual review: inspect automated rejection categories, add an independent evidence-faithfulness judge or targeted challenge set, and record its limits. Valid E IDs and a teacher self-check do not establish semantic faithfulness. The accepted CoT target omits `confidence`; decide how to train and evaluate this output format before using it as a checkpoint.

- [x] Add a separate CoT SFT training configuration and matching confidence-free evaluation prompt/format check. The 227 accepted teacher examples pass local preflight; training remains pending until the dataset and code are synced to SoC.

- [ ] Run the 227-row CoT SFT pilot and matched evaluation; track the NEI imbalance (51 NEI versus 90 SUPPORTED and 86 REFUTED). No CoT SFT checkpoint has been recorded yet. Do not interpret a comparison against the 5,895-row no-CoT SFT as an isolated CoT effect; make a no-CoT control on the same 227 claim IDs before claiming a training-method gain.

- [ ] After the oracle-prompt pilot, add **train-only** title-retrieval contexts so CoT SFT sees the evidence distribution used at evaluation. For non-NEI claims whose gold evidence is absent, do not teach the original FEVER label as though it were supported by the shown text; exclude them from ordinary CoT SFT or make a separately labeled contextual-abstention set. Keep the original FEVER gold label for evaluation and provenance.

- [ ] Once teacher responses pass validation, run the separate CoT SFT pilot adapter and evaluate it on the same held-out IDs as the no-CoT SFT checkpoint. Record accuracy, FEVER score, citation quality, output length and token probability; stated confidence is absent in this pilot.

- [ ] At a fixed total token budget, compare one response with multiple samples / voting.

- [ ] Compare direct, short evidence reasoning, and explicit evidence chain on quality, token use, and citation faithfulness.

- [ ] Audit the 1024-token sweep's failures and repeated reasoning: Base direct has 130 invalid outputs but only 3 at the cap; SFT long CoT has 48 invalid outputs, 44 capped. To isolate the cap's own effect, rerun selected prompts at both limits with the same code revision. Repeat on a CoT SFT checkpoint once available; the current no-CoT SFT has no reasoning target.


## P2 — confidence and reward (RL-2 with RL-1)

- [ ] Measure stated confidence and verdict-token probability separately; both predict whether the chosen verdict is correct. Inspect NEI and retrieval-miss subsets.

- [ ] Fit temperature scaling and a simple correctness predictor on a separate calibration split; evaluate once on holdout.

- [ ] Audit reward components, wrong-`SUPPORTED` harm, NEI behaviour, and confidence collapse. Compare reward ablations only with matched SFT and multiple seeds.


## P3 — train and integrate only after controls (RL-1/2/3, agent group)

- [ ] After the SFT-started GRPO pilot, compare it against the same SFT checkpoint with matched prompts, IDs and evaluator. For CoT GRPO, measure reasoning/citation quality, reward spread, accuracy, evidence, calibration, variance and compute cost; expand only if the pilot is informative.

- [ ] Connect the verifier to the agent loop's `verify()` interface. Compare one search, fixed NEI retry and confidence-guided retry before training a controller.
- [ ] Save the agent-side exhaustive 10-sentence Qwen batch outputs once under `agent/data/evidence_batches/`. Compare RSI-Jev v2.1 2B versus a fixed heuristic choosing cited-union or full-context evidence from the same batch summaries, then run fresh final Qwen verification for each chosen route; report paired accuracy, FEVER, evidence F1, route agreement and call cost. OpenJev-1.7B is an optional same-state controller comparison. See [`agent/data/evidence_batches/`](../agent/data/evidence_batches/) and [`agent/evaluation/results/rsi_jev_controller_baseline/`](../agent/evaluation/results/rsi_jev_controller_baseline/).

Agent-specific evaluation and controller work is tracked in [`agent/TODO.md`](../agent/TODO.md).

- [ ] Prepare the team failure audit, shortcut checks, transfer results and final demo from the same evaluation definitions.
