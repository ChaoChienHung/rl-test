# Project plan and ownership

Updated 2026-10-14. This is the current synthesis of the original proposal, team brief and later experiment reports. Measured results live in the untrained baseline, SFT and GRPO documents; tasks live in [TODO.md](../TODO.md).

## Goal and system

For each claim, produce a three-way verdict, evidence sentence IDs, and a confidence that estimates whether the chosen verdict is correct. FEVER supplies training and the initial 900-claim evaluation. SciFact and AVeriTeC are intended transfer sets; their loaders and final results are still pending.

```text
claim -> search -> evidence -> verifier -> verdict + cited evidence + confidence
                         ^                         |
                         +--- optional agent/controller retry or abstain
```

The retriever finds candidate text; it does not decide entailment. The verifier judges the evidence. The agent loop owns search, decomposition, retries, logging and final commitment. A controller can choose actions from those observations, but it is useful only if a simple fixed retry policy leaves an accuracy–cost gap. RL on the verifier and RL on a controller are separate experiments, not two competing copies of the whole system. The shared contracts are `search()`, `verify()`, `reward()`, and the [trace schema](../../agent/docs/trace-format.md).

SFT can teach the labelled verdict and evidence format directly. A verifier GRPO run is worth testing only if it improves an outcome SFT does not already provide, such as calibrated confidence, harmful-error trade-offs, or evidence use. The written SFT confidence target is always `1.0`, so it cannot teach graded uncertainty. For a controller, the reward is final decision quality minus search and token cost; its possible gain comes from deciding *when* another search is worth that cost. These are hypotheses to measure, not automatic reasons to train.

## Team ownership

The 8 October team brief assigns:

| Role | Person | Main responsibility |
|---|---|---|
| RL-1 | Chao Chien Hung | Verifier SFT/GRPO, baselines, training and before/after comparisons |
| RL-2 | Michel Yildirim | Reward, confidence calibration, ablations and reliability analysis |
| RL-3 | Yao Tianhao | Real-data controller, fixed-policy controls and accuracy–cost curve |
| AG-1 | Yu Jing | Retrieval and search tools, recall and transfer knowledge stores |
| AG-2 | Chia Kwang Yang | Agent loop, traces, prompted agent baseline and multi-turn GRPO |
| AG-3 | Bryan Wu Jiahe | Data splits, shared evaluation, error audit and demo |

These are ownership notes, not proof that any pending component has been completed. The proposal and the later team brief disagree on some earlier AG name-to-role assignments; this table follows the later brief.

## Evidence and decisions so far

1. Title matching lifted FEVER page recall@5 to 0.875, but the untrained 2B verifier still reached only 0.459 accuracy on the full 900-claim title-retrieval test. See [untrained baseline](prompted-baseline-results.md).
2. SFT gave a large gain: 0.759 accuracy with title retrieval. The training target writes confidence `1.0`, so stated confidence cannot be treated as learned calibration. See [SFT](sft-probe.md).
3. The recorded single-turn GRPO runs have not beaten the 6,000-claim SFT accuracy, and their verdict-token ECE is worse. The matched 1,200-claim SFT control and seed variance matter when comparing runs. See [GRPO](grpo-results.md).
4. A direct-JSON versus evidence-chain prompt comparison is the next inexpensive test. It changes prompting, not model weights. CoT + GRPO is a hypothesis to test if reasoning improves outcomes and rollout rewards vary; a chain alone does not guarantee faithful evidence use.
5. Confidence must mean probability that the chosen verdict is correct. Evaluate stated text and verdict-token probability separately with binary correctness Brier/ECE and risk–coverage. Fit post-hoc calibration on a separate split before claiming RL adds value.

## Evaluation and scope

Use identical claim IDs, retrieval, model and decoding for prompt comparisons, and log token cost. Separate oracle evidence from real title retrieval; otherwise retrieval failures can be mistaken for verifier failures. Audit claims with confidently wrong answers, missing gold evidence, NEI, and unsupported citations. For the controller, compare one search, a fixed NEI retry, and a learned policy on the same accuracy–cost axes. Do not infer transfer from FEVER alone.

The original proposal's claims about novelty, fixed six-week schedule and expected RL gains were hypotheses. [Related work](related-work.md) is a reading list whose unverified entries need source checks before citation.
