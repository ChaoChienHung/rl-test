# Stage 1b: GRPO on the verifier, results so far

Single-turn GRPO with `src/grpo_verifier.py`: claim plus evidence in, the
stage-0 JSON out, scored by `src/reward.py` (full reward: correctness with
harm weights, negative Brier calibration, evidence, format). All runs use
Qwen3.5-2B, 1,200 FEVER train claims with the gold pages in context, 8
samples per prompt, 2 prompts per step and lr 2e-5. Every row is scored by
`stage0.py --adapter` on the same 900 dev claims as `stage0-results.md`
and `sft-probe.md`.

Run metadata: `data/grpo/<run>/meta.json`. Training logs:
`data/grpo/*.log`. Eval logs: `data/grpo/eval_*.log` and
`data/sft_probe/eval_n1200_*.log`. Per-claim outputs: `data/stage0/`.

## 1. Runs

| Run (`data/grpo/...`) | Starts from | KL beta | Train claims | Steps | Train time | Peak GB |
|---|---|---|---|---|---|---|
| `Qwen_Qwen3.5-2B_full_n1200` | SFT adapter (trained in place) | 0 | all 1,200 | 600 | 3.0 h | 13.6 |
| `Qwen_Qwen3.5-2B_base_full_n1200` | base model | 0.04 | all 1,200 | 600 | 3.6 h | 13.5 |
| `Qwen_Qwen3.5-2B_full_n1200_kl` | SFT merged, fresh LoRA | 0.04 | only those with reward spread | 200 | 1.0 h | 13.7 |
| `Qwen_Qwen3.5-2B_full_n1200_kl_s1` | same, seed 1 | 0.04 | only those with reward spread | 200 | 0.7 h | 13.6 |
| `Qwen_Qwen3.5-2B_full_n1200_kl05` | SFT merged, fresh LoRA | 0.5 | only those with reward spread | 200 | 0.8 h | 13.6 |
| `Qwen_Qwen3.5-2B_full_n16_smoke` | SFT adapter | 0 | 16 | 3 | 72 s | 9.7 |

"Reward spread" is the `--filter` flag. It keeps only the claims where
`disagreement.py` found the 8 SFT samples scoring differently.

## 2. Results on 900 FEVER dev claims

| Model | Retrieval | Acc | FEVER | Ev F1 | R-SUP | R-REF | R-NEI | Acc, gold in ctx | ECE stated | ECE token | Format |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2B, no training (stage 0) | titles | 0.459 | 0.340 | 0.423 | 0.940 | 0.300 | 0.137 | 0.632 | - | 0.308 | 0.843 |
| 2B SFT, 6,000 claims | oracle | 0.779 | 0.702 | 0.688 | 0.867 | 0.750 | 0.720 | 0.814 | - | 0.028 | 1.000 |
| 2B SFT, 6,000 claims | titles | 0.759 | 0.667 | 0.631 | 0.827 | 0.727 | 0.723 | 0.824 | - | 0.034 | 1.000 |
| 2B SFT, same 1,200 claims (matched control) | oracle | 0.736 | 0.634 | 0.629 | 0.827 | 0.737 | 0.643 | 0.785 | 0.264 | 0.040 | 1.000 |
| 2B SFT, same 1,200 claims (matched control) | titles | 0.712 | 0.616 | 0.594 | 0.793 | 0.693 | 0.650 | 0.793 | 0.288 | 0.037 | 1.000 |
| GRPO from SFT, no KL | oracle | 0.769 | 0.697 | 0.684 | 0.817 | 0.773 | 0.717 | 0.800 | 0.231 | 0.214 | 1.000 |
| GRPO from SFT, no KL | titles | 0.747 | 0.658 | 0.625 | 0.797 | 0.730 | 0.713 | 0.818 | 0.253 | 0.232 | 1.000 |
| GRPO from base, KL 0.04 | oracle | 0.718 | 0.580 | 0.626 | 0.783 | 0.857 | 0.513 | 0.823 | 0.255 | 0.252 | 1.000 |
| GRPO from base, KL 0.04 | titles | 0.688 | 0.528 | 0.540 | 0.717 | 0.830 | 0.517 | 0.799 | 0.282 | 0.283 | 1.000 |
| GRPO from SFT, KL 0.04, filtered | oracle | 0.771 | 0.694 | 0.713 | 0.817 | 0.860 | 0.637 | 0.843 | 0.229 | 0.209 | 1.000 |
| same, seed 1 | oracle | 0.746 | 0.694 | 0.644 | 0.857 | 0.607 | 0.773 | 0.737 | 0.254 | 0.237 | 1.000 |
| GRPO from SFT, KL 0.5, filtered | oracle | 0.769 | 0.691 | 0.746 | 0.860 | 0.850 | 0.597 | 0.860 | 0.231 | 0.164 | 1.000 |

The stage-0 and SFT-6,000 rows come from `stage0-results.md` and
`sft-probe.md`. Column meanings are the same as in those docs. "Oracle"
guarantees the gold pages are in context. "Titles" is the real retrieval
pipeline.

## 3. What the numbers say

- **No run beats SFT on 6,000 claims yet.** The best GRPO rows reach
  0.77 accuracy under oracle retrieval. That is level with SFT's 0.779,
  and above the SFT control trained on the same 1,200 claims (0.736).
- **Calibration got worse.** SFT's verdict-token probability is well
  calibrated (ECE 0.03 to 0.04). Every GRPO run pushes it to 0.16 to 0.25:
  the model becomes certain. Stated confidence stays at 1.0 throughout, so
  its ECE equals 1 minus accuracy.
- **Why the first run barely moved.** `disagreement.py` sampled the SFT
  model 8 times per training claim (`data/grpo/disagreement_n1200.log`).
  77% of claims had some reward spread and 69% had disagreeing verdicts,
  but during the unfiltered run about 88% of groups had zero spread. As
  the model sharpened, its 8 samples converged and GRPO lost its signal.
  The KL term and the `--filter` flag are the response to that.
- **Seed variance is large.** The KL 0.04 run with seed 0 vs seed 1
  differs by 2.5 points of accuracy and by 25 points of REFUTED recall
  (0.86 vs 0.61). Single-seed differences under about 3 points are not
  evidence.
- **Starting from the base model is worse** on every metric except
  REFUTED recall. NEI recall falls to 0.51.

## 4. Built for the next runs, not yet trained on

| Set (`data/grpo_data/...`) | Built by | What it is |
|---|---|---|
| `titles_n1200` | `build_grpo_data.py` | The same 1,200 claims with title-match retrieval instead of gold pages, so GRPO practises on what the search loop will hand the verifier |
| `nei_page_n1200` | `build_nei_ablation.py --mode page` | NEI by construction: SUPPORTED/REFUTED claims with every gold page removed |
| `nei_sentence_n1200` | `build_nei_ablation.py --mode sentence` | Gold pages kept, annotated evidence lines removed. Noisier, because FEVER annotation is incomplete |

`probe_nei_set.py` ran the 6,000-claim SFT adapter on 300 claims of each
NEI set (`data/grpo_data/probe_sft.log`). It answered NEI on 58% of the
page set: (77 + 96) / 300. On the sentence set it answered NEI on 52%:
(73 + 82) / 300. So the sets are not trivially easy for the current
model, and both have room to teach abstention.
