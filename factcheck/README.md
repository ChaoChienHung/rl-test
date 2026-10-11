# Fact verification with RL

A claim goes in, and the system returns three things: a verdict
(SUPPORTED / REFUTED / NOT ENOUGH INFO), the Wikipedia sentences it rests
on, and a confidence. The aim is a small model (Qwen3.5-2B) that a
calibration-aware reward trains to be right *and* honest about its
uncertainty, then a search controller that decides when to look further
and when to abstain.

## Where it stands

| Stage | Status | Doc |
|---|---|---|
| 0. Prompted baseline, retrieval | Done. 900 FEVER dev claims; title-match retrieval at 87.5% page recall@5 | [`docs/stage0-results.md`](docs/stage0-results.md) |
| 1a. SFT capacity probe (2B vs 4B) | Done. Decision: 2B | [`docs/sft-probe.md`](docs/sft-probe.md) |
| 1b. Single-turn GRPO on the verifier | First runs done, no gain over SFT yet; calibration worsens | [`docs/grpo-results.md`](docs/grpo-results.md) |
| 2. Agentic search controller | Toy scaffold only (12 hand-written claims) | [`docs/plan.md`](docs/plan.md), [`docs/trace-format.md`](docs/trace-format.md) |

## Docs

- [`docs/data.md`](docs/data.md): how to download FEVER, SciFact and AVeriTeC and build the index, plus what is in git and what is not.
- [`REPORT.md`](REPORT.md): plain-language report on what the project does and why continue with it over the math version.
- [`docs/plan.md`](docs/plan.md): architecture, what RL is for, the three stages, open decisions.
- [`docs/related-work.md`](docs/related-work.md): what is already published, what is open, numbers to beat.
- [`docs/models.md`](docs/models.md): which small model to train and which big model to call, with sizes and prices.
- [`docs/stage0-results.md`](docs/stage0-results.md): prompted baseline, 2B vs a 180B model, BM25 vs title-match retrieval.
- [`docs/sft-probe.md`](docs/sft-probe.md): SFT capacity probe (design, commands, decision rule, results).
- [`docs/grpo-results.md`](docs/grpo-results.md): GRPO runs, the 900-claim results table, and the training sets built for the next runs.
- [`docs/trace-format.md`](docs/trace-format.md): the per-claim trace schema shared by the verifier and the controller.
- [`docs/original-proposal.md`](docs/original-proposal.md): the source proposal, verbatim.

The team plan, proposal and one-page brief are at the repo root:
`report.md`, `proposal.tex` / `.pdf` / `.docx`, `team-brief.tex` / `.pdf`.

## Run

Set up the venv (top-level README), then fetch the data with
[`docs/data.md`](docs/data.md). All scripts run from `factcheck/src`.

```
python stage0.py --model local --n 900 --retrieval titles          # baseline on 900 dev claims
python sft_probe.py --model Qwen/Qwen3.5-2B --n-train 6000         # SFT
python grpo_verifier.py --beta 0.04 --filter --max-steps 200 --tag kl   # GRPO
python stage0.py --model local --adapter ../data/grpo/<run> --n 900 --retrieval oracle
python -m pytest test_reward.py                                    # reward unit tests
python run_scaffold.py                                             # toy controller scaffold
```

The GPU runs were made under WSL2 (Ubuntu 22.04, torch cu128, TRL).
See `docs/sft-probe.md` for the exact setup.

## Files

Verifier pipeline:

- `src/build_bm25_index.py`: BM25 index and `pages.sqlite` over the FEVER Wikipedia dump.
- `src/fever_retrieval.py`, `src/eval_retrieval.py`: query side of the index; page recall@k.
- `src/stage0.py`: prompt a model on N dev claims and score it (accuracy, FEVER score, evidence F1, per-label recall, ECE, format). `--adapter` scores a trained LoRA.
- `src/sft_probe.py`: LoRA / QLoRA SFT on FEVER train with gold pages in context.
- `src/reward.py`: correctness with asymmetric harm weights, negative-Brier calibration, evidence F1 with the NEI rule, format, and the `ABLATIONS` switches. `src/test_reward.py` pins its intended behaviour.
- `src/grpo_verifier.py`: single-turn GRPO from the SFT adapter, optional KL and spread filter.
- `src/disagreement.py`: samples the SFT model 8 times per claim to find which claims give GRPO a signal.
- `src/build_grpo_data.py`: GRPO prompts with retrieved evidence instead of gold.
- `src/build_nei_ablation.py`, `src/probe_nei_set.py`: evidence-removed NEI training sets and a probe of how a model answers them.

Controller scaffold (toy data, no model):

- `src/toy_corpus.py`: 12-claim stand-in for FEVER, word-overlap retriever, split-on-"and" decomposer.
- `src/verification_env.py`: Gymnasium MDP with retrieve / decompose / cross-check / abstain / commit at a stated confidence.
- `src/run_scaffold.py`: random vs heuristic policy over 200 episodes.

## Known gaps

- **The controller still runs on toy data.** It has not been wired to the
  FEVER retriever or to a trained verifier.
- **Retrieval is not entailment in the scaffold.** Word overlap finds
  sentences *about* the claim, including ones that refute it, so the toy
  heuristic never predicts REFUTED.
- **SciFact and AVeriTeC are downloaded but have no loaders yet.** They
  are the transfer test sets.
