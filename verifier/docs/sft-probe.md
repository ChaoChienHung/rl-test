# SFT capacity probe: 2B or 4B?

Decision experiment, set up 2026-09-25. 2B run 2026-09-26, 4B run
2026-10-03; results in section 5.

## 1. Question

Untrained baseline measured a 32-point gap between the 2B and the 180B model when
the gold evidence is in context (63% vs 95%). That gap is between two
models of very different size. It says nothing about how much of it a
2B model *can* close. Before GRPO time is spent, find the 2B model's
ceiling with the cheapest training signal there is: supervised
fine-tuning with the answer on the page.

If SFT with gold evidence in context cannot get the 2B model to roughly
80% accuracy, the limit is capacity, not training method, and stage 1
should run on Qwen3.5-4B.

## 2. Design

| | |
|---|---|
| Train data | FEVER train, balanced by label, default 6,000 claims |
| Context | "Oracle" retrieval: pages of the smallest gold evidence set are guaranteed, title-match pages fill to 5, order shuffled per claim. NEI claims get plain title-match pages. Same prompt and sentence cut as untrained baseline. |
| Target | `{"verdict": <gold>, "evidence": [<gold ids>], "confidence": 1.0}` |
| Method | LoRA r=16 on all linear layers, lr 1e-4, 1 epoch, batch 8 (1 x 8 accumulation), 4,096 max tokens. 4B uses 4-bit base weights (QLoRA). |
| Eval | `python -m verifier.evaluation.eval_verifier --adapter`, the same 900 dev claims as untrained baseline, in two conditions: `oracle` (ceiling) and `titles` (the real pipeline) |
| Metrics | Accuracy, FEVER score, evidence F1, per-label recall, token-probability ECE, format adherence. Accuracy on the gold-in-context subset under `titles` is the number that compares directly to untrained baseline's 0.632. |

The stated confidence is 1.0 in every training target, so after SFT the
stated-confidence ECE is meaningless by construction. Ignore it here.
Token-probability ECE is still informative.

## 3. Runs and decision rule

Four models, two retrieval conditions each. Order matters: the 2B runs
decide whether the 4B runs are needed at all.

```bash
# From the repository root; use a fresh run name because historical run1 exists.
python verifier/experiments/sft_qwen_3.5-2b_n60/train.py --run-name run2
python -m verifier.evaluation.eval_verifier --model local --adapter verifier/experiments/sft_qwen_3.5-2b_n60/runs/run2/artifacts/checkpoints --n 30 --retrieval oracle

python verifier/experiments/sft_qwen_3.5-2b_n6000/train.py --run-name run2
python -m verifier.evaluation.eval_verifier --model local --adapter verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints --n 900 --retrieval oracle
python -m verifier.evaluation.eval_verifier --model local --adapter verifier/experiments/sft_qwen_3.5-2b_n6000/runs/run2/artifacts/checkpoints --n 900 --retrieval titles

python -m verifier.evaluation.eval_verifier --model local --n 900 --retrieval oracle
python -m verifier.evaluation.eval_verifier --model local:Qwen/Qwen3.5-4B --n 900 --retrieval oracle
python -m verifier.evaluation.eval_verifier --model local:Qwen/Qwen3.5-4B --n 900 --retrieval titles

python verifier/experiments/sft_qwen_3.5-4b_n6000_qlora/train.py --run-name run2
python -m verifier.evaluation.eval_verifier --model local --adapter verifier/experiments/sft_qwen_3.5-4b_n6000_qlora/runs/run2/artifacts/checkpoints --n 900 --retrieval oracle
python -m verifier.evaluation.eval_verifier --model local --adapter verifier/experiments/sft_qwen_3.5-4b_n6000_qlora/runs/run2/artifacts/checkpoints --n 900 --retrieval titles
```

Standard error on 900-claim accuracy is about 1.6 points, so treat
differences under 4 points as noise.

| Outcome of 2B SFT, oracle accuracy | Decision |
|---|---|
| >= 0.80 | Stay on 2B. Capacity is not the limit; the gap is trainable. Skip the 4B runs. |
| 0.70 to 0.80 | Run 4B. Pick 4B only if it beats 2B by 5+ points *and* its training time per step is under 3x the 2B time; otherwise 2B, since GRPO needs many more steps than SFT. |
| < 0.70 | Check before blaming the model: training loss still falling at the end (then raise `--n-train`), format failures (see per-key format rate), evidence F1 near zero (then the target ids are wrong). If none of those, run 4B and expect to switch. |

Also record from `meta.json`: `train_seconds` and `peak_gpu_gb`. GRPO on
the chosen model will need several times the SFT memory (reference
model, 8 rollouts per prompt) and far more steps. If 4B QLoRA already
sits above 12 GB or runs slower than 3x the 2B rate, 4B GRPO is not
realistic on one 16 GB card in six weeks whatever its accuracy.

## 4. Known risks in running this

- Qwen3.5's linear-attention layers run on slow reference kernels on
  Windows (see `models.md`). Training may be several times slower than
  a normal 2B model. If a 6,000-example epoch is projected past about
  six hours, either move to WSL2 first or cut `--n-train` to 3,000.
- bitsandbytes 4-bit on Windows works for standard linear layers; the
  linear-attention projections have not been checked. If the 4B QLoRA
  load fails, run it under WSL2.
- `target_modules="all-linear"` may pick up vision-tower layers if the
  causal-LM load keeps them. They get no gradient and only cost a little
  memory. If memory is tight, list the text-tower module names instead.
- TRL's `SFTConfig` field names change between versions. The script
  targets the `max_length` / `completion_only_loss` names; if it errors
  on a field, check the installed TRL version's `SFTConfig`.

## 5. Results

Fill in. Untrained baseline rows are from `prompted-baseline-results.md`.

| Model | Trained | Retrieval | Acc | FEVER | Ev F1 | R-SUP | R-REF | R-NEI | Acc, gold in ctx | ECE token | Format | Train s | Peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2B | no | titles | 0.459 | 0.340 | 0.423 | 0.940 | 0.300 | 0.137 | 0.632 | 0.308 | 0.843 | - | - |
| 2B | no | oracle | 0.469 | 0.370 | 0.428 | 0.953 | 0.307 | 0.147 | 0.630 | 0.298 | 0.877 | - | - |
| 2B | SFT | oracle | 0.779 | 0.702 | 0.688 | 0.867 | 0.750 | 0.720 | 0.814 | 0.028 | 1.000 | 2190 | 5.18 |
| 2B | SFT | titles | 0.759 | 0.667 | 0.631 | 0.827 | 0.727 | 0.723 | 0.824 | 0.034 | 1.000 | 2190 | 5.18 |
| 4B | no | titles | 0.669 | 0.569 | 0.635 | 0.903 | 0.883 | 0.220 | 0.913 | 0.234 | 0.998 | - | - |
| 4B | no | oracle | 0.674 | 0.616 | 0.671 | 0.937 | 0.887 | 0.200 | 0.911 | 0.233 | 1.000 | - | - |
| 4B | SFT | oracle | 0.788 | 0.721 | 0.687 | 0.877 | 0.753 | 0.733 | 0.821 | 0.047 | 1.000 | 5143 | 4.98 |
| 4B | SFT | titles | 0.772 | 0.690 | 0.652 | 0.853 | 0.737 | 0.727 | 0.851 | 0.050 | 1.000 | 5143 | 4.98 |
| 180B | no | titles | 0.720 | 0.640 | 0.717 | 0.907 | 0.907 | 0.347 | 0.952 | 0.214 | 0.999 | - | - |

2B rows run 2026-09-26 under WSL2 (Ubuntu 22.04, torch 2.11 cu128, transformers
5.17, TRL 1.14, flash-linear-attention with Triton; `causal_conv1d` not
installed). 5,895 training examples after 105 drops (gold line past the
sentence cut); 1 example over the 4,096-token limit. Training ran at about
3 s per step, 737 steps; each 900-claim eval took about 5 minutes. Loss fell
from 0.23 to about 0.05 by step 90 and stayed flat after that.

4B rows run 2026-10-03, same WSL2 setup, QLoRA (nf4, double quant), trained
on the same 5,895 examples as 2B (`train.jsonl` copied over). About 7 s per
step, 2.35x the 2B time; peak 4.98 GB, below 2B's bf16 5.18 GB. Loss fell
from 0.14 to about 0.04. Eval loads the base in bf16 and merges the adapter,
so the adapter runs on slightly different weights than it was trained on.
Each 900-claim eval took about 10 minutes.

Reading. Under the rule in section 3, 2B SFT oracle is 0.779 (0.70 to 0.80
band), and 4B SFT beats it by 0.9 points oracle and 1.3 points titles, inside
noise and short of the 5-point bar. The rule says 2B. One caveat: the
untrained 4B already gets 0.91 on gold-in-context claims (2B: 0.63, 180B:
0.95), with NEI recall at 0.20. SFT bought NEI recall (0.20 to 0.73) at the
cost of about 9 points on gold-in-context claims and 13 points of REFUTED
recall, so after SFT the two sizes look alike. The 4B's evidence reading is
much stronger than SFT shows; a reward that does not trade it away (GRPO)
could keep more of it. Worth weighing before locking in 2B.

Decision: Qwen3.5-2B (2026-10-08, team decision, recorded by KY). 4B SFT
beat 2B SFT by 0.9 points oracle and 1.3 titles, inside noise and short of
the 5-point bar, and the untrained 2B's 63% vs 95% on gold-in-context claims
leaves the most room to show what RL adds. The 4B's stronger untrained
evidence reading (0.91) stays a known caveat for the write-up.
