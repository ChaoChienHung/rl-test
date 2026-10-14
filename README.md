# Fact verification with RL

This repository studies evidence-grounded claim verification. Given a claim and retrieved evidence, a verifier returns `SUPPORTED`, `REFUTED`, or `NOT_ENOUGH_INFO`, cited sentence IDs, and a confidence. The team compares supervised fine-tuning (SFT), reinforcement learning (GRPO), and an agent loop that can search again before answering. The `rl` branch contains the verifier baseline work; it has not been pushed to `origin` yet.

## Current status (2026-10-14)

- **Retrieval and stage 0:** FEVER title matching reaches 0.875 page recall@5. The untrained Qwen3.5-2B scores 0.459 accuracy / 0.340 FEVER on 900 dev claims with title retrieval.
- **SFT:** A 2B LoRA trained on roughly 6,000 FEVER train claims scores 0.759 accuracy / 0.667 FEVER with title retrieval. Saved 900-claim outputs are in Git; adapter weights are not.
- **GRPO:** Several verifier runs are recorded. None yet beats the full 6,000-claim SFT on accuracy; verdict-token calibration worsens. See the exact conditions in the results document.
- **Agent/controller:** The agent loop has 900-claim traces and a 16-claim multi-turn GRPO spike. The separate controller remains a toy scaffold; no real-data controller result is established.
- **Next verifier comparison:** Fresh, matched direct-JSON and zero-shot evidence-chain prompts on SoC, followed by calibration and training only if the controls justify it.

## Doc map

| File | Use |
|---|---|
| [TODO.md](TODO.md) | Priorities, owners, and incomplete experiments |
| [factcheck/README.md](factcheck/README.md) | Data/results layout and exact baseline commands |
| [factcheck/docs/plan.md](factcheck/docs/plan.md) | System architecture, team roles, and experiment decisions |
| [factcheck/docs/data.md](factcheck/docs/data.md) | Download raw data and build the FEVER retrieval index |
| [factcheck/docs/stage0-results.md](factcheck/docs/stage0-results.md) | Untrained prompted baseline and retrieval results |
| [factcheck/docs/sft-probe.md](factcheck/docs/sft-probe.md) | SFT settings and outcomes |
| [factcheck/docs/grpo-results.md](factcheck/docs/grpo-results.md) | GRPO runs, comparisons, and limitations |
| [factcheck/docs/trace-format.md](factcheck/docs/trace-format.md) | Agent trace schema and tool interfaces |
| [factcheck/docs/related-work.md](factcheck/docs/related-work.md) | Literature notes; verify claims before citing |

The older proposal, team brief, and reports have been consolidated into these current files. Historical claims and schedules should be checked against the recorded runs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
# On SoC, install a CUDA-compatible PyTorch build for the allocated GPU first.
python -m pip install -r requirements.txt
```

Download FEVER and build the index using [factcheck/docs/data.md](factcheck/docs/data.md). Raw datasets and model weights are excluded from Git. Saved per-claim outputs can be evaluated without a GPU (see [factcheck/README.md](factcheck/README.md)). API keys for optional big-model calls belong in the ignored `.env` file.
