# rl-test: fact verification with RL

Claim verification with a small model trained by RL (SFT, then GRPO with a
calibration-aware reward), and a cost-aware search controller on top. The
code, docs and results are in [`factcheck/`](factcheck/README.md).

| File | What |
|---|---|
| [`factcheck/README.md`](factcheck/README.md) | Project overview, status by stage, how to run, file map |
| [`factcheck/docs/data.md`](factcheck/docs/data.md) | Download FEVER, SciFact and AVeriTeC and build the index |
| [`report.md`](report.md) | Team plan: split into RL and agentic sub-teams, schedule, failure scenarios, decisions |
| `proposal.tex` / `.pdf` / `.docx` | Project proposal |
| `team-brief.tex` / `.pdf` | One-page team brief (week of 8 Oct 2026) |

## Setup

```
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
```

Then fetch the datasets with [`factcheck/docs/data.md`](factcheck/docs/data.md).
API keys (OpenRouter, for the big-model baseline) go in `.env` at the repo
root, which is gitignored.
