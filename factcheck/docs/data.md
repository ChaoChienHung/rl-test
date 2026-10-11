# Data: what is in git, and how to get the rest

Everything lives under `factcheck/data/`. The raw datasets and the
retrieval index are about 21 GB, so git carries only what the project
produced (eval outputs, training sets, logs, run metadata). The rest is
downloaded and built once per machine with the steps below.

## 1. Download the raw datasets

Run from the repo root. Needs `curl`, `unzip` and `tar` (Git Bash on
Windows has all three).

```bash
# FEVER 1.0: claims (train, shared-task dev) and the 2017 Wikipedia dump
mkdir -p factcheck/data/fever && cd factcheck/data/fever
curl -LO https://fever.ai/download/fever/train.jsonl              # 33 MB
curl -LO https://fever.ai/download/fever/shared_task_dev.jsonl    # 4.3 MB
curl -LO https://fever.ai/download/fever/wiki-pages.zip           # 1.7 GB
unzip -q wiki-pages.zip && rm wiki-pages.zip                      # -> wiki-pages/wiki-001..109.jsonl
cd ../../..

# SciFact: claims and the abstract corpus
mkdir -p factcheck/data/scifact && cd factcheck/data/scifact
curl -LO https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz   # 3 MB
tar -xzf data.tar.gz                                              # -> data/claims_*.jsonl, corpus.jsonl
cd ../../..

# AVeriTeC: train and dev claims with QA evidence (official repo)
mkdir -p factcheck/data/averitec && cd factcheck/data/averitec
curl -LO https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/train.json   # 10 MB
curl -LO https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json     # 1.8 MB
cd ../../..
```

Sources: [fever.ai](https://fever.ai/dataset/fever.html),
[SciFact](https://github.com/allenai/scifact),
[AVeriTeC](https://github.com/MichSchli/AVeriTeC). The sizes above match
the copies the results in this repo were produced from. FEVER is
CC BY-SA 3.0 (Wikipedia-derived); check each dataset's license before
redistributing.

## 2. Build the retrieval index

```bash
cd factcheck/src
python build_bm25_index.py     # writes data/fever/bm25_index/ and data/fever/pages.sqlite
python eval_retrieval.py       # optional check: page recall@5 should match stage0-results.md
```

The index is about 2.3 GB and `pages.sqlite` about 5.4 GB. The build runs once.

## 3. Expected layout

```
factcheck/data/
  fever/          train.jsonl, shared_task_dev.jsonl, wiki-pages/, bm25_index/, pages.sqlite   (not in git)
  scifact/data/   claims_{train,dev,test}.jsonl, corpus.jsonl, cross_validation/               (not in git)
  averitec/       train.json, dev.json                                                         (not in git)
  stage0/         per-claim eval outputs, one jsonl per (model, adapter, retrieval) run        (in git)
  sft_probe/      SFT runs: meta.json, train_log.json, train.jsonl, adapter_config.json; eval logs (in git)
  grpo/           GRPO runs: meta.json, train_log.json, adapter_config.json; train/eval logs   (in git)
  grpo_data/      alternative GRPO training sets (retrieved evidence, evidence-removed NEI)     (in git)
```

## 4. What is left out of git, and how to get it back

| Left out | Size | How to regenerate |
|---|---|---|
| LoRA adapters (`adapter_model.safetensors`) | 65 MB each | Rerun the command in that run's `meta.json` (`sft_probe.py` or `grpo_verifier.py`); the docs give the exact commands |
| Trainer checkpoints (`checkpoint-*/`) | 220 MB each | Written during training; only needed to resume a run |
| Tokenizer and chat template copies, `training_args.bin`, TRL model cards | 20 MB per run | Written next to every adapter; identical to the base model's |
| `sft_probe/Qwen_Qwen3.5-4B_n6000_qlora/train.jsonl` | 30 MB | Byte-identical copy of `sft_probe/Qwen_Qwen3.5-2B_n6000/train.jsonl` |

Paths inside the committed `meta.json` files are absolute paths from the
machine that ran them (`/mnt/c/...` under WSL2). Read them as relative to
`factcheck/data/`.
