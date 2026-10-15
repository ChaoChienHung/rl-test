# Data: what is in git, and how to get the rest

Raw and prepared datasets live under `verifier/data/`; run artifacts live
under `verifier/experiments/` and fresh evaluation results under
`verifier/evaluation/results/`. The raw datasets and retrieval index are about
21 GB, so they are downloaded and built once per machine with the steps below.

## 1. Download the raw datasets

Run from the repo root. Needs `curl`, `unzip` and `tar` (Git Bash on
Windows has all three).

```bash
# FEVER 1.0: claims (train, shared-task dev) and the 2017 Wikipedia dump
mkdir -p verifier/data/fever && cd verifier/data/fever
curl -LO https://fever.ai/download/fever/train.jsonl              # 33 MB
curl -LO https://fever.ai/download/fever/shared_task_dev.jsonl    # 4.3 MB
curl -LO https://fever.ai/download/fever/wiki-pages.zip           # 1.7 GB
unzip -q wiki-pages.zip && rm wiki-pages.zip                      # -> wiki-pages/wiki-001..109.jsonl
cd ../../..

# SciFact: claims and the abstract corpus
mkdir -p verifier/data/scifact && cd verifier/data/scifact
curl -LO https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz   # 3 MB
tar -xzf data.tar.gz                                              # -> data/claims_*.jsonl, corpus.jsonl
cd ../../..

# AVeriTeC: train and dev claims with QA evidence (official repo)
mkdir -p verifier/data/averitec && cd verifier/data/averitec
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
python -m retrieval.build_bm25_index     # writes verifier/data/fever/bm25_index/ and verifier/data/fever/pages.sqlite
python -m retrieval.eval_retrieval       # optional check: page recall@5 should match prompted-baseline-results.md
```

The index is about 2.3 GB and `pages.sqlite` about 5.4 GB. The build runs once.

## 3. Expected layout

```
verifier/data/
  fever/          train.jsonl, shared_task_dev.jsonl, wiki-pages/, bm25_index/, pages.sqlite   (not in git)
  scifact/data/   claims_{train,dev,test}.jsonl, corpus.jsonl, cross_validation/               (not in git)
  averitec/       train.json, dev.json                                                         (not in git)
  train/sft/      model-independent SFT labeled_prompts.jsonl sets                         (in git)
  train/grpo/     GRPO labeled_prompts.jsonl sets and disagreement samples                 (in git)
  eval/           frozen FEVER dev claim IDs                                                (in git)
  traces/         archived agent search/verify action histories                             (in git)
verifier/experiments/<configuration>/runs/<run>/artifacts/
  meta.json, logs/train_log.json, checkpoints/adapter_config.json, optional rollouts.jsonl (in git)
  checkpoints/*.safetensors and checkpoint-*/                                             (not in git)
verifier/evaluation/results/archive/        archived per-claim evaluation outputs                           (in git)
verifier/evaluation/results/<condition>/<run>/  fresh predictions, metrics, manifest and logs                (not in git)
```

## 4. What is left out of git, and how to get it back

| Left out | Size | How to regenerate |
|---|---|---|
| LoRA adapters (`adapter_model.safetensors`) | 65 MB each | Rerun `verifier/experiments/<config>/train.py --run-name <new-run>` after downloading raw data and restoring any required SFT adapter; settings come from the historical metadata |
| Trainer checkpoints (`checkpoint-*/`) | 220 MB each | Written during training; only needed to resume a run |
| Tokenizer and chat template copies, `training_args.bin`, TRL model cards | 20 MB per run | Written next to every adapter; identical to the base model's |
| Duplicate 4B SFT training JSONL | 30 MB | The same model-independent `verifier/data/train/sft/fever-oracle-full/labeled_prompts.jsonl` is shared by the 2B and 4B configurations |

Paths inside historical `meta.json` files are absolute paths from the
machine that ran them (`/mnt/c/...` under WSL2). They describe those old
runs and may point to the former `data/` artifact layout; use the current
`verifier/experiments/<configuration>/runs/<run>/artifacts/` location for files in
this branch.

The historical terminal `.log` files were removed because their useful
results are in the per-claim JSONL, `meta.json`, structured `train_log.json`
or the experiment documents. Future raw terminal logs under `verifier/data/` are
ignored; new SoC job logs belong under their experiment run's
`artifacts/logs/` for training, or under `verifier/evaluation/results/<condition>/<run>/logs/` for evaluation; terminal logs are ignored.

## 5. Fixed SFT/GRPO claim split

`python -m utils.build_disjoint_sft_grpo_split --check` verifies the frozen 5,895-row oracle set and its claim-disjoint 4,695/1,200 split. The 1,200-row side reuses `sft/fever-oracle-matched-grpo-1200/labeled_prompts.jsonl`; the 4,695-row complement is `sft/fever-oracle-disjoint-4695/labeled_prompts.jsonl`. `split_manifest.json` records SHA-256 hashes, row and label counts, and zero overlap with the frozen 900 dev IDs. To regenerate the 4,695-row side from those committed inputs, run `python -m utils.build_disjoint_sft_grpo_split` from the repo root.
