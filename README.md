# Mini Dev Homework README

This document is my personal usage note for running the `mini_dev` repository.

## 1. Pull latest code

```bash
git pull --ff-only origin main
```

## 2. Environment setup

Use Python 3.11 and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Project structure (quick view)

- `evaluation/`: EX, R-VES, Soft-F1 evaluation scripts
- `llm/`: LLM inference and prompt-related code
- `live_sql_bench_sqlite/`: LiveSQLBench SQLite evaluation pipeline
- `finetuning/`: data prep, inference, and finetuning helpers

## 4. Run standard EX evaluation

1) Edit `evaluation/run_evaluation.sh` and set:
- `predicted_sql_path` to your prediction file
- `sql_dialect` (for example `SQLite`)

2) Run:

```bash
cd evaluation
sh run_evaluation.sh
```

The script calls `evaluation/evaluation_ex.py` and writes results to `eval_result/`.

## 5. Single-instance evaluation support

The repo includes single-instance evaluation code:

- Script: `live_sql_bench_sqlite/evaluation/single_instance_eval_sqlite.py`

Expected usage:

```bash
python3 live_sql_bench_sqlite/evaluation/single_instance_eval_sqlite.py \
  --jsonl_file <single_instance.jsonl> \
  --output_file <result.json> \
  --mode pred \
  --logging true
```

Notes:

- The script reads the first record in `--jsonl_file` as one instance.
- It supports `--mode pred` and `--mode gold`.
- By default it uses a hardcoded SQLite DB path (`/Volumes/SN770/...`), unless `EPHEMERAL_DB_PATH` is set.
- You may need to adjust database path logic in this script for your local machine.

## 6. Output and submission suggestion

- Keep your prediction files in a separate folder (for example `sql_result/`).
- Keep your evaluation outputs in `eval_result/`.
- Include this README and your command history in your homework submission.
