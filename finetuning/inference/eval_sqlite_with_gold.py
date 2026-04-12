#!/usr/bin/env python3
"""
Compare predicted SQL with gold SQL on SQLite and export per-example report.

Inputs:
- pred_json: {"0": "...", "1": "..."} produced by vllm_infer.py
- prompt_jsonl: each row includes {"db_id": "...", "source_index": ...}

Outputs:
- report_jsonl: one row per example with pred/gold SQL and execution results
- summary printed to stdout
"""
import argparse
import json
import os
import sqlite3
import sys
from typing import Any

HF_DATASET = "birdsql/bird_mini_dev"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_rows_from_hf(split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets first: pip install datasets")
        sys.exit(1)
    ds = load_dataset(HF_DATASET, trust_remote_code=True)[split]
    return [ds[i] for i in range(len(ds))]


def load_rows_from_local_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_sql(db_path: str, sql: str) -> tuple[bool, list[Any] | None, str | None]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return True, rows, None
    except Exception as e:
        return False, None, str(e)
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate predicted SQL against gold SQL (SQLite)")
    ap.add_argument("--pred_json", required=True)
    ap.add_argument("--prompt_jsonl", required=True)
    ap.add_argument("--db_root", required=True, help="Path to dev_databases/")
    ap.add_argument("--report_jsonl", required=True)
    ap.add_argument("--source", choices=("auto", "local", "hf"), default="auto")
    ap.add_argument("--eval_json", default="")
    ap.add_argument("--hf_split", default="mini_dev_sqlite")
    args = ap.parse_args()

    prompts = load_jsonl(args.prompt_jsonl)
    with open(args.pred_json, "r", encoding="utf-8") as f:
        pred_map = json.load(f)

    use_hf = args.source == "hf"
    if args.source == "auto":
        use_hf = not (args.eval_json and os.path.isfile(args.eval_json))

    if use_hf:
        full_rows = load_rows_from_hf(args.hf_split)
    else:
        if not args.eval_json or not os.path.isfile(args.eval_json):
            print("Missing --eval_json for local source. Use --source hf/auto or set --eval_json.")
            sys.exit(1)
        full_rows = load_rows_from_local_json(args.eval_json)

    os.makedirs(os.path.dirname(os.path.abspath(args.report_jsonl)), exist_ok=True)

    total = 0
    correct = 0
    executed_pred = 0
    executed_gold = 0

    with open(args.report_jsonl, "w", encoding="utf-8") as out:
        for i, meta in enumerate(prompts):
            idx = meta["source_index"]
            db_id = meta["db_id"]
            pred_sql = pred_map.get(str(i), "")
            gold_sql = full_rows[idx]["SQL"]
            question = full_rows[idx].get("question", "")

            db_path = os.path.join(args.db_root, db_id, f"{db_id}.sqlite")
            if not os.path.isfile(db_path):
                rec = {
                    "row_id": i,
                    "source_index": idx,
                    "db_id": db_id,
                    "question": question,
                    "pred_sql": pred_sql,
                    "gold_sql": gold_sql,
                    "pred_ok": False,
                    "gold_ok": False,
                    "pred_error": f"DB not found: {db_path}",
                    "gold_error": f"DB not found: {db_path}",
                    "ex": 0,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                continue

            pred_ok, pred_rows, pred_err = run_sql(db_path, pred_sql)
            gold_ok, gold_rows, gold_err = run_sql(db_path, gold_sql)

            if pred_ok:
                executed_pred += 1
            if gold_ok:
                executed_gold += 1

            ex = int(pred_ok and gold_ok and set(pred_rows or []) == set(gold_rows or []))
            correct += ex
            total += 1

            rec = {
                "row_id": i,
                "source_index": idx,
                "db_id": db_id,
                "question": question,
                "pred_sql": pred_sql,
                "gold_sql": gold_sql,
                "pred_ok": pred_ok,
                "gold_ok": gold_ok,
                "pred_error": pred_err,
                "gold_error": gold_err,
                "pred_rows_preview": (pred_rows[:5] if pred_rows else []),
                "gold_rows_preview": (gold_rows[:5] if gold_rows else []),
                "pred_row_count": (len(pred_rows) if pred_rows is not None else -1),
                "gold_row_count": (len(gold_rows) if gold_rows is not None else -1),
                "ex": ex,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    acc = (correct / total * 100.0) if total else 0.0
    print(f"[Eval] total={total} ex_correct={correct} ex_acc={acc:.2f}%")
    print(f"[Eval] pred_executed={executed_pred}/{total}, gold_executed={executed_gold}/{total}")
    print(f"[Eval] report={args.report_jsonl}")


if __name__ == "__main__":
    main()
