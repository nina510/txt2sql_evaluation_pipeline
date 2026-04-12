#!/usr/bin/env python3
"""从 Mini-Dev 写出 vLLM 所需的 prompt JSONL（每行含字段 prompt）。

题目可从本地 mini_dev_sqlite.json 读取，也可用 Hugging Face 数据集 birdsql/bird_mini_dev
（需 pip install datasets）。HF 上只有题目与标注，不含 .sqlite 库文件；拼 schema 仍需本地 dev_databases。
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LLM_SRC = os.path.join(REPO_ROOT, "llm", "src")
sys.path.insert(0, LLM_SRC)

from prompt import generate_combined_prompts_one  # noqa: E402

HF_DATASET = "birdsql/bird_mini_dev"
DB_DOWNLOAD_HINT = (
    "SQLite 库文件需单独下载完整包中的 dev_databases，例如：\n"
    "  https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip\n"
    "解压后将 dev_databases 放到 --db_root（默认 llm/data/dev_databases）。"
)


def load_rows_from_hf(split: str, offset: int, limit: int):
    try:
        from datasets import load_dataset
    except ImportError:
        print("从 Hugging Face 读取请先安装: pip install datasets")
        sys.exit(1)
    ds = load_dataset(HF_DATASET, trust_remote_code=True)[split]
    n = len(ds)
    end = min(n, offset + limit)
    if offset >= n:
        return []
    return [ds[i] for i in range(offset, end)]


def main():
    ap = argparse.ArgumentParser()
    default_eval = os.path.join(REPO_ROOT, "llm", "data", "mini_dev_sqlite.json")
    default_db = os.path.join(REPO_ROOT, "llm", "data", "dev_databases")
    ap.add_argument("--eval_json", default=default_eval, help="mini_dev_sqlite.json（--source local 时）")
    ap.add_argument("--db_root", default=default_db, help="dev_databases 目录")
    ap.add_argument("--out_jsonl", required=True, help="输出 JSONL")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sql_dialect", default="SQLite")
    ap.add_argument(
        "--source",
        choices=("auto", "local", "hf"),
        default="auto",
        help="auto：有本地 json 用本地，否则从 HF 拉题目；hf：始终用 HF；local：仅用本地 json",
    )
    ap.add_argument(
        "--hf_split",
        default="mini_dev_sqlite",
        help="HF 上的 split，如 mini_dev_sqlite / mini_dev_mysql / mini_dev_pg",
    )
    ap.add_argument(
        "--no_knowledge",
        action="store_true",
        help="不把 evidence 写进 prompt（默认会带上）",
    )
    args = ap.parse_args()

    use_hf = args.source == "hf"
    if args.source == "auto":
        use_hf = not os.path.isfile(args.eval_json)
        if use_hf:
            print("[info] 未找到本地 json，从 Hugging Face 加载题目表。")

    if use_hf:
        rows = load_rows_from_hf(args.hf_split, args.offset, args.limit)
    else:
        if not os.path.isfile(args.eval_json):
            print(
                f"找不到数据文件:\n  {args.eval_json}\n"
                "请放置 json 或改用 --source hf / --source auto。\n" + DB_DOWNLOAD_HINT
            )
            sys.exit(1)
        with open(args.eval_json, encoding="utf-8") as f:
            data = json.load(f)
        end = min(len(data), args.offset + args.limit)
        rows = data[args.offset : end]
    if not rows:
        print("没有可写入的样本（检查 offset/limit）。")
        sys.exit(1)

    out_abs = os.path.abspath(args.out_jsonl)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_ok = 0
    with open(out_abs, "w", encoding="utf-8") as out:
        for j, item in enumerate(rows):
            db_id = item["db_id"]
            sqlite_path = os.path.join(args.db_root, db_id, f"{db_id}.sqlite")
            if not os.path.isfile(sqlite_path):
                print(f"找不到数据库文件:\n  {sqlite_path}\n" + DB_DOWNLOAD_HINT)
                sys.exit(1)
            evidence = None if args.no_knowledge else item.get("evidence")
            prompt = generate_combined_prompts_one(
                db_path=sqlite_path,
                question=item["question"],
                sql_dialect=args.sql_dialect,
                knowledge=evidence,
            )
            idx = args.offset + j
            rec = {
                "prompt": prompt,
                "db_id": db_id,
                "source_index": idx,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"已写入 {n_ok} 行 -> {out_abs}")


if __name__ == "__main__":
    main()
