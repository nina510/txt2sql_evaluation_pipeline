#!/usr/bin/env bash
# 先试 5 条：需 vLLM；题目默认从 HF 拉（pip install datasets），无本地 json 也可。
# SQLite 库仍需解压 minidev.zip 里的 dev_databases/ 到 llm/data/dev_databases/
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-./outputs_try5}"
mkdir -p "$OUT_DIR"
PROMPTS="${OUT_DIR}/prompts_5.jsonl"
PRED="${OUT_DIR}/pred_5.json"
REPORT="${OUT_DIR}/report_5.jsonl"
DB_ROOT="${DB_ROOT:-/home/yw28498/mini_dev/llm/data/dev_databases}"
EVAL_JSON="${EVAL_JSON:-/home/yw28498/mini_dev/llm/data/mini_dev_sqlite.json}"

echo "[1/3] 生成 prompt JSONL（5 条，--source auto：有本地 json 用本地，否则 Hugging Face）..."
python3 build_minidev_prompts.py --out_jsonl "$PROMPTS" --limit 5 --source auto

echo "[2/3] vLLM 推理..."
python3 vllm_infer.py \
  --model_path "${MODEL_PATH:-Qwen/Qwen2.5-Coder-1.5B-Instruct}" \
  --prompt_path "$PROMPTS" \
  --output_path "$PRED" \
  --gpu "${GPU:-0}" \
  --batch_size 2 \
  --max_token_length 8192

echo "[3/3] 在 SQLite 执行并对照标注 SQL..."
python3 eval_sqlite_with_gold.py \
  --pred_json "$PRED" \
  --prompt_jsonl "$PROMPTS" \
  --db_root "$DB_ROOT" \
  --report_jsonl "$REPORT" \
  --source auto \
  --eval_json "$EVAL_JSON" \
  --hf_split mini_dev_sqlite

echo "完成。预测 SQL: $PRED"
echo "完成。逐条执行对照报告: $REPORT"
