#!/usr/bin/env bash

URL="http://127.0.0.1:8000/v1/completions"
INTERVAL=0.01   # 0.1 秒
LOG_DIR="./vllm_log"
mkdir -p "$LOG_DIR"

echo ">>> 开始持续访问 vLLM，每 ${INTERVAL}s 一次"
echo ">>> 日志目录: $LOG_DIR"

i=0
while true; do
  i=$((i+1))
  curl -s "$URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{
      "model": "Qwen3-32B",
      "prompt": "什么是强化学习？请详细描述",
      "max_tokens": 256,
      "temperature": 0.7
    }' \
    >> "$LOG_DIR/resp.log" 2>> "$LOG_DIR/error.log"

  echo "[`date '+%F %T'`] request $i sent"
  sleep "$INTERVAL"
done
