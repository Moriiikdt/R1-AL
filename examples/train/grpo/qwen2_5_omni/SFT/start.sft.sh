#!/bin/bash
# run_all.sh
set -euo pipefail

LOG_DIR="/ssd1/maxiangnan/mrx/Evidence/ms-swift-main/examples/train/grpo/qwen2_5_omni/SFT/log"
mkdir -p "$LOG_DIR"

echo ">>> 启动 sft.sh ..."
nohup bash sft.sh \
  >> "$LOG_DIR/train_sft_5K-5e_step_new.log" 2>&1 &
TRAIN_PID=$!
echo "sft.sh 已在后台运行，PID=$TRAIN_PID"

echo "✅ 全部任务已启动！日志目录：$LOG_DIR"
