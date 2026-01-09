LOG_DIR="./log"
mkdir -p "$LOG_DIR"

echo ">>> 启动 grpo_R1234 训练 ..."
nohup bash grpo_R1234.sh >> "$LOG_DIR/train1234.log" 2>&1
echo ">>> grpo_R1234 训练完成"

echo ">>> 启动 grpo_R12_simple 训练 ..."
nohup bash grpo_R12_simple.sh >> "$LOG_DIR/train12_simple.log" 2>&1
echo ">>> grpo_R12_simple 训练完成"

echo ">>> 启动 grpo_R12 训练 ..."
nohup bash grpo_R12.sh >> "$LOG_DIR/train12.log" 2>&1
echo ">>> grpo_R12 训练完成"

echo ">>> 启动 grpo_R123 训练 ..."
nohup bash grpo_R123.sh >> "$LOG_DIR/train123.log" 2>&1
echo ">>> grpo_R123 训练完成"

echo ">>> 启动 grpo_R124 训练 ..."
nohup bash grpo_R124.sh >> "$LOG_DIR/train124.log" 2>&1
echo ">>> grpo_R124 训练完成"
