LOG_DIR="./log"
mkdir -p "$LOG_DIR"

echo ">>> 启动 grpo_R1234 训练 ..."
nohup bash grpo_R1234.sh \
  >> "$LOG_DIR/train1234.log" 2>&1 &
TRAIN_PID=$!
echo "grpo.sh 已在后台运行，PID=$TRAIN_PID"

echo ">>> 启动 grpo_R12 训练 ..."
nohup bash grpo_R12_simple.sh \
  >> "$LOG_DIR/train12.log" 2>&1 &
TRAIN_PID=$!
echo "grpo.sh 已在后台运行，PID=$TRAIN_PID"

echo ">>> 启动 grpo_R12 训练 ..."
nohup bash grpo_R12.sh \
  >> "$LOG_DIR/train12.log" 2>&1 &
TRAIN_PID=$!
echo "grpo.sh 已在后台运行，PID=$TRAIN_PID"

echo ">>> 启动 grpo_R123 训练 ..."
nohup bash grpo_R123.sh \
  >> "$LOG_DIR/train123.log" 2>&1 &
TRAIN_PID=$!
echo "grpo.sh 已在后台运行，PID=$TRAIN_PID"

echo ">>> 启动 grpo_R124 训练 ..."
nohup bash grpo_R124.sh \
  >> "$LOG_DIR/train124.log" 2>&1 &
TRAIN_PID=$!
echo "grpo.sh 已在后台运行，PID=$TRAIN_PID"
