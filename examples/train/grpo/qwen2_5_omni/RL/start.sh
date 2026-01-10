LOG_DIR="./log"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve /mnt/hdfs/if_au/models/Qwen3-32B --gpu-memory-utilization 0.8  --served-model-name Qwen3-32B --tensor-parallel-size 4 --port 8000 --host 0.0.0.0 &

while ! nc -z 127.0.0.1 8000; do
  sleep 1
done

echo "vllm 启动成功"

nohup bash vllm_stress.sh &

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
