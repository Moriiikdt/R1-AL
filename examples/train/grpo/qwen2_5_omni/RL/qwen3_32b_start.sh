export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve /data/maxiangnan/model-data/Qwen3-32B \
    --gpu-memory-utilization 0.8 \
    --served-model-name Qwen3-32B \
    --tensor-parallel-size 4 \
    --port 8000 \
    --host 0.0.0.0 \