export NCCL_IB_DISABLE="1"
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model /data/maxiangnan/model-data/Qwen2.5-Omni-7B \
    --dataset '/ssd1/maxiangnan/mrx/Evidence/ms-swift-main/data/sft/new_step/avqa_5K_mcts_sft.jsonl' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 2 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir /ssd1/maxiangnan/mrx/Evidence/ms-swift-main/examples/train/grpo/qwen2_5_omni/SFT/Qwen2.5-Omni-7B-SFT \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
