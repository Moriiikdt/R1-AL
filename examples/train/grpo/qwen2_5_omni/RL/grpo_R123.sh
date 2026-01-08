NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type grpo \
    --model /ssd1/maxiangnan/mrx/Evidence/ms-swift-main/examples/train/grpo/qwen2_5_omni/SFT/Qwen2.5-Omni-7B-SFT/v1-20251231-161222/checkpoint-948-merged \
    --reward_funcs evidence_driven_format answer_exact_match perception_judge_vllm \
    --reward_weights 0.1 1.5 1.5 \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "/ssd1/maxiangnan/mrx/Evidence/ms-swift-main/data/avqa_RL_46544_format.json" \
    --load_from_cache_file true \
    --external_plugins "/ssd1/maxiangnan/mrx/Evidence/ms-swift-main/examples/train/grpo/plugin/plugin.py" \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 50 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_step_reason_4K6_R1234_v2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1. \
    --top_p 0.99 \
    --top_k 50 \
    --system '/ssd1/maxiangnan/mrx/Evidence/ms-swift-main/examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --resume_from_checkpoint /ssd1/maxiangnan/mrx/Evidence/ms-swift-main/examples/train/grpo/qwen2_5_omni/RL/output_step_reason_4K6_R1234_v2/v1-20260104-164918/checkpoint-4400 \
    --log_completions true

# --resume_from_checkpoint /ssd1/maxiangnan/mrx/Evidence/ms-swift-main/examples/train/grpo/qwen2_5_omni/RL/output_step_reason_4K6_R1234/v2-20251231-173855/checkpoint-300 \
