NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type grpo \
    --model sft_path \
    --reward_funcs evidence_driven_format answer_exact_match cot_quality_judge_vllm \
    --reward_weights 0.5 1 1 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "../../../../../data/avqa_RL_46544_format_abs_audio.json" \
    --load_from_cache_file true \
    --external_plugins "../../plugin/plugin.py" \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_step_reason_4K6_R124 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1. \
    --top_p 0.99 \
    --top_k 50 \
    --system '../../prompt.txt' \
    --deepspeed zero2 \
    --log_completions true

