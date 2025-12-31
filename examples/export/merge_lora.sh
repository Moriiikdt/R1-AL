# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters /ssd1/maxiangnan/mrx/Evidence/ms-swift-main/examples/train/grpo/qwen2_5_omni/SFT/Qwen2.5-Omni-7B-SFT/v1-20251104-023237/checkpoint-235 \
    --merge_lora true
