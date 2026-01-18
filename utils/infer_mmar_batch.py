import os
import json
import torch
import argparse
import re
from tqdm import tqdm
import librosa

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ----------------------------
# Configuration
# ----------------------------
args = parse_args()
model_path = args.model_path
tag = build_tag_from_model_path(model_path)

input_file = "./data/MMAR.json"
output_file = f"./mmar_{tag}.jsonl"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 16
sys_prompt = "You are a helpful assistant."


# ----------------------------
# Command line arguments
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to checkpoint-XXXX-merged directory"
    )
    return parser.parse_args()


def build_tag_from_model_path(model_path: str) -> str:
    """
    Extract step number from checkpoint-XXXX-merged
    and build tag string.
    """
    match = re.search(r"checkpoint-(\d+)-merged", model_path)
    if match is None:
        raise ValueError(
            f"Cannot parse checkpoint step from model_path: {model_path}"
        )
    step = match.group(1)
    tag = f"step_review_4K6-3e-sft-RL-{step}_1.03"
    return tag


def main():
    # Step 1: Build model
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    # Load dataset
    with open(input_file, "r", encoding="utf-8") as f:
        contents = json.load(f)

    # Open output file
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(contents), batch_size), desc="Processing Batches"):
            batch_data = contents[i:i + batch_size]

            batch_conversations = []
            batch_info = []

            for item in batch_data:
                prompt = (
                    item['question']
                    + "\nSelect one option from the provided choices:\n"
                    + "\n".join(item['choices'])
                )

                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": sys_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": item['audio_path']},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                batch_conversations.append(conversation)
                batch_info.append(item)

            USE_AUDIO_IN_VIDEO = False

            text = processor.apply_chat_template(
                batch_conversations,
                add_generation_prompt=True,
                tokenize=False
            )

            audios, images, videos = process_mm_info(
                batch_conversations,
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )

            inputs = processor(
                text=text,
                audio=audios,
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(model.device).to(model.dtype)

            generate_ids = model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                return_audio=False,
                thinker_max_new_tokens=2048,
                output_scores=False,
                repetition_penalty=1.03
            )

            input_length = inputs['input_ids'].size(1)
            generate_ids = generate_ids[:, input_length:]

            responses = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            for idx, model_response in enumerate(responses):
                original_item = batch_info[idx]
                result_entry = {
                    "model_output": model_response,
                    "answer": original_item['answer'],
                    "modality": original_item['modality'],
                    "category": original_item['category'],
                    "choices": original_item['choices'],
                    "sub-category": original_item['sub-category'],
                    "id": original_item['id'],
                    "audio_path": original_item['audio_path'],
                    "question": original_item['question'],
                }
                f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
