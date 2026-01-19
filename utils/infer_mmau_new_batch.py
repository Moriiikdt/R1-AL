import os
import json
import torch
import argparse
import re
from tqdm import tqdm
import librosa  # noqa: F401

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to checkpoint-XXXX-merged directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output jsonl file path. If not set, use default path based on model step."
    )
    return parser.parse_args()


def build_tag_from_model_path(model_path: str, repetition_penalty: float = 1.04) -> str:
    """
    Tag format:
      checkpoint-4400-merged -> step_review_4K6-3e-sft-RL-4400_1.02
      checkpoint_R1234_v2_3100 -> step_review_4K6-3e-sft-RL-3100_1.02
      .../checkpoint-3100 -> step_review_4K6-3e-sft-RL-3100_1.02
    """
    base = os.path.basename(os.path.normpath(model_path))

    patterns = [
        r"checkpoint-(\d+)-merged",          # checkpoint-3100-merged
        r"checkpoint-(\d+)$",                # checkpoint-3100
        r"checkpoint_(\d+)$",                # checkpoint_3100
        r"checkpoint_[A-Za-z0-9_]+_(\d+)$",  # checkpoint_R1234_v2_3100
    ]

    step = None
    for pat in patterns:
        m = re.search(pat, model_path) or re.search(pat, base)
        if m:
            step = m.group(1)
            break

    if step is None:
        # last resort: use the last number that appears in the path
        nums = re.findall(r"(\d+)", model_path)
        if nums:
            step = nums[-1]

    if step is None:
        raise ValueError(f"Cannot parse checkpoint step from model_path: {model_path}")

    return f"step_review_4K6-3e-sft-RL-{step}_{repetition_penalty:.2f}"



def load_processed_ids(output_path: str) -> set:
    """
    Load processed sample IDs from an existing JSONL output file.
    Each line is expected to be a JSON object containing an "id" field.
    """
    processed = set()
    if not os.path.exists(output_path):
        print(f"Output file not found, will start fresh: {output_path}")
        return processed

    bad_lines = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    processed.add(obj["id"])
            except json.JSONDecodeError:
                bad_lines += 1
                continue

    print(f"Found {len(processed)} processed IDs in: {output_path}")
    if bad_lines > 0:
        print(f"Warning: skipped {bad_lines} malformed lines in output JSONL.")
    return processed


# ----------------------------
# Configuration
# ----------------------------
args = parse_args()
model_path = args.model_path

# Keep this in sync with generate() so tag matches run setting
REPETITION_PENALTY = 1.02
tag = build_tag_from_model_path(model_path, repetition_penalty=REPETITION_PENALTY)

input_file = "./data/MMAU_new.json"
output_file = args.output


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 16
sys_prompt = "You are a helpful assistant."



def main():
    # ===== Load processed ids (for resume) =====
    processed_ids = load_processed_ids(output_file)

    # Step 1: Build your model
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    # Load the entire dataset into memory
    with open(input_file, "r", encoding="utf-8") as f:
        contents = json.load(f)

    # Open the output file once and write to it in append mode
    with open(output_file, "a", encoding="utf-8") as f_out:
        # Step 2: Iterate through the data in batches
        for i in tqdm(range(0, len(contents), batch_size), desc="Processing Batches"):
            raw_batch = contents[i:i + batch_size]

            # ---- Skip already processed by id ----
            batch_data = [item for item in raw_batch if item.get("id") not in processed_ids]
            if len(batch_data) == 0:
                continue

            batch_conversations = []
            batch_info = []

            # Step 3: Prepare all items in the batch
            for item in batch_data:
                prompt = item["question"] + "Select one option from the provided choices.\n" + "\n".join(item["choices"])

                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": sys_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": item["audio_id"]},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                batch_conversations.append(conversation)
                batch_info.append(item)

            USE_AUDIO_IN_VIDEO = False

            texts = processor.apply_chat_template(
                batch_conversations,
                add_generation_prompt=True,
                tokenize=False
            )

            audios, images, videos = process_mm_info(
                batch_conversations,
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )

            inputs = processor(text=texts, audio=audios, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device).to(model.dtype)

            generate_ids = model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                return_audio=False,
                thinker_max_new_tokens=2048,
                repetition_penalty=REPETITION_PENALTY
            )

            input_length = inputs["input_ids"].size(1)
            generate_ids = generate_ids[:, input_length:]

            responses = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # Step 5: Save the results for each item in the batch
            for idx, model_response in enumerate(responses):
                original_item = batch_info[idx]

                result_entry = {
                    "model_output": model_response,
                    "answer": original_item["answer"],
                    "task": original_item["task"],
                    "difficulty": original_item["difficulty"],
                    "choices": original_item["choices"],
                    "sub-category": original_item["sub-category"],
                    "id": original_item["id"],
                    "audio_id": original_item["audio_id"],
                    "question": original_item["question"],
                    "dataset": original_item["dataset"],
                    "category": original_item["category"],
                    "split": original_item["split"],
                }

                f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                processed_ids.add(original_item["id"])


if __name__ == "__main__":
    main()
