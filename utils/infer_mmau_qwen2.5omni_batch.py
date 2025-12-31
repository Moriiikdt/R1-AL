import os
import json
import torch
from tqdm import tqdm
import librosa  # noqa: F401

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# --- Configuration ---
tag = "R1234"

input_file = f'../bench/mmau-test-mini.json'
output_file = f"./mmau_{tag}.jsonl"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set your desired batch size here. This will now be used for inference.
batch_size = 8  # For example, batch size of 8

sys_prompt = "You are a helpful assistant."


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


def main():
    # ===== Load processed ids (for resume) =====
    processed_ids = load_processed_ids(output_file)

    # RL完成的模型路径
    model_path = "path"
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
    with open(output_file, 'a', encoding='utf-8') as f_out:
        # Step 2: Iterate through the data in batches
        for i in tqdm(range(0, len(contents), batch_size), desc="Processing Batches"):
            raw_batch = contents[i:i + batch_size]

            # ---- Skip already processed by id ----
            batch_data = [item for item in raw_batch if item.get("id") not in processed_ids]
            if len(batch_data) == 0:
                continue

            batch_conversations = []
            batch_info = []  # To store original item data for saving later

            # Step 3: Prepare all items in the batch
            for item in batch_data:
                prompt = item['question'] + "Select one option from the provided choices.\n" + "\n".join(item['choices'])

                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": sys_prompt}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": item['audio_id']},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                batch_conversations.append(conversation)
                batch_info.append(item)  # Save the original item

            # Step 4: Process the entire batch at once
            USE_AUDIO_IN_VIDEO = False

            # Create a list of text prompts for the batch
            texts = processor.apply_chat_template(batch_conversations, add_generation_prompt=True, tokenize=False)

            # Process multimedia info for the batch.
            audios, images, videos = process_mm_info(batch_conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)

            # The processor tokenizes the batch of texts and audio, padding them to the same length.
            inputs = processor(text=texts, audio=audios, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device).to(model.dtype)

            # Generate responses for the entire batch
            generate_ids = model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                return_audio=False,
                thinker_max_new_tokens=2048,  # v3
            )

            # Remove the input tokens from the generated output
            input_length = inputs['input_ids'].size(1)
            generate_ids = generate_ids[:, input_length:]

            # Decode the batch of responses
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
                    "answer": original_item['answer'],
                    "task": original_item['task'],
                    "difficulty": original_item['difficulty'],
                    "choices": original_item['choices'],
                    "sub-category": original_item['sub-category'],
                    # Retain original info
                    "id": original_item['id'],
                    "audio_id": original_item['audio_id'],
                    "question": original_item['question'],
                    "dataset": original_item['dataset'],
                    "category": original_item['category'],
                    "split": original_item['split']
                }

                # Write the result as a new line in the JSONL file
                f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

                # Update processed ids so we won't re-run within this execution
                processed_ids.add(original_item["id"])


if __name__ == "__main__":
    main()
