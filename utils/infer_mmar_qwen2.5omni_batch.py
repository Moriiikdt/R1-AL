import os
import json
import torch
from tqdm import tqdm
import librosa

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# --- Configuration ---
tag = "R1234"

input_file = f'../bench/MMAR-meta.json'

output_file = f"./mmar_{tag}.jsonl"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set your desired batch size here. This will now be used for inference.
batch_size = 8 # For example, batch size of 8

# sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
# sys_prompt = "You are a multitask audio reasoning and understanding model. "
sys_prompt = "You are a helpful assistant."

def main():
    # RL完成的模型路径
    model_path = "path"
    # We recommend enabling flash_attention_2 for better acceleration and memory saving.
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
            # Get the slice of data for the current batch
            batch_data = contents[i:i + batch_size]
            
            batch_conversations = []
            batch_info = [] # To store original item data for saving later

            # Step 3: Prepare all items in the batch
            for item in batch_data:
                # choices_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(item['choices'])])
                # choices_str = " ".join([f"{choice}" for i, choice in enumerate(item['choices'])])
                suffix = (
                    "; Upon receiving a question, please respond in two parts: "
                    "'<think> and <answer>. The <think> section should be further divided into four parts: "
                    "<evidence>, <planning>, <reasoning>, <validating>, and <summarizing>'."
                )
                prompt = item['question'] + "\nSelect one option from the provided choices:\n" + "\n".join(item['choices']) 
                
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
                            {"type": "audio", "audio": item['audio_path']},
                            {"type": "text", "text": prompt}
                        ]
                    }]
                batch_conversations.append(conversation)
                batch_info.append(item) # Save the original item

            # Step 4: Process the entire batch at once
            USE_AUDIO_IN_VIDEO = False

            # Create a list of text prompts for the batch
            # texts = [processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)[0] for conv in batch_conversations]

            text = processor.apply_chat_template(batch_conversations, add_generation_prompt=True, tokenize=False)

            
            # Process multimedia info for the batch. `process_mm_info` should handle a list of conversations.
            # audios, images, videos  = process_mm_info(batch_conversations, SR=batch_data[0]["sample_rate"], use_audio_in_video=USE_AUDIO_IN_VIDEO)
            audios, images, videos  = process_mm_info(batch_conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)

            # The processor tokenizes the batch of texts and audio, padding them to the same length.
            inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device).to(model.dtype)

            # Generate responses for the entire batch
            generate_ids = model.generate(**inputs, 
                                        use_audio_in_video=USE_AUDIO_IN_VIDEO, 
                                        return_audio=False, 
                                        thinker_max_new_tokens=2048,
                                        output_scores=False,
                                        ) 

            # Remove the input tokens from the generated output
            input_length = inputs['input_ids'].size(1)
            generate_ids = generate_ids[:, input_length:]

            # Decode the batch of responses
            responses = processor.batch_decode(generate_ids, 
                                            skip_special_tokens=True, 
                                            clean_up_tokenization_spaces=False)

            # Step 5: Save the results for each item in the batch
            for idx, model_response in enumerate(responses):
                # Retrieve the original item corresponding to this response
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

                # Write the result as a new line in the JSONL file
                f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

