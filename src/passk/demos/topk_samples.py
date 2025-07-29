from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cpu",
)

model = model.to(torch.device(0))
model = model.eval()

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
image_path = "../../../data/oxford_flowers/jpg/image_02395.jpg"
prompt = " This is an image containing a plant. Please identify the species of the plant based on the image.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:\n<think> ... </think> <answer>species name</answer>\nPlease strictly follow the format. "
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": prompt},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generation_args = {
    "max_new_tokens": 512,
    "temperature": 1,
    "top_p": 0.95,
    "do_sample": True,
    "num_return_sequences": 20,
    "repetition_penalty": 1.1,
}

# Inference: Generation of the output
generated_ids = model.generate(**inputs, **generation_args)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)

input_id_length = inputs.input_ids.shape[1] # Length of the single input sequence
num_sequences = generation_args["num_return_sequences"]

output_texts = []
for i in range(num_sequences):
    # Extract each generated sequence
    # Each generated sequence in `generated_ids` will start with the input_id_length
    # and then the new tokens.
    start_index = i * generated_ids.shape[1] if num_sequences > 1 else 0 # this is incorrect
    # Correction: The `generate` method, when num_return_sequences > 1 for a single input,
    # returns a tensor where each row is a full generated sequence (input + generated).
    # So, generated_ids will have a shape of (num_return_sequences, sequence_length).
    
    # Let's verify the shape of generated_ids directly.
    # If inputs.input_ids has shape (1, input_len) and num_return_sequences=4,
    # then generated_ids will have shape (4, output_len).

    # So, we just need to iterate through the rows of generated_ids.
    trimmed_id = generated_ids[i][input_id_length:]
    decoded_text = processor.decode(trimmed_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    print("------------------------------------------")
    print(f"Decoded text for sequence {i+1}: {decoded_text}")
    print("------------------------------------------")
    output_texts.append(decoded_text)

# print(output_texts)