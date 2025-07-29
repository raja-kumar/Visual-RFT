from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cpu", # Load to CPU initially
)

# Move model to CUDA device 0 after loading
model = model.to(torch.device(0))
model = model.eval()

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

image_path = "../../data/oxford_flowers/jpg/image_02395.jpg"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": "What is the most likely flower species in the image? Only give the name of the flower species."},
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
    "max_new_tokens": 10,  # Allow for a few tokens, as "Gladiolus" is one token
    "temperature": 1,
    "top_p": 0.95,
    "do_sample": True,
    "num_return_sequences": 1, # Keep as 1 to analyze the probabilities of one generated sequence
    "repetition_penalty": 1.1,
    "output_scores": True,        # Key parameter to get scores for all generated tokens
    "return_dict_in_generate": True, # Returns a GenerationOutput object
}

# Inference: Generation of the output
generated_output = model.generate(**inputs, **generation_args)

# --- Code to get top K predictions for each generated token ---

# Access the generated sequence (the best one since num_return_sequences=1)
generated_ids = generated_output.sequences[0] # Get the first (and only) sequence

# The input_ids length needs to be considered to get only the newly generated tokens
# The generated_ids will include the prompt tokens as well.
# The scores array corresponds *only* to the newly generated tokens.
input_len = inputs.input_ids.shape[1]
generated_new_tokens = generated_ids[input_len:]

# Define K for top K predictions
top_k = 5 # You can change this value

print(f"\nPrompt: {messages[0]['content'][1]['text']}")
print(f"Complete generated response (decoded): {processor.tokenizer.decode(generated_new_tokens, skip_special_tokens=True).strip()}")

if generated_output.scores:
    print("\nTop K predictions for each generated token:")
    for i, logits_for_step in enumerate(generated_output.scores):
        # logits_for_step will have shape (batch_size, vocab_size).
        # Since num_return_sequences=1 and batch_size=1, we can squeeze.
        probabilities_for_step = torch.softmax(logits_for_step.squeeze(0), dim=-1)

        # Get the top K probabilities and their indices (token IDs)
        top_k_probs, top_k_indices = torch.topk(probabilities_for_step, top_k, dim=-1)

        # Decode the actual token generated at this step
        actual_token_id = generated_new_tokens[i].item()
        actual_token = processor.tokenizer.decode([actual_token_id]).strip()

        print(f"\n--- Step {i+1} (Generated Token: '{actual_token}') ---")
        for j in range(top_k):
            decoded_token = processor.tokenizer.decode([top_k_indices[j].item()]).strip()
            probability = top_k_probs[j].item()
            print(f"  {j+1}. Token: '{decoded_token}', Probability: {probability:.4f}")
else:
    print("No scores were returned. Generation might have been empty or an issue occurred.")