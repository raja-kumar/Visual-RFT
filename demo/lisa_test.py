import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
from PIL import Image
import logging
from tqdm import tqdm
import re
import math
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)
img2description = dict()

THINKING = False

if (THINKING):
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

else:
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "answer is enclosed within <answer> </answer> tag, i.e., "
        "<answer> answer here </answer>"
    )

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward", device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained("Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward")


def prepare_inputs(img_path, instruction):

    if (THINKING):
        message = f"Output the bounding box in the image corresponding to the instruction: {instruction}. Output the thinking process in <think> </think> and your grouding box. Following \"<think> thinking process </think>\n<answer>(x1,y1),(x2,y2)</answer>)\" format."
    else:
        message = f"Output the bounding box in the image corresponding to the instruction: {instruction}. Output your grouding box after thinking. Following \"<answer>(x1,y1),(x2,y2)</answer>)\" format."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": message}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to("cuda")

# image_path = "../assets/pokeymon.jpg"
# inputs = prepare_inputs(image_path, "the pokeymon that can perform Thunderbolt. Output thinking process as detail as possibile")

# image_path = "./275616162_91d7ca1ed6_o.jpg"
# inputs = prepare_inputs(image_path, "In the process of remodeling or repairing a room, which tool in the picture would be used for creating holes in the walls or ceiling?")

# image_path = "./302806585_b4aa483f69_o.jpg"
# inputs = prepare_inputs(image_path, "the damaged part of the silk stockings")

image_path = "./276438292_33033cd241_o.jpg"
inputs = prepare_inputs(image_path, "the food with high protein")

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)
response = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(response)

pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
matches = re.findall(pattern, response)
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)
w, h = Image.open(image_path).size
x1, y1, x2, y2 = map(int, matches[0])
box_r1 = [int(x1) / 1000, int(y1) / 1000, int(x2) / 1000, int(y2) / 1000]
draw = ImageDraw.Draw(image)
draw.rectangle([box_r1[0] * w, box_r1[1] * h, box_r1[2] * w, box_r1[3] * h], outline="green", width=5)
image_id = os.path.basename(image_path).split(".")[0]
image.save(f"{image_id}_thinking_{THINKING}.jpg")