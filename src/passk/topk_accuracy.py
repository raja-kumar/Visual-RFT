import io
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.generation import GenerationConfig
torch.manual_seed(1234)

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools
import itertools
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool
import argparse
from prompts import PROMPTS

import random
random.seed(21)

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色



def parse_args():
    parser = argparse.ArgumentParser(description="Top-K Accuracy Evaluation")
    parser.add_argument("--model_root", type=str, default="/app/saved_models/vrft/CUB_200_2011/")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--exp_name", type=str, default="Qwen2_5-VL-7B-Instruct_GRPO_cub_base_and_hard_mcq")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-400")
    parser.add_argument("--zero_shot", type=str, default=True)
    parser.add_argument("--eval_type", type=str, default="baseline")
    parser.add_argument("--use_cat_list", type=str, default=False)
    parser.add_argument("--data_root", type=str, default="/data2/raja/")
    parser.add_argument("--dataset", type=str, default="CUB_200_2011")
    parser.add_argument("--split", type=str, default="new_val")
    parser.add_argument("--num_return_sequences", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()

def clean_string(text):
    """
    Cleans the input text by removing unwanted characters and formatting.
    """
    text = text.replace("'s", "")
    text = re.sub(r'[^a-zA-Z0-9-]', ' ', text)
    text = text.strip().lower()
    
    return text

args = parse_args()

MODEL_ROOT = args.model_root
BASE_MODEL = args.base_model
EXP_NAME = args.exp_name
CHECKPOINT = args.checkpoint
zero_shot = args.zero_shot.lower() == "true"
eval_type = args.eval_type
use_cat_list = args.use_cat_list.lower() == "true"
DATA_ROOT = args.data_root
dataset = args.dataset
split = args.split
num_return_sequences = args.num_return_sequences
temperature = args.temperature
max_new_tokens = args.max_new_tokens


model_path = os.path.join(MODEL_ROOT, f"{EXP_NAME}", CHECKPOINT)  # full path to the model"
model_base = BASE_MODEL  # base model name


if EXP_NAME == "baseline":
    model_path = BASE_MODEL

zero_shot_json_path = f"{DATA_ROOT}/{dataset}/zero_shot/subsample_{split}.json"
output_path = f"./output/{dataset}/topk_accuracy/{eval_type}/"

if "checkpoint" in model_path:
    model_name = model_path.split("/")[-2] + "_" + model_path.split("/")[-1] # use checkpoint name
else:
    model_name = model_path.split("/")[-1]  # model name

data_name = zero_shot_json_path.split("/")[-1].split(".")[0]  # data name
output_file = f"{model_name}_{data_name}_{use_cat_list}_{num_return_sequences}_{temperature}.json"  # output file name

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_file_path = os.path.join(output_path, output_file)

print(GREEN + "output path" + output_file_path + RESET)
output_data = {}

if use_cat_list:
    split_name = split.split("_")[0] 
    category_file = f"{DATA_ROOT}/{dataset}/zero_shot/{split_name}_categories.txt"

    with open(category_file, 'r') as f:
        categories = f.read().splitlines()

def run(rank, world_size):

    local_output_data = {}

    if "Qwen2.5" in model_base:

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
    
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )

    processor = AutoProcessor.from_pretrained(model_base) 

    model = model.to(torch.device(rank))
    model = model.eval()

    with open(zero_shot_json_path, 'r') as f:
        infer_data = json.load(f)
    
    random.seed(21)
    random.shuffle(infer_data)

    # infer_data = infer_data[:100]

    print(GREEN + "Number of images in infer data: " + str(len(infer_data)) + RESET)
    

    rank = rank
    world_size = world_size
    import math
    split_length = math.ceil(len(infer_data)/world_size)
    logger.info("Split Chunk Length:" + str(split_length))
    split_images = infer_data[int(rank*split_length) : int((rank+1)*split_length)]
    logger.info(len(split_images))

    
    if (temperature == 0.0):
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_return_sequences": num_return_sequences,
            "use_cache": True,
            "temperature": None
        }
    else:
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "do_sample": True,
            "num_return_sequences": num_return_sequences,
            "repetition_penalty": 1.1,
            "use_cache": True,
        }
    
    print(YELLOW + f"generation args: {generation_args}" + RESET)

    for item in tqdm(split_images, total=len(split_images), desc=f"Rank {rank} Processing"):
        image_path = item['image_path']
        image_label = item['solution']

        prompt = item['problem']
        image_label = re.search(r"<answer>(.*?)</answer>", image_label).group(1)
        image_label = clean_string(image_label)
        
        if (dataset == "fgvc_aircraft"):
            image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/fgvc_aircraft/", 
                        DATA_ROOT)
        else:
            image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/", 
                        DATA_ROOT)
        
        temp, answer_format, data_name = PROMPTS[dataset]["instruction"], PROMPTS[dataset]["answer_format"], PROMPTS[dataset]["data_name"]
        if use_cat_list:
            question = (
            f"This is an image containing a {data_name}. {temp}\n"
            f"the {answer_format} of the {data_name} strictly belongs to below category list {categories}.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "The output answer format should be as follows:\n"
            f"<think> ... </think> <answer> {answer_format} </answer>\n"
            +
            "Please strictly follow the format."
            )
        else:
            question = prompt
        # print(RED + question + RESET)
    
        query = "<image>\n"+question
        # print(RED+query+RESET)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ] + [{"type": "text", "text": query}],
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
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        try:
            generated_ids = model.generate(**inputs, **generation_args)
        except Exception as e:
            print(RED + "Error during model generation: " + str(e) + RESET)
            print(RED + "Skipping image: " + image_path + RESET)
            continue
        
        input_id_length = inputs.input_ids.shape[1] # Length of the single input sequence
        num_sequences = generation_args["num_return_sequences"]

        image_id = image_path.split("/")[-1].split(".")[0]

        # curr_pred = set()  # Use a set to avoid duplicates
        curr_pred = {}

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
            response = processor.decode(trimmed_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # output_texts.append(decoded_text)

            try:
                # For other cases, keep the original parsing logic
                reasoning = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
                reasoning_content = reasoning.group(1).strip() if reasoning else ""
                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if not match:
                    match = re.search(r"<answer>\n(.*?)</answer>", response, re.DOTALL)
                if not match:
                    match = re.search(r"<answer>\n(.*?)\n</answer>", response, re.DOTALL)
                
                answer_content = match.group(1).strip().lower().replace(f"{answer_format}: ", "")
                answer_content = clean_string(answer_content)

                # local_output_data[image_id].append({
                #     "groundtruth": image_label,
                #     "reasoning": reasoning_content,
                #     "answer": answer_content
                # })
                # curr_pred.add(answer_content.strip().lower())
                if answer_content not in curr_pred:
                    curr_pred[answer_content] = 1
                else:
                    curr_pred[answer_content] += 1

            except Exception as e:
                print(RED + f"Error in processing response: {e}" + RESET)
                print(RED + "Response: " + response + RESET)
        
        ## add the groundtruth to the output data
        local_output_data[image_id] = {
            "groundtruth": image_label,
            "predictions": curr_pred,  # Store predictions as a set to avoid duplicates
        }

        
    return [local_output_data]

def main():
    multiprocess = torch.cuda.device_count() >= 1
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size)
            result_lists = pool.map(func, range(world_size))
        
        logger.info('finished generation')
        for i in range(world_size):
            output_data.update(result_lists[i][0])  # merge local output data
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()

    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Output saved to {output_file_path}")