#!/bin/bash

MODEL_ROOT="/app/saved_models/vrft/oxford-iiit-pets/"  # root path for saved models
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# EXP_NAME="Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_and_hard_mcq"  # experiment name for saving models
CHECKPOINT="checkpoint-400"  # checkpoint name for saved models


# ==== configurations ====
zero_shot="True"
eval_type="rft"  # "sft" or everything else

# ==== dataset and output paths ====
DATA_ROOT="/data2/raja/datasets/"
dataset="oxford-iiit-pet"  # oxford_flowers, oxford-iiit-pet, CUB_200_2011

## === generation settings ===
max_new_tokens=1024

splits=("base_val" "new_val")  # splits to evaluate on
EXP_NAMES=("Qwen2_5-VL-7B-Instruct_GRPO_pets_base_virft_updated_reward" "Qwen2_5-VL-7B-Instruct_GRPO_pets_base_qwen_mcq")
use_cat_lists=("False" "True")  # whether to use category list in the prompt

for EXP_NAME in "${EXP_NAMES[@]}"; do
    for split in "${splits[@]}"; do
        for use_cat_list in "${use_cat_lists[@]}"; do
            python classification_inference.py \
                --model_root "$MODEL_ROOT" \
                --base_model "$BASE_MODEL" \
                --exp_name "$EXP_NAME" \
                --checkpoint "$CHECKPOINT" \
                --zero_shot "$zero_shot" \
                --eval_type "$eval_type" \
                --data_root "$DATA_ROOT" \
                --dataset "$dataset" \
                --split "$split" \
                --max_new_tokens "$max_new_tokens" \
                --use_cat_list "$use_cat_list"
        done
    done
done