#!/bin/bash

MODEL_ROOT="/app/saved_models/vrft/ckpts/"  # root path for saved models
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
EXP_NAME="Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_and_hard_mcq"  # experiment name for saving models
CHECKPOINT="checkpoint-400"  # checkpoint name for saved models


# ==== configurations ====
zero_shot="True"
eval_type="baseline"  # "sft" or everything else
use_cat_list="False"

# ==== dataset and output paths ====
DATA_ROOT="/data2/raja/"
dataset="oxford_flowers"  # oxford_flowers, oxford-iiit-pet, CUB_200_2011

## === generation settings ===
temperature=1.0
max_new_tokens=512

splits=("new_test")  # splits to evaluate on
# eval_types = ("baseline" "rft" "rft_mcq")  # evaluation types
num_return_sequences=(1 5 10)  # number of sequences to return

for split in "${splits[@]}"; do
    for NSEQ in "${num_return_sequences[@]}"; do
        python topk_accuracy.py \
            --model_root "$MODEL_ROOT" \
            --base_model "$BASE_MODEL" \
            --exp_name "$EXP_NAME" \
            --checkpoint "$CHECKPOINT" \
            --zero_shot "$zero_shot" \
            --eval_type "$eval_type" \
            --use_cat_list "$use_cat_list" \
            --data_root "$DATA_ROOT" \
            --dataset "$dataset" \
            --split "$split" \
            --num_return_sequences "$NSEQ" \
            --temperature "$temperature" \
            --max_new_tokens "$max_new_tokens" 
    done
done