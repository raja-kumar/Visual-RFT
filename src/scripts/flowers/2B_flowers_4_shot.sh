cd /app/Visual-RFT/src/virft/

export DEBUG_MODE="true"
export LOG_PATH="./logs/debug_log_2b_GRPO_flowers_4shot.txt"

export DATA_PATH=/app/Visual-RFT/data/ViRFT_CLS_flower_4_shot
export CKPT_PATH="Qwen/Qwen2-VL-2B-Instruct"
export SAVE_PATH=./ckpts/Qwen2-VL-2B-Instruct_GRPO_flowers_4_shot


torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_classification.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 6 \
    --run_name Qwen2-VL-2B_GRPO_flowers_4shot \
    --save_steps 50 \
    --save_only_model true \
    --num_generations 4 \
    --deepspeed local_scripts/zero3_offload.json \
