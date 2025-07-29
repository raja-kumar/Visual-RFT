#!/bin/bash
# filepath: /home/raja/gdrive/Visual-RFT/classification/run_llm_eval.sh

# Array of JSON files to evaluate

OUTPUT_FILE_ROOT="/app/Visual-RFT/src/passk/output/oxford_flowers/topk_accuracy/rft"
OUTPUT_FILES=("${OUTPUT_FILE_ROOT}/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_checkpoint-300_subsample_new_test_False_1_1.0.json"
"${OUTPUT_FILE_ROOT}/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_checkpoint-300_subsample_new_test_False_5_1.0.json"
    "${OUTPUT_FILE_ROOT}/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_checkpoint-300_subsample_new_test_False_20_1.0.json"
    "${OUTPUT_FILE_ROOT}/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_checkpoint-300_subsample_base_val_False_1_1.0.json"
    "${OUTPUT_FILE_ROOT}/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_checkpoint-300_subsample_base_val_False_5_1.0.json"
    "${OUTPUT_FILE_ROOT}/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_checkpoint-300_subsample_base_val_False_20_1.0.json"
)
# Loop through each JSON file
for OUTPUT_FILE in "${OUTPUT_FILES[@]}"; do
    # Extract the base name of the file (without path and extension)
    echo "Running evaluation for: $OUTPUT_FILE"
    # Run the Python script with the current JSON file
    python llm_eval.py --output_file "$OUTPUT_FILE" --one_answer "False"
done