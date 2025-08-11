#!/bin/bash
# filepath: /home/raja/gdrive/Visual-RFT/classification/run_llm_eval.sh

# Array of JSON files to evaluate

OUTPUT_FILE_ROOT="/app/Visual-RFT/classification/output/oxford_flowers/two_steps"
OUTPUT_FILES=("${OUTPUT_FILE_ROOT}/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_qwen_mcq_checkpoint-400_base_val.json" 
                "${OUTPUT_FILE_ROOT}/Qwen2.5-VL-7B-Instruct_base_val.json"
)
# Loop through each JSON file
for OUTPUT_FILE in "${OUTPUT_FILES[@]}"; do
    # Extract the base name of the file (without path and extension)
    echo "Running evaluation for: $OUTPUT_FILE"
    # Run the Python script with the current JSON file
    python llm_eval.py --output_file "$OUTPUT_FILE" --one_answer "True"
done