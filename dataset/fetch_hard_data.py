import json

# File paths
data_folder = "fgvc_aircraft"
file_name = "Qwen2_5-VL-7B-Instruct_GRPO_aircraft_base_mcq_checkpoint-400_subsample_base_train_mcq.json"
data_root = "/data2/raja/"

file1_path = f"{data_root}/{data_folder}/zero_shot/subsample_base_train_mcq.json"  # original training data
file2_path = f"/home/raja/gdrive/LLaMA-Factory/evaluation/classification/output/{data_folder}/rft_mcq/{file_name}"  # output on training data using trained model
output_path = f"{data_root}/{data_folder}/zero_shot/subsample_base_train_hard_mcq.json"

# Load the JSON files
with open(file1_path, "r") as file1:
    first_json = json.load(file1)

with open(file2_path, "r") as file2:
    second_json = json.load(file2)

# Find mismatched entries
hard_examples = []
for key, value in second_json.items():
    if value.get("groundtruth") != value.get("answer"):
        # Extract the corresponding entry from the first JSON
        image_key = f"{key}.jpg"  # Assuming the key corresponds to the image file name
        for entry in first_json:
            if image_key in entry.get("image_path", ""):
                hard_examples.append(entry)
                break

# Save the results to a new JSON file
with open(output_path, "w") as output_file:
    json.dump(hard_examples, output_file, indent=4)

print(f"Hard examples saved to {output_path}")