import json
import math
import random
import os

def combine_json_files(json_file_paths, output_folder, target_total=2400, output_filename="combined_mcq_2400.json"):
    """
    Combines multiple JSON files into one, sampling proportionally from each file.

    Args:
        json_file_paths (list): List of JSON file paths.
        output_folder (str): Folder to save the combined JSON.
        target_total (int): Approximate total number of samples in the output.
        output_filename (str): Name of the output JSON file.
    """
    all_data = []
    file_entries = []
    for file in json_file_paths:
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            file_entries.append(len(data))
            all_data.append(data)

    total_entries = sum(file_entries)
    samples_per_file = [
        max(1, math.floor(target_total * (count / total_entries)))
        for count in file_entries
    ]

    # Adjust to match exactly target_total if needed
    diff = target_total - sum(samples_per_file)
    for i in range(abs(diff)):
        idx = i % len(samples_per_file)
        if diff > 0:
            samples_per_file[idx] += 1
        elif samples_per_file[idx] > 1:
            samples_per_file[idx] -= 1

    combined = []
    for data, n in zip(all_data, samples_per_file):
        if len(data) <= n:
            combined.extend(data)
        else:
            combined.extend(random.sample(data, n))
    
    random.shuffle(combined)
    print(f"Total combined entries: {len(combined)}")

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=4)
    print(f"Combined JSON saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    json_files = [
        "/data2/raja/oxford_flowers/qwen_mcq/subsample_base_train_pass_20_mcq.json",
        "/data2/raja/oxford-iiit-pet/qwen_mcq/subsample_base_train_pass_20_mcq.json",
        "/data2/raja/stanford_cars/qwen_mcq/subsample_base_train_pass_20_mcq.json",
        "/data2/raja/fgvc_aircraft/qwen_mcq/subsample_base_train_pass_20_mcq.json",
        "/data2/raja/CUB_200_2011/qwen_mcq/subsample_base_train_pass_20_mcq.json"
    ]
    output_dir = "/data2/raja/combined_datasets"
    combine_json_files(json_files, output_dir)