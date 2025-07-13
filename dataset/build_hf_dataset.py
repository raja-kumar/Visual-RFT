import time
from datasets import DatasetDict, Dataset
from PIL import Image
import json
import random

def json_to_dataset(json_file_path):
    # read json file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    random.seed(42)  # Set a seed for reproducibility
    random.shuffle(data)  # Shuffle the data

    image_paths = [item['image_path'].replace("/home/raja/OVOD/git_files/VLM-COT/data/", "/data/raja/") for item in data]
    problems = [item['problem'] for item in data]
    solutions = [item['solution'] for item in data]

    images = [Image.open(image_path).convert('RGBA') for image_path in image_paths]

    dataset_dict = {
        'image': images,
        'problem': problems,
        'solution': solutions
    }

    # print(dataset_dict)

    dataset = Dataset.from_dict(dataset_dict)
    dataset_dict = DatasetDict({
        'train': dataset
    })
    return dataset_dict

def save_dataset(dataset_dict, save_path):
    # save DatasetDict to your disk
    dataset_dict.save_to_disk(save_path)

def load_dataset(save_path):
    # load DatasetDict
    return DatasetDict.load_from_disk(save_path)

if __name__ == "__main__":
    # json_file_path = "/home/raja/OVOD/git_files/VLM-COT/data/oxford_flowers/fewshot/1_shots_base_and_hard_train_mcq.json"
    # output_path = "/home/raja/OVOD/git_files/VLM-COT/data/oxford_flowers/fewshot/1_shots_base_and_hard_train_mcq_dataset"

    # json_file_path = "/data2/datasets/oxford-iiit-pet/zero_shot/subsample_base_train_hard_mcq.json"
    # json_file_path = "/data/raja/fgvc_aircraft/fgvc_aircraft/zero_shot/subsample_base_train_mcq.json"
    # json_file_path = "/data/raja/stanford_cars/zero_shot/subsample_base_train_mcq.json"
    json_file_path = "/data/raja/CUB_200_2011/zero_shot/subsample_base_train_mcq.json"
    output_path = json_file_path.replace('.json', '_dataset')
    print("output_path:", output_path)

    dataset_dict = json_to_dataset(json_file_path)
    save_dataset(dataset_dict, output_path)

    ## test if the dataset is saved correctly
    test_dataset_dict = load_dataset(output_path)
    print(test_dataset_dict)

