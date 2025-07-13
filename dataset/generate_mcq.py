from tqdm import tqdm
import random
import re
import json
import textwrap

def generate_mcq_data(top5_pred_file, json_data_path, data_name):
    
    with open(top5_pred_file, 'r') as f:
        top5_data = json.load(f)
    
    with open(json_data_path, 'r') as f:
        json_data = json.load(f)
    
    print(f"Top 5 data length: {len(top5_data)}")
    print(f"JSON data length: {len(json_data)}")
    # assert len(top5_data) == len(json_data), "Top 5 data and JSON data length mismatch"
    
    dataset = []
    count = 0
    for item in tqdm(json_data, total=len(json_data)):

        image_path = item["image_path"]
        gt_cat_name = re.search("<answer>(.*?)</answer>", item["solution"]).group(1)

        image_id = image_path.split("/")[-1].split(".")[0]
        # print(f"{count}: {image_id}")
        try:
            curr_data = top5_data[image_id]
            # print(f"curr_data: {curr_data}")
        except Exception as e:
            # print(f"Error processing image_id {image_id}: {e}")
            continue
        gpt_preds = curr_data["gpt_pred"]
        gpt_labels = curr_data["pred_labels"]
        gt_label = curr_data["gt_label"]        

        if len(gpt_preds) > 5:
            gpt_preds = gpt_preds[:5]
            gpt_labels = gpt_labels[:5]

        if gt_label == -1 or (not gt_label in gpt_labels) or (len(gpt_labels) == 0):
            gpt_preds[-1] = gt_cat_name # replace the last option with the ground truth category name if it is not present in the predictions
        
        random.shuffle(gpt_preds)
        
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']
        answer = -1

        # print(f"Processing image_id: {image_id}, gt_cat_name: {gt_cat_name}, gpt_preds: {gpt_preds}")
        
            
        options = "\n".join([f"{letters[i]}. {option}" for i, option in enumerate(gpt_preds)])
        for option in gpt_preds:
            if option.lower().replace("-", " ") == gt_cat_name.lower():
                answer = letters[gpt_preds.index(option)]
                break
        
        if answer == -1:
            print(f"gt_cat_name {gt_cat_name}, gpt_preds {gpt_preds}")
            count += 1
            continue
        
        prompt = f""" This is an image containing a {data_name}. Please find the most likely {data_name} in the image from the below options.
{options}
Please output the letter corresponding to the correct {data_name} name.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:
<think> ... </think> <answer>option letter</answer>
Please strictly follow the format. """
        data_json = {
                    "image_path": image_path,
                    "problem": prompt,
                    "solution": f"<answer>{answer}</answer>"
                }
        
        # print(f"data_json: {data_json}")
        
        dataset.append(data_json)
    random.shuffle(dataset)
    print(f"Total count of skipped images: {count}")
    return dataset

if __name__ == "__main__":

    data_root = "/home/raja/OVOD/git_files/VLM-COT/data"
    data= "CUB200"
    data_folder = "CUB_200_2011"
    split = "new"
    phase = "val"  # "train" or "val" or "test"
    data_name = "bird"


    top5_pred_file = f"/home/raja/OVOD/git_files/VLM-COT/outputs/{data}/{data}_step1_baseline_{split}_{phase}_gemini-2.5-flash-lite-preview-06-17_cat_True.json"
    json_data_path = f"{data_root}/{data_folder}/zero_shot/subsample_{split}_{phase}.json"
    output_path = f"{data_root}/{data_folder}/zero_shot/subsample_{split}_{phase}_mcq.json"

    dataset = generate_mcq_data(top5_pred_file, json_data_path, data_name)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Dataset saved to {output_path}")
    print(f"Generated MCQ dataset with {len(dataset)} items.")



