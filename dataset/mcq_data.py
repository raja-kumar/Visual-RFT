## I have the dataset. -> done

## generated the base novel category list -> done

## correct the category name in train/eval data -> done

## run the step1 and generate the output

## post process step1 output to make it mcq and in the required dataset format

'''

import json
import re

base_path = "/home/raja/OVOD/git_files/VLM-COT/data/oxford_flowers/zero_shot/subsample_base_train.json"
novel_path = "/home/raja/OVOD/git_files/VLM-COT/data/oxford_flowers/zero_shot/subsample_new_test.json"

with open(base_path, "r") as f:
    base_data = json.load(f)

with open(novel_path, "r") as f:
    novel_data = json.load(f)

base_classes = set()
novel_classes = set()

for item in base_data:
    base_classes.add(re.search(r'<answer>(.*?)</answer>', item["solution"]).group(1).strip())

for item in novel_data:
    novel_classes.add(re.search(r'<answer>(.*?)</answer>', item["solution"]).group(1).strip())

print("Base Classes:", base_classes)
print("Novel Classes:", novel_classes)

with open('base_classes.txt', 'w') as f:
    for class_name in list(base_classes):
        f.write(f"{class_name}\n")

# Write novel classes to file
with open('novel_classes.txt', 'w') as f:
    for class_name in list(novel_classes):
        f.write(f"{class_name}\n")

print(f"Saved {len(base_classes)} base classes to base_classes.txt")
print(f"Saved {len(novel_classes)} novel classes to novel_classes.txt")

'''

import json

novel_path = "/home/raja/OVOD/git_files/VLM-COT/data/oxford_flowers/zero_shot/subsample_new_test.json"

with open(novel_path, "r") as f:
    novel_data = json.load(f)

for item in novel_data:
    item["solution"] = item["solution"].replace("<answer>watercress</answer>", "<answer>Nasturtium</answer>")

with open(novel_path, "w") as f:
    json.dump(novel_data, f, indent=4)


