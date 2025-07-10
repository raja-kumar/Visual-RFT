import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing
import json
# from flags import DATA_FOLDER

@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):

    dataset_dir = "oxford-iiit-pet"

    def __init__(self, cfg):
        # root = os.path.abspath(os.path.expanduser(DATA_FOLDER))
        root = "/home/raja/OVOD/git_files/VLM-COT/data"
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.zero_shot_dir = os.path.join(self.dataset_dir, "zero_shot")
        self.few_shot_dir = os.path.join(self.dataset_dir, "fewshot")
        mkdir_if_missing(self.zero_shot_dir)
        mkdir_if_missing(self.split_fewshot_dir)
        mkdir_if_missing(self.few_shot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # subsample = cfg.SUBSAMPLE_CLASSES
        # train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        subsample = cfg.SUBSAMPLE_CLASSES
        subsample_data, categories = self.subsample_classes(train, val, test, subsample=subsample)
        train, val, test = subsample_data

        with open(os.path.join(self.zero_shot_dir, f"{cfg.SUBSAMPLE_CLASSES}_classes.txt"), "w") as f:
            for cat in categories:
                f.write(f"{cat}\n")
        
        if (cfg.SUBSAMPLE_CLASSES == "base" or cfg.SUBSAMPLE_CLASSES == "all"):

            if (num_shots >= 1):
                processed_path_train = os.path.join(self.few_shot_dir, f"{num_shots}_shots_{subsample}_train.json")
                processed_path_val = os.path.join(self.few_shot_dir, f"{num_shots}_shots_{subsample}_val.json")
            else:
                processed_path_train = os.path.join(self.zero_shot_dir, f"subsample_{subsample}_train.json")
                processed_path_val = os.path.join(self.zero_shot_dir, f"subsample_{subsample}_val.json")

            with open(processed_path_train, "w") as f:
                json.dump(train, f, indent=4)
            
            with open(processed_path_val, "w") as f:
                json.dump(val, f, indent=4)
        
        elif (cfg.SUBSAMPLE_CLASSES == "new"):
            processed_path_val = os.path.join(self.zero_shot_dir, f"subsample_{subsample}_val.json")
            with open(processed_path_val, "w") as f:
                json.dump(val, f, indent=4)


        # super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        categories = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        print(f"Selected {len(selected)} classes: {selected}")
        
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        json_output = []
        for dataset in args:
            dataset_new = []
            dataset_json = []
            for item in dataset:
                if item.label not in selected:
                    continue
                data_json = {
                    "image_path": item.impath,
                    "problem": """ This is an image containing a pet. Please identify the species of the pet based on the image.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:
<think> ... </think> <answer>species name</answer>
Please strictly follow the format. """,
                    "solution": f"<answer>{item.classname}</answer>"
                }
                dataset_json.append(data_json)
                item_new = Datum(
                    impath=item.impath,
                    label=item.label,
                    classname=item.classname
                )
                dataset_new.append(item_new)
                categories.add(item.classname)
            output.append(dataset_new)
            json_output.append(dataset_json)
        
        return json_output, categories