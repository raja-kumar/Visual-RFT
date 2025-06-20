import os
import pickle
import random
from scipy.io import loadmat
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing
import json

from oxford_pets import OxfordPets
# from flags import DATA_FOLDER


@DATASET_REGISTRY.register()
class OxfordFlowers(DatasetBase):

    dataset_dir = "/home/raja/OVOD/git_files/VLM-COT/data/oxford_flowers"

    def __init__(self, cfg):
        # root = os.path.abspath(os.path.expanduser(DATA_FOLDER))
        root = "/home/raja/OVOD/git_files/VLM-COT/data"
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.zero_shot_dir = os.path.join(self.dataset_dir, "zero_shot")
        mkdir_if_missing(self.zero_shot_dir)
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_data()
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

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
        # train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        subsample = cfg.SUBSAMPLE_CLASSES
        subsample_data, categories = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        train, val, test = subsample_data

        with open(os.path.join(self.zero_shot_dir, f"{cfg.SUBSAMPLE_CLASSES}_categories.json"), "w") as f:
            for cat in categories:
                f.write(f"{cat}\n")

        if (cfg.SUBSAMPLE_CLASSES == "base" or cfg.SUBSAMPLE_CLASSES == "all"):
            processed_path_train = os.path.join(self.zero_shot_dir, f"subsample_{subsample}_train.json")
            processed_path_val = os.path.join(self.zero_shot_dir, f"subsample_{subsample}_val.json")

            with open(processed_path_train, "w") as f:
                json.dump(train, f, indent=4)
            
            with open(processed_path_val, "w") as f:
                json.dump(val, f, indent=4)
        
        elif (cfg.SUBSAMPLE_CLASSES == "new"):
            processed_path_test = os.path.join(self.zero_shot_dir, f"subsample_{subsample}_test.json")
            with open(processed_path_test, "w") as f:
                json.dump(test, f, indent=4)

        # super().__init__(train_x=train, val=val, test=test)

    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y - 1, classname=c)  # convert to 0-based label
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train : n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val :], label, cname))

        return train, val, test