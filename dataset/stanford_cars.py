import os
import pickle
from scipy.io import loadmat
import json

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class StanfordCars(DatasetBase):

    dataset_folder = "stanford_cars"

    def __init__(self, cfg):
        # root = os.path.abspath(os.path.expanduser(DATA_FOLDER))
        root = "/home/raja/OVOD/git_files/VLM-COT/data"
        self.dataset_dir = os.path.join(root, self.dataset_folder)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.zero_shot_dir = os.path.join(self.dataset_dir, "zero_shot")
        self.few_shot_dir = os.path.join(self.dataset_dir, "fewshot")
        mkdir_if_missing(self.zero_shot_dir)
        mkdir_if_missing(self.split_fewshot_dir)
        mkdir_if_missing(self.few_shot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")
            trainval = self.read_data("cars_train", trainval_file, meta_file)
            test = self.read_data("cars_test", test_file, meta_file)
            train, val = OxfordPets.split_trainval(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

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

        subsample = cfg.SUBSAMPLE_CLASSES
        subsample_data, category_data = OxfordPets.subsample_classes(train, val, test, dataset_name=self.dataset_folder, subsample=subsample)
        train, val, test = subsample_data
        categories, idx_to_class, class_to_idx = category_data

        with open(os.path.join(self.zero_shot_dir, f"{cfg.SUBSAMPLE_CLASSES}_categories.txt"), "w") as f:
            for cat in categories:
                f.write(f"{cat}\n")
        
        with open(os.path.join(self.zero_shot_dir, f"{cfg.SUBSAMPLE_CLASSES}_idx_to_class.json"), "w") as f:
            json.dump(idx_to_class, f, indent=4)
        
        with open(os.path.join(self.zero_shot_dir, f"{cfg.SUBSAMPLE_CLASSES}_class_to_idx.json"), "w") as f:
            json.dump(class_to_idx, f, indent=4)
        
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
            processed_path_test = os.path.join(self.zero_shot_dir, f"subsample_{subsample}_test.json")
            with open(processed_path_test, "w") as f:
                json.dump(test, f, indent=4)
            
            processed_path_val = os.path.join(self.zero_shot_dir, f"subsample_{subsample}_val.json")
            with open(processed_path_val, "w") as f:
                json.dump(val, f, indent=4)

        # super().__init__(train_x=train, val=val, test=test)

    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)

        return items