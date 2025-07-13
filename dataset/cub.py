import os
import pickle
import json
import pandas as pd

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class CUB200(DatasetBase):

    dataset_folder = "CUB_200_2011"

    def __init__(self, cfg):
        root = "/home/raja/OVOD/git_files/VLM-COT/data/"
        self.dataset_dir = os.path.join(root, self.dataset_folder)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.zero_shot_dir = os.path.join(self.dataset_dir, "zero_shot")
        self.few_shot_dir = os.path.join(self.dataset_dir, "fewshot")
        mkdir_if_missing(self.zero_shot_dir)
        mkdir_if_missing(self.split_fewshot_dir)
        mkdir_if_missing(self.few_shot_dir)

        train, test = self.read_data()
        train, val = OxfordPets.split_trainval(train)
        OxfordPets.save_split(train, val, test, os.path.join(self.dataset_dir, "split_zhou_CUB200.json"), self.dataset_dir)

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

    def read_data(self):

        images = pd.read_csv(os.path.join(self.dataset_dir, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.dataset_dir, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        bb_labels = pd.read_csv(os.path.join(self.dataset_dir, 'bounding_boxes.txt'), sep=' ',
                                                names=['img_id', 'min_x', 'min_y', 'w', 'h'])
        train_test_split = pd.read_csv(os.path.join(self.dataset_dir, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')
        data = data.merge(bb_labels, on='img_id')
        data['class_name'] = data['filepath'].apply(lambda x: x.split('/')[0].split(".")[-1].replace('_', ' ') + '.')

        train, test = data[data.is_training_img == 1], data[data.is_training_img == 0]
        
        training_data = zip(train["filepath"].tolist(), train["target"].tolist(), train["class_name"].tolist())
        test_data = zip(test["filepath"].tolist(), test["target"].tolist(), test["class_name"].tolist())

        train = [Datum(impath=os.path.join(self.dataset_dir, "images", fp), label=target-1, classname=class_name) for fp, target, class_name in training_data]
        test = [Datum(impath=os.path.join(self.dataset_dir, "images", fp), label=target-1, classname=class_name) for fp, target, class_name in test_data]
        
        return train, test