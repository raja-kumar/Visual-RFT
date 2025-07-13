from argparse import Namespace
from oxford_flowers import OxfordFlowers
from oxford_pets import OxfordPets
from stanford_cars import StanfordCars
from fgvc_aircraft import FGVCAircraft
from cub import CUB200

if __name__ == "__main__":
    seed = 42
    num_shots = 0
    phase = 'test'
    dataset_args = Namespace(
            SEED=seed,
            NUM_SHOTS=num_shots,
            SUBSAMPLE_CLASSES='new' if phase == 'test' else 'base',
    )
    
    print (dataset_args)
    # dataset = OxfordFlowers(cfg=dataset_args)
    # dataset = OxfordPets(cfg=dataset_args)
    # dataset = StanfordCars(cfg=dataset_args)
    # dataset = FGVCAircraft(cfg=dataset_args)
    dataset = CUB200(cfg=dataset_args)

    # print(f"Number of training samples: {len(dataset.train_x)}")
    # print(f"Number of validation samples: {len(dataset.val)}")
    # print(f"Number of test samples: {len(dataset.test)}")

