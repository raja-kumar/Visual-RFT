from argparse import Namespace
from oxford_flowers import OxfordFlowers
from oxford_pets import OxfordPets

if __name__ == "__main__":
    seed = 42
    num_shots = 0
    phase = 'train'
    few_shot = False
    dataset_args = Namespace(
            SEED=seed,
            NUM_SHOTS=num_shots,
            SUBSAMPLE_CLASSES='new' if phase == 'test' else 'base',
    )

    if (few_shot):
        dataset_args.SUBSAMPLE_CLASSES = 'all'
    
    print (dataset_args)
    dataset = OxfordFlowers(cfg=dataset_args)

    # print(f"Number of training samples: {len(dataset.train_x)}")
    # print(f"Number of validation samples: {len(dataset.val)}")
    # print(f"Number of test samples: {len(dataset.test)}")

