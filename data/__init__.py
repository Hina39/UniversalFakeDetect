import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import RealFakeDataset


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(opt, preprocess=None):
    # shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = RealFakeDataset(opt)
    if "2b" in opt.arch:
        dataset.transform = preprocess
    # sampler = get_bal_sampler(dataset) if opt.class_bal else None

    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=shuffle,
    #     sampler=sampler,
    #     num_workers=int(opt.num_threads),
    # )

    # Create a list of indices from 0 to the length of the dataset
    indices = list(range(len(dataset)))

    # Shuffle the indices
    np.random.shuffle(indices)
    train_indices = indices[:48600]
    test_indices = indices[48600:48600+1350]

    if opt.isTrain:
        # Get the first 48600 indices
        print("Using 48600 indices")
        indices = train_indices
    else:
        # Get the last 1350 indices
        print("Using 1350 indices")
        indices = test_indices
    # # Get the first 35000 indices
    # indices = indices[:35000]

    # Create a SubsetRandomSampler
    sampler = torch.utils.data.SubsetRandomSampler(indices)

    # Create the DataLoader with the sampler
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # We're already shuffling with the sampler
        sampler=sampler,
        num_workers=int(opt.num_threads),
    )
    return data_loader
