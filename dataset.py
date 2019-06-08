from collections import namedtuple
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class OpenAfroDatasetSplit(Dataset):
    def __init__(self, hdf5_dataset, split, transform=lambda x: x):
        self.X = hdf5_dataset["X_" + split]
        self.y = hdf5_dataset["y_" + split]
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.transform(torch.from_numpy(self.X[idx]).to(dtype=torch.float32)),
                torch.from_numpy(self.y[idx]))

OpenAfroDataset = namedtuple("OpenAfroDataset", [
    "train",
    "val",
    "test",
    "input_bands",
    "output_bands",
    "input_size",
    "output_size"
])

def load_dataset(path, normalize=True, exclude_empty=False):
    f = h5py.File(path, "r")

    if normalize:
        t = T.Normalize(torch.from_numpy(f.attrs.get("train_mean")).to(dtype=torch.float32),
                        torch.from_numpy(f.attrs.get("train_std")).to(dtype=torch.float32))
    else:
        t = lambda x: x

    return OpenAfroDataset(
        OpenAfroDatasetSplit(f, "train", t),
        OpenAfroDatasetSplit(f, "val", t),
        OpenAfroDatasetSplit(f, "test", t),
        f.attrs.get("input_bands"),
        f.attrs.get("output_bands"),
        f.attrs.get("input_size"),
        f.attrs.get("output_size"),
    )
