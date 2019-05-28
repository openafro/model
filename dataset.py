from collections import namedtuple
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class OpenAfroDatasetSplit(Dataset):
    def __init__(self, hdf5_dataset, split):
        self.X = hdf5_dataset["X_" + split]
        self.y = hdf5_dataset["y_" + split]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx]))

OpenAfroDataset = namedtuple("OpenAfroDataset", [
    "train",
    "val",
    "test",
    "input_bands",
    "output_bands",
    "input_size",
    "output_size"
])

def load_dataset(path):
    f = h5py.File(path, "r")

    return OpenAfroDataset(
        OpenAfroDatasetSplit(f, "train"),
        OpenAfroDatasetSplit(f, "val"),
        OpenAfroDatasetSplit(f, "test"),
        f.attrs.get("input_bands"),
        f.attrs.get("output_bands"),
        f.attrs.get("input_size"),
        f.attrs.get("output_size"),
    )
