import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py


class H5Dataset(Dataset):
    def __init__(self, h5_paths, limit=-1):
        self.limit = limit
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self.indices = {}
        idx = 0
        for a, archive in enumerate(self.archives):
            for i in range(len(archive)):
                self.indices[idx] = (a, i)
                idx += 1

        self._archives = None

    @property
    def archives(self):
        if self._archives is None:
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __getitem__(self, index):
        a, i = self.indices[index]
        archive = self.archives[a]
        data = archive[f"pts"]
        labels = archive[f"label"]

        return torch.Tensor(data), torch.Tensor(labels)

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)