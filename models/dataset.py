import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CanopyDataset(Dataset):
    def __init__(self, feature_dir, target_dir):
        self.feature_paths = sorted([os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".tif")])
        self.target_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".tif")])

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        x = np.load(self.feature_paths[idx]).astype(np.float32)
        y = np.load(self.target_paths[idx]).astype(np.float32)

        return (
            torch.from_numpy(x), 
            torch.from_numpy(y), 
        )