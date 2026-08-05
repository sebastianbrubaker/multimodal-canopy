import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CanopyDataset(Dataset):
    def __init__(self, stack_dir):
        self.stack_paths = sorted([os.path.join(stack_dir, f) for f in os.listdir(stack_dir) if f.endswith(".npy")])
        
    def __len__(self):
        return len(self.stack_paths)

    def __getitem__(self, idx):
        # Load in NumPy array
        arr = np.load(self.stack_paths[idx]).astype(np.float32)
        # TODO:
        # Compute valid mask

        # Split into X, Y, val and return as PyTorch Tensors

        return (
            torch.from_numpy(x), 
            torch.from_numpy(y), 
        )