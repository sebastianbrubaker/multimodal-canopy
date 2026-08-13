import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CanopyDataset(Dataset):
    def __init__(self, stack_dir, feature_range, target_idx):
        self.stack_paths = sorted([os.path.join(stack_dir, f) for f in os.listdir(stack_dir) if f.endswith(".npy")])
        self.feature_range = feature_range
        self.target_idx = target_idx

    def __len__(self):
        return len(self.stack_paths)


    def __getitem__(self, idx):
        arr = np.load(self.stack_paths[idx]).astype(np.float32)

        # Split into X, y
        X = arr[self.feature_range[0]:self.feature_range[1], :, :]
        y = arr[self.target_idx, :, :]
        y = y[np.newaxis, :, :]    # expand dims (H, W) -> (C, H, W)

        # Compute valid mask (valid: 1, invalid: 0) then fill NaNs
        valid = np.where(np.isnan(y), 0, 1).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0)   # 0.0 as BC LiDAR water-masks 

        return (
            torch.from_numpy(X), 
            torch.from_numpy(y),
            torch.from_numpy(valid)
        ) 