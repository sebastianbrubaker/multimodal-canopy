import sys
import os
import warnings
import numpy as np
from rasterio.enums import Resampling
import rioxarray as rxr

TARGET_H, TARGET_W = 128, 128

def align_rasters(metrics_dir, gee_dir) -> dict[str, np.ndarray]:
    """
    Reads and aligns GeoTIFF pairs, clips to a globally specified spatial window,
    then returns them as a dict[bcgs_tile_name, data].
    """
    metric_fps = [os.path.join(metrics_dir, f) for f in os.listdir(metrics_dir) if f.endswith(".tif")]
    stacks = {}

    # Iterate directories and align stacks
    for metric_fp in metric_fps:
        bcgs_tile = os.path.splitext(os.path.basename(metric_fp))[0]
        
        gee_path = os.path.join(gee_dir, f"{bcgs_tile}.tif")
        if not os.path.exists(gee_path):
            warnings.warn(f"No matching GEE file found for tile '{bcgs_tile}'. Skipping.")
            continue
            
        # Read rasters
        metrics_ds = rxr.open_rasterio(metric_fp, masked=True)
        gee_ds = rxr.open_rasterio(gee_path, masked=True)
        
        # Align GEE feature stack to target CRS/resolution/grid
        gee_aligned = gee_ds.rio.reproject_match(metrics_ds, resampling=Resampling.bilinear)
        
        # Extract spatial dimensions (H, W)
        _, h_target, w_target = metrics_ds.shape
        _, h_feat, w_feat = gee_aligned.shape
        
        # Check minimum spatial dimensions
        if h_target < TARGET_H or w_target < TARGET_W or h_feat < TARGET_H or w_feat < TARGET_W:
            warnings.warn(
                f"Tile '{bcgs_tile}' is smaller than required ({TARGET_H}x{TARGET_W}). "
                f"Target: ({h_target}, {w_target}), Features: ({h_feat}, {w_feat}). Skipping tile."
            )
            metrics_ds.close()
            gee_ds.close()
            continue
            
        # Crop down to exact global (128, 128) window starting at top-left min(x), min(y)
        targets_cropped = metrics_ds.values[:, :TARGET_H, :TARGET_W].astype(np.float32)
        features_cropped = gee_aligned.values[:, :TARGET_H, :TARGET_W].astype(np.float32)

        # Clean up raster memory handles explicitly
        metrics_ds.close()
        gee_ds.close()
        
        # Stack along channel and store to dict
        stacks[bcgs_tile] = np.concatenate([features_cropped, targets_cropped], axis=0)

    return stacks


def standarize(arr, axis) -> np.ndarray:
    """
    Perform z-score standardization across a specified axis or axes.
    """ 
    eps = 1e-8

    mean = np.nanmean(arr, axis=axis, keepdims=True)
    std = np.nanstd(arr, axis=axis, keepdims=True)

    return (arr - mean) / (std + eps)


def write_np(out_dir: str, stacks: dict[str, np.ndarray]):
    """
    Writes the stacks.values() NumPy arrays as .npy binaries to out_dir using the
    corresponding stacks.keys() as file names.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Iterate stacks and write to file
    for name, arr in stacks.items():
        dst = os.path.join(out_dir, f"{name}.npy")
        np.save(dst, arr)


def main():
    if len(sys.argv) < 4:
        print("Usage: python process_rasters.py <metrics_dir> <gee_dir> <out_dir>")
        sys.exit(1)

    metrics_dir = sys.argv[1]
    gee_dir = sys.argv[2]
    out_dir = sys.argv[3]

    write_np(out_dir, align_rasters(metrics_dir, gee_dir))


if __name__ == "__main__":
    main()