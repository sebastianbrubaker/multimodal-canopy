import sys
import os
import warnings
import numpy as np
from rasterio.enums import Resampling
import rioxarray as rxr
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt

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


def compute_valid_mask(arr):
    """
    Returns a copy of the original array with a binary valid mask 
    (1: valid, 0: invalid) for a spatially aligned array of shape (C, H, W) and infs filled.
    """
    ret_arr = arr.copy()
    ret_arr[np.isinf(ret_arr)] = np.nan
    mask_valid = (~np.isnan(ret_arr).any(axis=0)).astype(np.float32)

    return np.concatenate([ret_arr, mask_valid[np.newaxis, :, :]], axis=0)



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


def fill_features_bilinear(stack: np.ndarray, num_feature_channels: int = 11) -> np.ndarray:
    """
    Performs 2D bilinear spatial interpolation to fill NaNs and Infs on feature channels 
    (indices 0 to num_feature_channels - 1) of a (C, H, W) array.
    
    Leaves target channels (indices num_feature_channels to C - 1) un-interpolated.
    """
    out_stack = stack.copy()
    
    # Convert Infs in features to NaNs first
    features = out_stack[:num_feature_channels]
    features[np.isinf(features)] = np.nan
    
    C_feat, H, W = features.shape
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    
    for c in range(C_feat):
        band = features[c]
        nan_mask = np.isnan(band)
        
        # Skip channel if there are no missing values
        if not np.any(nan_mask):
            continue
            
        # If the entire channel is missing, default fill to 0.0
        if np.all(nan_mask):
            features[c] = 0.0
            continue
            
        # Coordinates of valid pixels
        valid_coords = np.array(np.nonzero(~nan_mask)).T  # Shape: (N_valid, 2)
        valid_values = band[~nan_mask]                     # Shape: (N_valid,)
        
        # Coordinates of missing pixels to fill
        missing_coords = (grid_y[nan_mask], grid_x[nan_mask])
        
        # Perform 2D Linear (Bilinear on regular grid) Interpolation
        interpolated = griddata(
            points=valid_coords,
            values=valid_values,
            xi=missing_coords,
            method='linear'
        )
        
        # Assign interpolated values back to missing positions
        band[nan_mask] = interpolated
        
        # Fallback for boundary NaNs (griddata returns NaN for points outside the convex hull of valid pixels)
        still_nan = np.isnan(band)
        if np.any(still_nan):
            # Nearest valid neighbor via Euclidean Distance Transform
            indices = distance_transform_edt(still_nan, return_distances=False, return_indices=True)
            band[still_nan] = band[tuple(indices)][still_nan]
            
        features[c] = band
        
    out_stack[:num_feature_channels] = features
    return out_stack


def main():
    if len(sys.argv) < 4:
        print("Usage: python process_rasters.py <metrics_dir> <gee_dir> <out_dir>")
        sys.exit(1)

    metrics_dir = sys.argv[1]
    gee_dir = sys.argv[2]
    out_dir = sys.argv[3]

    stacks_dict = align_rasters(metrics_dir, gee_dir)

    # Compute valid masks in place
    for name, stack in stacks_dict.items():
        stack_filled = fill_features_bilinear(stack, num_feature_channels=11)
        stacks_dict[name] = compute_valid_mask(stack_filled)

    write_np(out_dir, stacks_dict)


if __name__ == "__main__":
    main()