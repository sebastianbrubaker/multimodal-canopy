import sys
import os
import warnings
import numpy as np
from rasterio.enums import Resampling
import rioxarray as rxr

TARGET_H, TARGET_W = 128, 128

def align_and_write_np(metrics_dir, gee_dir, out_dir):
    """
    Reads and aligns GeoTIFF pairs, clips to a globally specified spatial window,
    then writes to NumPy binaries.
    """
    os.makedirs(os.path.join(out_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "targets"), exist_ok=True)

    metric_fps = [os.path.join(metrics_dir, f) for f in os.listdir(metrics_dir) if f.endswith(".tif")]
    
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
        
        # Save binaries
        features_fp = os.path.join(out_dir, "features", f"{bcgs_tile}.npy")
        targets_fp = os.path.join(out_dir, "targets", f"{bcgs_tile}.npy")  
        
        np.save(features_fp, features_cropped)
        np.save(targets_fp, targets_cropped)
        
        print(f"Processed '{bcgs_tile}' -> Features: {features_cropped.shape}, Targets: {targets_cropped.shape}")
        
        # Clean up raster memory handles explicitly
        metrics_ds.close()
        gee_ds.close()


def main():
    if len(sys.argv) < 4:
        print("Usage: python process_rasters.py <metrics_dir> <gee_dir> <out_dir>")
        sys.exit(1)
        
    metrics_dir = sys.argv[1]
    gee_dir = sys.argv[2]
    out_dir = sys.argv[3]

    align_and_write_np(metrics_dir, gee_dir, out_dir)


if __name__ == "__main__":
    main()