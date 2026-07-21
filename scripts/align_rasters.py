import sys
import os
import numpy as np
from rasterio.enums import Resampling
import rioxarray as rxr



def align_and_write_np(metrics_dir, gee_dir, out_dir):
    """
    Reads in and aligns GeoTiff pairs, then writes to NumPy binaries.
    """
    os.makedirs(f"{out_dir}/features", exist_ok=True)
    os.makedirs(f"{out_dir}/targets", exist_ok=True)

    # Get all target tif paths.
    metric_fps = [os.path.join(metrics_dir, f) for f in os.listdir(metrics_dir) if f.endswith(".tif")]
    for metric_fp in metric_fps:
        # Get filename and extract BCGS Tile 
        bcgs_tile = os.path.splitext(os.path.basename(metric_fp))[0]
        print(bcgs_tile)
        
        # Locate matching GEE feature stack
        gee_path = os.path.join(gee_dir, f"{bcgs_tile}.tif")
        if not os.path.exists(gee_path):
            print(f"Warning: No matching GEE file found for tile {bcgs_tile}. Skipping.")
            continue
            
        # Read data
        metrics_ds = rxr.open_rasterio(metric_fp)
        gee_ds = rxr.open_rasterio(gee_path)
        
        # Align feature to target
        gee_aligned = gee_ds.rio.reproject_match(metrics_ds, resampling=Resampling.nearest)
        
        # Fill GeoSpatial NoData with NaNs
        target_array = metrics_ds.values
        feature_array = gee_aligned.values
        target_array[target_array == metrics_ds.rio.nodata] = np.nan
        
        # Write as numpy binaries
        features_fp = os.path.join(out_dir, "features", f"{bcgs_tile}.npy")
        targets_fp = os.path.join(out_dir, "targets", f"{bcgs_tile}.npy")  
        np.save(features_fp, feature_array.astype(np.float32))
        np.save(targets_fp, target_array.astype(np.float32))
        print(f"Successfully processed and aligned tile: {bcgs_tile}. Shape: {feature_array.shape[1:]}")


def main():
    """
    """
    args = sys.argv[1:]
    metrics_dir = args[0]
    gee_dir = args[1]
    out_dir = args[2]

    align_and_write_np(metrics_dir, gee_dir, out_dir)


    
if __name__ == "__main__":
    main()