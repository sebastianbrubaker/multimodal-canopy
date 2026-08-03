import sys
import os
import gc
import numpy as np
import re
import laspy
import pdal
import json
import rasterio
from rasterio.windows import Window
from skimage.measure import block_reduce
from pyforestscan.calculate import assign_voxels, calculate_pad, \
    calculate_pai, calculate_chm, calculate_canopy_cover
from concurrent.futures import ProcessPoolExecutor
from functools import partial



TRIM = 6    # metres trimmed = TRIM * 10



def strip_fields(arr, fields=("X", "Y", "Z", "HeightAboveGround")):
    """
    Strips all fields not contained in fields tuple parameter.
    """
    dtype = np.dtype([(f, arr.dtype[f]) for f in fields])
    out = np.empty(arr.shape, dtype=dtype)
    for f in fields:
        out[f] = arr[f]
    return out


def extract_date_from_filename(filename):
    """
    Extracts an 8-digit date string (YYYYMMDD) from LAZ filename. 
    Raises an error if not found.
    """
    match = re.search(r"(\d{8})", filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not parse YYYYMMDD date from filename: {filename}")

    

def process_tile(in_fp, out_dir):
    try:
        filename = os.path.basename(in_fp)
        acq_date = extract_date_from_filename(filename)
        bcgs_tile = filename[0:16]

        # Quick read for EPSG string
        with laspy.open(in_fp) as las_file:
            try:
                crs = las_file.header.parse_crs()
                epsg_string = f"EPSG:{crs.to_epsg()}"
            except Exception as e:
                print(f"Failed to parse CRS from header: {e}")
                epsg_string = "EPSG:3005" # Fallback to BC Albers

        # Construct PDAL pipeline
        pipeline_json = {
            "pipeline": [
                in_fp,
                {
                    "type": "filters.range",
                    # Drop noise classes 7 and 18
                    "limits": "Classification![7:7],Classification![18:18]"
                },
                {
                    # HAG filtering
                    "type": "filters.hag_delaunay" 
                }
            ]
        }

        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
        points = pipeline.arrays[0]

        # Strip unnecessary fields to save memory before voxelization
        points = strip_fields(points)

        # Create voxels
        voxel_resolution = (1, 1, 1) 
        voxels, extent = assign_voxels(points, voxel_resolution)

        # Compute CHM 
        chm, _ = calculate_chm(
            points,
            voxel_resolution,
            interpolation="linear",
            interp_valid_region=True,
        )
        chm_10m = block_reduce(chm, block_size=(10, 10), func=np.nanmean)
        
        # Compute rugosity
        rugosity_10m = block_reduce(chm, block_size=(10, 10), func=np.nanstd)

        # Compute PAD and PAI
        pad = calculate_pad(voxels, voxel_resolution[-1])        
        pai = calculate_pai(pad, voxel_height=1)
        pai_10m = block_reduce(pai, block_size=(10,10), func=np.nanmean)

        # Compute canopy cover
        cover = calculate_canopy_cover(
            pad, 
            voxel_height=voxel_resolution[-1],
            min_height=2.0, # m
            k=0.5   # forest structure assumption
        )   
        cover_10m = block_reduce(cover, block_size=(10,10), func=np.nanmean)

        # Ensure all block_reduce outputs share identical minimum dimensions
        min_y = min(chm_10m.shape[0], rugosity_10m.shape[0], pai_10m.shape[0], cover_10m.shape[0])
        min_x = min(chm_10m.shape[1], rugosity_10m.shape[1], pai_10m.shape[1], cover_10m.shape[1])

        chm_10m = chm_10m[:min_y, :min_x]
        rugosity_10m = rugosity_10m[:min_y, :min_x]
        pai_10m = pai_10m[:min_y, :min_x]
        cover_10m = cover_10m[:min_y, :min_x]

        # Stack and transpose spatial dims
        metrics_stack = np.stack([chm_10m, rugosity_10m, pai_10m, cover_10m])
        metrics_stack = metrics_stack.transpose(0, 2, 1)    # band, y, x for rasterio

        # Geospatial nodata
        nodata_value = -9999.0
        metrics_stack = np.nan_to_num(metrics_stack, nan=nodata_value)

        # Get shapes and extent
        xmin, xmax, ymin, ymax = extent
        bands, height, width = metrics_stack.shape

        # Generate the affine transform for the 10m grid
        transform = rasterio.transform.from_bounds(
            xmin, ymin,
            xmax, ymax,
            width, height
        )       # (west, south, east, north, width, height)

        # Create metadata
        out_meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": metrics_stack.dtype.name,
            "crs": epsg_string,
            "transform": transform,
            "nodata": nodata_value,
            "compress": "deflate"
        }

        out_fp = os.path.join(out_dir, f"{bcgs_tile}.tif")
        os.makedirs(out_dir, exist_ok=True)

        # Slice numpy array to clip bounds
        t = TRIM
        trimmed_metrics = metrics_stack[:, t:-t, t:-t] if t > 0 else metrics_stack
        
        # Update metadata parameters for the trimmed dimensions
        new_height, new_width = trimmed_metrics.shape[1], trimmed_metrics.shape[2]
        window = Window(t, t, new_width, new_height)
        new_transform = rasterio.windows.transform(window, transform)
        
        out_meta.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform
        })

        # Write to tif
        with rasterio.open(out_fp, "w", **out_meta) as dst:
            dst.write(trimmed_metrics)
            dst.set_band_description(1, "CHM_10m")
            dst.set_band_description(2, "Rugosity_10m")
            dst.set_band_description(3, "PAI_10m")
            dst.set_band_description(4, "Canopy_Cover_10m")
            dst.update_tags(acquisition_date=acq_date)

        # Explicitly clear heavy memory objects
        del points, voxels, pad, pai, cover, metrics_stack
        gc.collect()

        return True, filename, ""

    except Exception as e:
        return False, filename, str(e)


def main():
    args = sys.argv[1:]
    in_dir = args[0]
    out_dir = args[1]

    laz_fps = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith(".laz")]
    
    # Bind the out_dir argument to process_tile
    process_func = partial(process_tile, out_dir=out_dir)
    
    # Allocate N-1 cores or manually specify
    # num_workers = max(1, os.cpu_count() - 1) 
    MAX_WORKERS = 2
    
    print(f"Processing {len(laz_fps)} tiles across {MAX_WORKERS} cores...")

    failed_tiles = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_func, laz_fps))

    for success, fname, err in results:
        if not success:
            failed_tiles.append((fname, err))       

    print("\n" + "="*50)
    print(f"PROCESSING COMPLETE: {len(laz_fps) - len(failed_tiles)}/{len(laz_fps)} succeeded.")

    if failed_tiles:
        for fname, err in failed_tiles:
            print(f"Fail: {fname}: {err}", "\n", "-"*50, "\n" )

        log_path = os.path.join(out_dir, "data/failed_tiles_log.txt")
        with open(log_path, "w") as f:
            for fname, err in failed_tiles:
                f.write(f"Tile: {fname}\n{err}\n{'-'*50}\n")



if __name__ == "__main__":
    main()

