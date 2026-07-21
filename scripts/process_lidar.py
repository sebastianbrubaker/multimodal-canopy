import sys
import os
from tqdm import tqdm
import numpy as np
import re
import laspy
import rasterio
from rasterio.windows import Window
from skimage.measure import block_reduce
from pyforestscan.handlers import read_lidar
from pyforestscan.filters import filter_hag
from pyforestscan.calculate import assign_voxels, calculate_pad, \
    calculate_pai, calculate_chm, calculate_canopy_cover



TRIM = 3    # metres trimmed = TRIM * 10



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
    filename = os.path.basename(in_fp)
    acq_date = extract_date_from_filename(filename)
    bcgs_tile = filename[0:16]

    # Extract SRS and filter out noise classes 7 and 18
    with laspy.open(in_fp) as las_file:
        try:
            crs = las_file.header.parse_crs()
            epsg_code = crs.to_epsg()
            epsg_string = f"EPSG:{epsg_code}"
        except Exception as e:
            print(f"Failed to parse CRS from header: {e}")
            
        las = las_file.read()
        
        valid_points_mask = (las.classification != 7) & (las.classification != 18)
        las.points = las.points[valid_points_mask]
        
        # Re-export filtered point cloud as a temporary LAZ for PyForestScan
        temp_laz = "temp_filtered.laz"
        las.write(temp_laz)

    # Read the filtered points into PyForestScan and delete the temp
    try:
        # NOTE: Delauney Triangle HAG filter is a runtime bottleneck
        arrays = read_lidar(temp_laz, epsg_string, hag=True)        
    finally:
        if os.path.exists(temp_laz):
            os.remove(temp_laz)

    # Filter HAG and drop unecessary fields
    arrays = [strip_fields(a) for a in arrays]
    arrays = filter_hag(arrays)
    points = arrays[0]

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



def main():
    # Get and parse CLI args
    args = sys.argv[1:]
    in_dir = args[0]
    out_dir = args[1]

    # Construct a list of .laz files in the directory
    laz_fps = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith(".laz")]
    for fp in tqdm(laz_fps):
        process_tile(fp, out_dir)



if __name__ == "__main__":
    main()

