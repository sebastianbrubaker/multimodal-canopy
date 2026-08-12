import sys
import os
import re
import rioxarray as rxr
from rasterio.enums import Resampling
from tqdm.auto import tqdm


def extract_date_from_filename(filename):
    """
    Extracts an 8-digit date string (YYYYMMDD) from BC LiDAR filename. 
    Raises an error if not found.
    """
    match = re.search(r"(\d{8})", filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not parse YYYYMMDD date from filename: {filename}")
    

def compute_chms(dsm_dir, dtm_dir, chm_dir):
    os.makedirs(chm_dir, exist_ok=True)
    
    # Get all .tif files
    dsm_files = {f[:16]: os.path.join(dsm_dir, f) for f in os.listdir(dsm_dir) if f.endswith(".tif")}
    dtm_files = {f[:16]: os.path.join(dtm_dir, f) for f in os.listdir(dtm_dir) if f.endswith(".tif")}

    # Match explicitly on the first 16 characters (BCGS tile ID)
    common_tiles = set(dsm_files.keys()) & set(dtm_files.keys())
    if not common_tiles:
        print("No matching DSM and DTM tile prefixes found.")
        return

    for tile_id in tqdm(sorted(common_tiles)):
        dsm_fp = dsm_files[tile_id]
        dtm_fp = dtm_files[tile_id]
        
        acq_date = extract_date_from_filename(os.path.basename(dtm_fp))

        # Read in with masking
        dsm = rxr.open_rasterio(dsm_fp, masked=True)
        dtm = rxr.open_rasterio(dtm_fp, masked=True)

        # Align grids
        dsm_matched = dsm.rio.reproject_match(dtm, resampling=Resampling.nearest)
        
        # Calculate Canopy Height Model (CHM)
        chm = dsm_matched - dtm
        chm = chm.clip(min=0)

        # Block reduce
        chm_10m = chm.coarsen(x=10, y=10, boundary="trim").mean()

        # Force rioxarray to recalculate the transform matrix for the new 10m grid
        # chm_10m = chm_10m.rio.write_transform()

        # Update attributes and write
        chm_10m.attrs["acquisition_date"] = acq_date
        dst = os.path.join(chm_dir, f"{tile_id}_CHM_10m.tif")
        chm_10m.rio.to_raster(dst, driver="GTiff", compress="deflate")


def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <dsm_dir> <dtm_dir> <chm_dir>")
        sys.exit(1)

    dsm_dir = sys.argv[1]
    dtm_dir = sys.argv[2]
    chm_dir = sys.argv[3]

    compute_chms(dsm_dir, dtm_dir, chm_dir)

if __name__ == "__main__":
    main()