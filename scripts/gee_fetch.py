import sys
import os
from datetime import datetime, timedelta
import pyproj
import xarray as xr
import rioxarray as rxr
import ee
from tqdm import tqdm

# Appends sibling directory to path list to use gee_utils.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sibling_dir = os.path.join(parent_dir, "utils")
sys.path.append(sibling_dir)

from gee_utils import get_roi, fetch_s1, fetch_s2, fetch_alos


DELTA_WEEKS = 8
CLD_PERCENT = 5

def fetch_tile(in_fp, out_dir):
    """
    """
    # Extract collection date from BC LiDAR-named file 
    start_date_str = in_fp.strip().split("_")[-2].split(".")[0]
    start_date_obj = datetime.strptime(start_date_str, "%Y%m%d")
    end_date_obj = start_date_obj + timedelta(weeks=DELTA_WEEKS)
    start_date_str = str(start_date_obj.date())
    end_date_str = str(end_date_obj.date())
    year = start_date_obj.year

    # Get bounds
    bounds_ds = rxr.open_rasterio(in_fp)
    dest_epsg = f"EPSG:{pyproj.CRS(bounds_ds.rio.crs).to_2d().to_epsg()}"
    roi = get_roi(bounds_ds, 100, dest_epsg)

    # Fetch data
    s2_ds = fetch_s2(roi, dest_epsg, start_date_str, end_date_str, CLD_PERCENT)
    s1_ds = fetch_s1(roi, dest_epsg, start_date_str, end_date_str)
    alos_ds = fetch_alos(roi, dest_epsg, year)

    # Stack into data cube
    features = xr.concat(
        [
            s2_ds["B2"],
            s2_ds["B3"],
            s2_ds["B4"],
            s2_ds["B8"],
            s1_ds["VV"],
            s1_ds["VH"],
            alos_ds["HH"],
            alos_ds["HV"],
        ],
        dim="band"
    )

    # Assign band names
    features = features.assign_coords(
        band=[
            "B2",
            "B3",
            "B4",
            "B8",
            "s1_VV",
            "s1_VH",
            "alos_HH",
            "alos_HV",
        ]
    )

    if features.sizes["time"] == 1:
        features = features.squeeze("time", drop=True)     # flatten time dimension

    # Update metadata
    features.name = "Features"
    features.attrs.pop("id", None)
    features.attrs.pop("long_name", None)
    features.attrs["description"] = "Multimodal feature stack."
    features.attrs["date_start"] = start_date_str
    features.attrs["date_end"] = end_date_str
    features.attrs["source"] = "Google Earth Engine"
    
    # Extract BCGS tile for name write to tif
    bcgs_tile = os.path.basename(in_fp)[0:16]
    out_fp = os.path.join(out_dir, f"{bcgs_tile}.tif")
    features.rio.to_raster(out_fp)

def main():
    args = sys.argv[1:]
    in_dir = args[0]
    out_dir = args[1]
    
    # Initilaize GEE
    ee.Authenticate()
    ee.Initialize(
        project="multimodal-regression"
    )

    for f in tqdm(os.listdir(in_dir)):
        if not f.endswith((".tif", ".geotif")): continue    # skip non-tif files

        # Construct file path
        in_fp = os.path.join(in_dir, f)

        fetch_tile(in_fp, out_dir)


if __name__ == "__main__":
    main()