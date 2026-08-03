import sys
import os
import warnings
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
import ee
import geemap
import pyproj
from rasterio.enums import Resampling
import rioxarray as rxr
import xarray as xr

# Suppress the backend time-start warning
warnings.filterwarnings(
    "ignore", message="Unable to retrieve 'system:time_start'"
)

CLD_PERCENT = 5
DELTA_WEEKS = 8
S1_RESOLUTION = 10  # m


def get_roi(ds, box_buf_dist, dest_epsg) -> ee.Geometry:
    """
    Gets the 2D bounds of a local raster dataset, applies a buffer,
    and returns it as a WGS84 Earth Engine Rectangle.
    """
    bounds = ds.rio.bounds()
    xmin, ymin, xmax, ymax = bounds
    xmin -= box_buf_dist
    ymin -= box_buf_dist
    xmax += box_buf_dist
    ymax += box_buf_dist
    bounds = (xmin, ymin, xmax, ymax)

    transformer = pyproj.Transformer.from_crs(
        dest_epsg, "EPSG:4326", always_xy=True
    )
    xmin_wgs, ymin_wgs = transformer.transform(bounds[0], bounds[1])
    xmax_wgs, ymax_wgs = transformer.transform(bounds[2], bounds[3])

    return ee.Geometry.Rectangle([xmin_wgs, ymin_wgs, xmax_wgs, ymax_wgs])


def process_s2(image):
    """
    Masks clouds and snow from Sentinel-2 imagery using the provided scene
    classification layer and a probability threshold, then returns spectral
    bands scaled to percent reflectance.
    """
    mask_cld = image.select("MSK_CLDPRB").lt(2)
    mask_snw = image.select("MSK_SNWPRB").lt(2)
    bands = image.select([
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B11",
        "B12",
    ])
    return bands.updateMask(mask_cld).updateMask(mask_snw).divide(10000)


def fetch_s2_by_resolution(roi, dest_epsg, date_start, date_end, cld_percentage, bands, resolution):
    """
    Fetches Sentinel-2 data and creates median composite for specified ROI.
    """
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cld_percentage))
    ).sort("CLOUDY_PIXEL_PERCENTAGE")

    s2_col_processed = s2_col.map(process_s2)
    s2_img_med = s2_col_processed.median().clip(roi)

    # Select only the target bands for this resolution tier
    s2_img_selected = s2_img_med.select(bands)

    # Force projection based on the first band of the requested group
    native_projection = s2_col.first().select(bands[0]).projection()
    s2_img_projected = s2_img_selected.setDefaultProjection(native_projection)

    # Return wrapped in an ImageCollection to prevent xee warnings
    return geemap.ee_to_xarray(
        dataset=ee.ImageCollection(s2_img_projected),
        geometry=roi,
        scale=resolution,
        crs=dest_epsg,
    )


def fetch_s1(roi, dest_epsg, date_start, date_end):
    """
    Fetches dual-pol ascending C-Band SAR from the Sentinel-1 GRD collection
    and creates median composite. 
    """
    s1_asc_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    ).sort("system:time_start")

    s1_med = s1_asc_col.select(["VV", "VH"]).median().clip(roi)

    return geemap.ee_to_xarray(
        dataset=ee.ImageCollection(s1_med),
        geometry=roi,
        scale=S1_RESOLUTION,
        crs=dest_epsg,
    )


def fetch_tile(in_fp, out_dir):
    """
    Fetches Sentinel-1 and Sentinel-2 data from GEE for a given ROI,
    computes a pixel-wise median over a time range, upsamples all bands 
    to a 10m resolution locally, and writes to disk as GeoTiff.  
    """
    # Get target bounds projection collection date
    bounds_ds = rxr.open_rasterio(in_fp)
    dest_epsg = f"EPSG:{pyproj.CRS(bounds_ds.rio.crs).to_2d().to_epsg()}"
    roi = get_roi(bounds_ds, 100, dest_epsg)

    date_start_str = str(bounds_ds.attrs["acquisition_date"])
    date_start_obj = datetime.strptime(date_start_str, "%Y%m%d")
    date_end_obj = date_start_obj + timedelta(weeks=DELTA_WEEKS)
    date_start_str = str(date_start_obj.date())
    date_end_str = str(date_end_obj.date())

    # Fetch datasets at native resolutions
    bands_10m = ["B2", "B3", "B4", "B8"]
    bands_20m = ["B5", "B6", "B7", "B11", "B12"]

    s2_10m_ds = fetch_s2_by_resolution(roi, dest_epsg, date_start_str, date_end_str, CLD_PERCENT, bands_10m, 10)
    s2_20m_ds = fetch_s2_by_resolution(roi, dest_epsg, date_start_str, date_end_str, CLD_PERCENT, bands_20m, 20)
    s1_ds = fetch_s1(roi, dest_epsg, date_start_str, date_end_str)

    # Upsample 20m bands
    anchor_layer = s2_10m_ds["B2"]

    s2_b2 = s2_10m_ds["B2"]
    s2_b3 = s2_10m_ds["B3"]
    s2_b4 = s2_10m_ds["B4"]
    s2_b8 = s2_10m_ds["B8"]
    s1_vv = s1_ds["VV"]
    s1_vh = s1_ds["VH"]

    s2_b5_up = s2_20m_ds["B5"].rio.reproject_match(anchor_layer, resampling=Resampling.bilinear)
    s2_b6_up = s2_20m_ds["B6"].rio.reproject_match(anchor_layer, resampling=Resampling.bilinear)
    s2_b7_up = s2_20m_ds["B7"].rio.reproject_match(anchor_layer, resampling=Resampling.bilinear)
    s2_b11_up = s2_20m_ds["B11"].rio.reproject_match(anchor_layer, resampling=Resampling.bilinear)
    s2_b12_up = s2_20m_ds["B12"].rio.reproject_match(anchor_layer, resampling=Resampling.bilinear)

    # Concatenate the aligned arrays along the band dimension
    features = xr.concat(
        [
            s2_b2,
            s2_b3,
            s2_b4,
            s2_b5_up,
            s2_b6_up,
            s2_b7_up,
            s2_b8,
            s2_b11_up,
            s2_b12_up,
            s1_vv,
            s1_vh,
        ],
        dim="band",
    )

    # Name the bands
    features = features.assign_coords(
        band=[
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B11",
            "B12",
            "C_VV",
            "C_VH",
        ]
    )

    # Safely drop time dimension
    if features.sizes["time"] == 1:
        features = features.squeeze("time", drop=True)

    # Clean up metadata attributes
    features.name = "Features"
    features.attrs.pop("id", None)
    features.attrs.pop("long_name", None)
    features.attrs["description"] = "Multimodal feature stack."
    features.attrs["date_start"] = date_start_str
    features.attrs["date_end"] = date_end_str
    features.attrs["source"] = (
        "Google Earth Engine"
    )

    # Extract BCGS tile for name and write to GeoTiff
    bcgs_tile = os.path.basename(in_fp)[0:16]
    dst_fp = os.path.join(out_dir, f"{bcgs_tile}.tif")
    features.rio.to_raster(dst_fp)


def main():
    # Get and parse CLI arguments
    args = sys.argv[1:]
    in_dir = args[0]
    out_dir = args[1]

    # Initialize GEE
    ee.Authenticate()
    ee.Initialize(project="multimodal-regression")

    # Process
    tif_fps = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith(".tif")]
    
    for fp in tqdm(tif_fps):
        try:
            fetch_tile(fp, out_dir)
        except Exception as e:
            print(f"Failed to retreive GEE data for {fp}: {e}")
            



if __name__ == "__main__":
    main()
