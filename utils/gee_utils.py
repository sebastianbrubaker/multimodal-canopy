import sys
import xarray as xr
import rioxarray as rxr
import ee
import geemap
import pyproj


def get_roi(ds, box_buf_dist, dest_epsg) -> ee.Geometry:
    """
    """
    # Get 2D bounds
    bounds = ds.rio.bounds()

    xmin, ymin, xmax, ymax = bounds
    xmin -= box_buf_dist
    ymin -= box_buf_dist
    xmax += box_buf_dist
    ymax += box_buf_dist
    bounds = (xmin, ymin, xmax, ymax)

    # Project bounds to WGS84
    transformer = pyproj.Transformer.from_crs(dest_epsg, "EPSG:4326", always_xy=True)
    xmin_wgs, ymin_wgs = transformer.transform(bounds[0], bounds[1])
    xmax_wgs, ymax_wgs = transformer.transform(bounds[2], bounds[3])

    # Return ee geom obj
    return ee.Geometry.Rectangle([xmin_wgs, ymin_wgs, xmax_wgs, ymax_wgs])    


def process_s2(image):
    """
    Masks clouds and snow  from Sentinel-2 imagery using the
    provided scene classification layer and a probability
    threshold, then returns spectral bands scaled to percent reflectance.
    """
    mask_cld = image.select("MSK_CLDPRB").lt(2)    # pr_cld
    mask_snw = image.select("MSK_SNWPRB").lt(2)    # pr_snw

    spectral_bands = image.select(["B2", "B3", "B4", "B8"])

    return spectral_bands.updateMask(mask_cld).updateMask(mask_snw).divide(10000)


def fetch_s2(roi, dest_epsg, date_start, date_end, cld_percentage):
    """
    """
    # Filter Sentinel-2 collection
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cld_percentage))
    ).sort("CLOUDY_PIXEL_PERCENTAGE")

    print(f"Images found: {s2_col.size().getInfo()}")

    # Process collection
    s2_col_processed = s2_col.map(process_s2)             # pixelwise mask and scale
    s2_img_med = s2_col_processed.median().clip(roi)      # median composite

    # Covert to Xarray
    native_projection = s2_col.first().select("B2").projection()
    s2_img_med_projected = s2_img_med.setDefaultProjection(native_projection)   # force to native projection

    # Return GEE image as local Xarray dataset
    return geemap.ee_to_xarray(
        dataset=s2_img_med_projected,
        geometry=roi,
        scale=10,       # 10m native Sentinel-2 resolution
        crs=dest_epsg,
    )


def fetch_s1(roi, dest_epsg, date_start, date_end):
    """
    """
    # Filter Sentinel-1 GRD collection
    s1_asc_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    ).sort("system:time_start")

    print(f"Images found: {s1_asc_col.size().getInfo()}")

    # Compute median composite and clip
    s1_med = s1_asc_col.select(["VV", "VH"]).median().clip(roi)

    # Return GEE image as local Xarray dataset
    return geemap.ee_to_xarray(
        dataset=s1_med,
        geometry=roi,
        scale=10,
        crs=dest_epsg,
    )


def fetch_alos(roi, dest_epsg, year):
    """
    """
    palsar_img = ee.Image(f"JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH/{year}")

    # Select bands and clip
    palsar_pol = palsar_img.select(["HH", "HV"]).clip(roi)

    gamma_naught = palsar_pol.log10().multiply(20).subtract(83.0)

    # Return GEE image as local Xarray dataset
    return geemap.ee_to_xarray(
        dataset=gamma_naught,
        geometry=roi,
        scale=10,          # upsample to match Sentinel-2
        crs=dest_epsg,
    )


def main():
    # Get and parse CLI arguments
    args = sys.argv[1:]
    bounds_fp = args[0].lower()
    date_start = args[1]
    date_end = args[2]
    dst_fp = args[3]

    year = date_end.strip().split("-")[0]
    cld_percentage = 5

    # Initilaize GEE
    ee.Authenticate()
    ee.Initialize(
        project="multimodal-regression"
    )

    # Get bounds
    bounds_ds = rxr.open_rasterio(bounds_fp)
    dest_epsg = f"EPSG:{pyproj.CRS(bounds_ds.rio.crs).to_2d().to_epsg()}"
    roi = get_roi(bounds_ds, 100, dest_epsg)

    # Fetch data
    s2_ds = fetch_s2(roi, dest_epsg, date_start, date_end, cld_percentage)
    s1_ds = fetch_s1(roi, dest_epsg, date_start, date_end)
    alos_ds = fetch_alos(roi, dest_epsg, year)

    # Stack and write to destination
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

    features.name = "Features"
    features.attrs.pop("id", None)
    features.attrs.pop("long_name", None)
    features.attrs["description"] = "Multimodal feature stack."
    features.attrs["date_start"] = date_start
    features.attrs["date_end"] = date_end
    features.attrs["source"] = "Google Earth Engine"
    features.rio.to_raster(dst_fp)


if __name__ == "__main__":
    main()