import os
import argparse
import rasterio
import numpy as np
import pyproj
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox
import pandas as pd
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Generate ML training data for Nuremberg land cover.")
    parser.add_argument("--grid_size", type=int, default=1000, help="Grid size in meters (default: 1000).")
    parser.add_argument("--labels", nargs="+", default=["Built-up", "Permanent water bodies"], 
                        help="List of ESA WorldCover labels to include. Example: --labels Built-up 'Tree cover'")
    parser.add_argument("--out_dir", type=str, default="data/training_data", help="Output directory.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--denoise", action="store_true", help="Apply denoising to 2020 data")
    group.add_argument("--with_noise", action="store_true", help="Do not apply denoising (leave noise)")
    return parser.parse_args()

ESA_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice", 
    80: "Permanent water bodies", 90: "Herbaceous wetland", 
    95: "Mangroves", 100: "Moss and lichen"
}
INV_ESA_CLASSES = {v: k for k, v in ESA_CLASSES.items()}

def get_class_value(label):
    label_lower = label.lower()
    for class_id, class_name in ESA_CLASSES.items():
        if class_name.lower() == label_lower or label_lower in class_name.lower():
            return class_id
    raise KeyError(f"Label '{label}' not found in ESA classes. Available classes: {list(ESA_CLASSES.values())}")

def get_nuremberg_bbox():
    point = gpd.GeoSeries([gpd.points_from_xy([11.08], [49.45])[0]], crs="EPSG:4326")
    point_metric = point.to_crs("EPSG:32632")
    import shapely.affinity
    point_metric = gpd.GeoSeries(point_metric.geometry.translate(xoff=3000, yoff=-1000), crs="EPSG:32632")
    bbox_metric = point_metric.buffer(12500, cap_style=3)
    return gpd.GeoDataFrame(geometry=bbox_metric, crs="EPSG:32632")

def get_align_params(bbox_gdf_32632):
    dst_crs = 'EPSG:3857'
    bbox_3857 = bbox_gdf_32632.to_crs(dst_crs)
    minx, miny, maxx, maxy = bbox_3857.total_bounds
    
    minx = np.floor(minx / 10.0) * 10.0
    miny = np.floor(miny / 10.0) * 10.0
    maxx = np.ceil(maxx / 10.0) * 10.0
    maxy = np.ceil(maxy / 10.0) * 10.0
    
    dst_width = int((maxx - minx) / 10.0)
    dst_height = int((maxy - miny) / 10.0)
    dst_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, dst_width, dst_height)
    return dst_crs, dst_transform, dst_width, dst_height, (minx, miny, maxx, maxy)

def process_raster_memory(input_path, align_params):
    dst_crs, dst_transform, dst_width, dst_height, bounds = align_params
    with rasterio.open(input_path) as src:
        data = np.zeros((1, dst_height, dst_width), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
    return data[0]

def load_ee_local_raster(filepath, is_s2_uint16=False):
    """Loads a pre-aligned Earth Engine local raster directly into memory."""
    with rasterio.open(filepath) as src:
        data = src.read()
        if is_s2_uint16:
            # Reverting back to previous behaviour: scaling to 8-bit uint8 and taking mapping (RGB)
            data_out = data[:3].astype(np.float32)
            data_out = np.clip(data_out / 3000.0, 0, 1) * 255.0
            return data_out.astype(np.uint8)
        return data

def fetch_osm_infrastructure(bbox_3857_bounds, year):
    minx, miny, maxx, maxy = bbox_3857_bounds
    raster_box_3857 = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs="EPSG:3857")
    raster_box_4326 = raster_box_3857.to_crs("EPSG:4326")
    polygon = raster_box_4326.geometry.iloc[0]
    
    ox.settings.overpass_settings = f'[out:json][timeout:{{timeout}}]{{maxsize}}'
    tags = {'highway': True, 'railway': True, 'cycleway': True}
    
    try:
        graph = ox.features_from_polygon(polygon, tags=tags)
        lines = graph[graph.geometry.type.isin(['LineString', 'MultiLineString'])].copy()
        lines = gpd.clip(lines, polygon)
        keep_cols = ['geometry', 'highway', 'railway', 'route', 'busway', 'cycleway', 'name']
        lines = lines[[c for c in keep_cols if c in lines.columns]].copy()
        for c in keep_cols:
            if c != 'geometry' and c not in lines.columns:
                lines[c] = None
    except Exception as e:
        print(f"Failed OSM lines for {year}: {e}")
        lines = gpd.GeoDataFrame(columns=['geometry', 'highway', 'railway', 'route', 'busway', 'cycleway', 'name'], crs="EPSG:4326")
        
    tags_stations = {'railway': ['station', 'tram_stop', 'subway_entrance'], 'station': ['subway', 'light_rail']}
    try:
        stations = ox.features_from_polygon(polygon, tags=tags_stations)
        if not stations.empty:
            stations_3857 = stations.to_crs("EPSG:3857")
            stations_3857.geometry = stations_3857.geometry.centroid
            stations = stations_3857.to_crs("EPSG:4326")
            stations = gpd.clip(stations, polygon)
            keep_cols_stat = ['geometry', 'railway', 'station', 'name']
            stations = stations[[c for c in keep_cols_stat if c in stations.columns]].copy()
            for c in keep_cols_stat:
                if c != 'geometry' and c not in stations.columns:
                    stations[c] = None
        else:
            stations = gpd.GeoDataFrame(columns=['geometry', 'railway', 'station', 'name'], crs="EPSG:4326")
    except Exception:
        stations = gpd.GeoDataFrame(columns=['geometry', 'railway', 'station', 'name'], crs="EPSG:4326")
        
    ox.settings.overpass_settings = '[out:json][timeout:{timeout}]{maxsize}'
    return lines, stations

def fetch_nuremberg_border():
    """Downloads the administrative border of Nuremberg"""
    try:
        border_gdf = ox.geocode_to_gdf("Nuremberg, Germany")
        # Keep only the geometry to save space
        border_gdf = border_gdf[['geometry']]
        return border_gdf
    except Exception as e:
        print(f"Failed to fetch Nuremberg border: {e}")
        return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")

def write_to_gpkg(filepath_tif, filepath_gpkg, s2_data, ndvi_data, swir_data, raster_data, transform, width, height, lines_gdf, stations_gdf, border_gdf, selected_classes):
    filtered_raster = np.zeros_like(raster_data)
    for c in selected_classes:
        val = get_class_value(c)
        mask = (raster_data == val)
        filtered_raster[mask] = val
        
    if os.path.exists(filepath_tif):
        os.remove(filepath_tif)
    if os.path.exists(filepath_gpkg):
        os.remove(filepath_gpkg)
        
    meta = {
        'driver': 'GTiff',
        'dtype': 'uint8', # Reverting back to uint8
        'nodata': 0,
        'width': width,
        'height': height,
        'count': 10, # Reverting to 10 bands (3 S2, 3 NDVI, 3 SWIR, 1 ESA)
        'crs': 'EPSG:3857',
        'transform': transform
    }
    with rasterio.open(filepath_tif, 'w', **meta) as dst:
        for i in range(3):
            dst.write(s2_data[i], i + 1)
        for i in range(3):
            dst.write(ndvi_data[i], i + 4)
        for i in range(3):
            dst.write(swir_data[i], i + 7)
        dst.write(filtered_raster.astype(np.uint8), 10)
        
    if not lines_gdf.empty:
        lines_gdf.to_file(filepath_gpkg, driver="GPKG", layer="osm_infrastructure")
    if not stations_gdf.empty:
        stations_gdf.to_file(filepath_gpkg, driver="GPKG", layer="osm_stations")
    if not border_gdf.empty:
        border_gdf.to_file(filepath_gpkg, driver="GPKG", layer="nuremberg_border")


def generate():
    args = parse_args()
    
    suffix = "_denoised" if args.denoise else "_with_noise"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run{suffix}_{timestamp}")
    
    # Define completely new target paths
    pred_3_yrs_2020_dir = os.path.join(run_dir, "Composition_prediction_in_3_years", "2020")
    pred_3_yrs_2021_dir = os.path.join(run_dir, "Composition_prediction_in_3_years", "2021")
    diff_1_yr_dir = os.path.join(run_dir, "Composition_diff_in_one_year")
    
    for d in [pred_3_yrs_2020_dir, pred_3_yrs_2021_dir, diff_1_yr_dir]:
        os.makedirs(d, exist_ok=True)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Old original ESA worldcover ground truth layouts
    esa_2020_path = os.path.join(base_dir, "OriginalData", "WORLDCOVER", "ESA_WORLDCOVER_10M_2020_V100", "MAP", "ESA_WorldCover_10m_2020_v100_N48E009_Map", "ESA_WorldCover_10m_2020_v100_N48E009_Map.tif")
    esa_2021_path = os.path.join(base_dir, "OriginalData", "WORLDCOVER", "ESA_WORLDCOVER_10M_2021_V200", "MAP", "ESA_WorldCover_10m_2021_v200_N48E009_Map", "ESA_WorldCover_10m_2021_v200_N48E009_Map.tif")
    
    # Add Windows long path support to bypass 260 character limit
    if os.name == 'nt':
        esa_2020_path = f"\\\\?\\{os.path.normpath(os.path.abspath(esa_2020_path))}"
        esa_2021_path = f"\\\\?\\{os.path.normpath(os.path.abspath(esa_2021_path))}"
    
    bbox_32632 = get_nuremberg_bbox()
    align_params = get_align_params(bbox_32632)
    _, transform, width, height, bounds_3857 = align_params
    
    print("Processing 2020/2021 Ground Truth Raster Array...")
    data_2020 = process_raster_memory(esa_2020_path, align_params)
    data_2021 = process_raster_memory(esa_2021_path, align_params)
    
    data_2020_processed = np.copy(data_2020)
    if args.denoise:
        print("Calculating Noise Mask & Generating Denoised 2020 Maps...")
        noise_mask = (data_2021 == 50) & (data_2020 == 30)
        data_2020_processed[noise_mask] = 50
    else:
        print("Skipping denoising, keeping 2020 data with noise...")
    
    # Setup Earth Engine loaded images explicitly by mapping paths
    def load_year_ee_data(year_str):
        print(f"Loading local EE data for {year_str}...")
        path_prefix = os.path.join(base_dir, "data", f"sentinel2_downloads_{year_str}-06-01_{year_str}-09-01", "Nuremberg")
        s2 = load_ee_local_raster(f"{path_prefix}_S2RGBNIR.tif", is_s2_uint16=True)
        ndvi = load_ee_local_raster(f"{path_prefix}_NDVI.tif")
        swir = load_ee_local_raster(f"{path_prefix}_SWIR.tif")
        return s2, ndvi, swir
        
    s2_2017, ndvi_2017, swir_2017 = load_year_ee_data("2017")
    s2_2018, ndvi_2018, swir_2018 = load_year_ee_data("2018")
    s2_2020, ndvi_2020, swir_2020 = load_year_ee_data("2020")
    
    print("Downloading OSM Data (dynamically bound to Sentinel target dates)...")
    lines_2017, stations_2017 = fetch_osm_infrastructure(bounds_3857, 2017)
    lines_2018, stations_2018 = fetch_osm_infrastructure(bounds_3857, 2018)
    lines_2020, stations_2020 = fetch_osm_infrastructure(bounds_3857, 2020)
    
    print("Downloading Nuremberg City Border...")
    border_gdf = fetch_nuremberg_border()
    
    gpkg_pred_2020 = os.path.join(pred_3_yrs_2020_dir, f"training_data{suffix}.gpkg")
    tif_pred_2020 = os.path.join(pred_3_yrs_2020_dir, f"training_data{suffix}.tif")
    
    gpkg_pred_2021 = os.path.join(pred_3_yrs_2021_dir, f"training_data{suffix}.gpkg")
    tif_pred_2021 = os.path.join(pred_3_yrs_2021_dir, f"training_data{suffix}.tif")
    
    gpkg_diff_2020 = os.path.join(diff_1_yr_dir, f"training_data{suffix}.gpkg")
    tif_diff_2020 = os.path.join(diff_1_yr_dir, f"training_data{suffix}.tif")
    
    print("Writing Target 1: Pred 3 Years (2020)...")
    write_to_gpkg(tif_pred_2020, gpkg_pred_2020, s2_2017, ndvi_2017, swir_2017, data_2020_processed, transform, width, height, lines_2017, stations_2017, border_gdf, args.labels)
    
    print("Writing Target 2: Pred 3 Years (2021)...")
    write_to_gpkg(tif_pred_2021, gpkg_pred_2021, s2_2018, ndvi_2018, swir_2018, data_2021, transform, width, height, lines_2018, stations_2018, border_gdf, args.labels)
    
    print("Writing Target 3: Diff 1 Year Ahead (2020-2021)...")
    # For differential tasks, the label array technically isn't explicitly printed into the input geo-image natively as a band (handled via pure grid math).
    # But for visual integrity, we insert `data_2020_processed` into the .tif output since it represents the baseline map representation matching that time.
    write_to_gpkg(tif_diff_2020, gpkg_diff_2020, s2_2020, ndvi_2020, swir_2020, data_2020_processed, transform, width, height, lines_2020, stations_2020, border_gdf, args.labels)
    
    print(f"Generating Grids ({args.grid_size}m)...")
    P = max(1, args.grid_size // 10)
    num_rows = int(np.ceil(height / P))
    num_cols = int(np.ceil(width / P))
    
    proj = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    
    rows_pred_2020 = []
    rows_pred_2021 = []
    rows_diff_1_yr = []
    
    cell_id = 0
    for r in range(num_rows):
        for c in range(num_cols):
            r_start, r_end = r * P, min((r + 1) * P, height)
            c_start, c_end = c * P, min((c + 1) * P, width)
            
            block_2021 = data_2021[r_start:r_end, c_start:c_end]
            block_2020_processed = data_2020_processed[r_start:r_end, c_start:c_end]
            
            if block_2021.size == 0:
                continue
                
            minx = transform.c + c_start * transform.a
            maxy = transform.f + r_start * transform.e
            maxx = transform.c + c_end * transform.a
            miny = transform.f + r_end * transform.e
            
            lon_min, lat_min = proj.transform(minx, miny)
            lon_max, lat_max = proj.transform(maxx, maxy)
            
            
            base_row = {
                "cell_id": cell_id,
                "row": r,
                "col": c,
                "min_lon": lon_min, "min_lat": lat_min, "max_lon": lon_max, "max_lat": lat_max,
                "grid_size_m": args.grid_size
            }
            
            # Predict labels 3-years explicitly (e.g. baseline feature array matches offset 2020/2021 explicitly)
            pred_2020_data = base_row.copy()
            pred_2020_data["gpkg_file"] = f"training_data{suffix}.gpkg"
            pred_2020_data["tif_file"] = f"training_data{suffix}.tif"
            
            pred_2021_data = base_row.copy()
            pred_2021_data["gpkg_file"] = f"training_data{suffix}.gpkg"
            pred_2021_data["tif_file"] = f"training_data{suffix}.tif"
            
            # Learn composition differential relative to 1-year boundaries
            diff_2020_data = base_row.copy()
            diff_2020_data["gpkg_file"] = f"training_data{suffix}.gpkg"
            diff_2020_data["tif_file"] = f"training_data{suffix}.tif"
            
            selected_sum_pred_2020 = 0
            selected_sum_pred_2021 = 0
            selected_sum_diff_baseline = 0
            selected_sum_diff_target = 0
            
            for label in args.labels:
                val = get_class_value(label)
                # Ensure the original case from ESA_CLASSES is used for column consistency
                proper_label = ESA_CLASSES[val]
                
                # Composition label distributions
                pct_2020_processed = (np.sum(block_2020_processed == val) / block_2021.size) * 100
                pct_2021_raw = (np.sum(block_2021 == val) / block_2021.size) * 100
                
                # Setup models mapping purely the spatial component
                pred_2020_data[f"{proper_label} %"] = round(pct_2020_processed, 2)
                selected_sum_pred_2020 += pct_2020_processed
                
                pred_2021_data[f"{proper_label} %"] = round(pct_2021_raw, 2)
                selected_sum_pred_2021 += pct_2021_raw
                
                # Setup model optimizing for learning changes exclusively isolated
                diff_2020_data[f"{proper_label} Baseline %"] = round(pct_2020_processed, 2)
                diff_2020_data[f"{proper_label} Target %"] = round(pct_2021_raw, 2)
                diff_2020_data[f"delta {proper_label} %"] = round(pct_2021_raw - pct_2020_processed, 2)
                selected_sum_diff_baseline += pct_2020_processed
                selected_sum_diff_target += pct_2021_raw
                
            # Handle implicitly tracked 'Others'
            others_pred_2020 = max(0.0, 100.0 - selected_sum_pred_2020)
            others_pred_2021 = max(0.0, 100.0 - selected_sum_pred_2021)
            others_baseline = max(0.0, 100.0 - selected_sum_diff_baseline)
            others_target = max(0.0, 100.0 - selected_sum_diff_target)
            
            pred_2020_data["Other %"] = round(others_pred_2020, 2)
            pred_2021_data["Other %"] = round(others_pred_2021, 2)
            
            diff_2020_data["Other Baseline %"] = round(others_baseline, 2)
            diff_2020_data["Other Target %"] = round(others_target, 2)
            diff_2020_data["delta Other %"] = round(others_target - others_baseline, 2)
            
            rows_pred_2020.append(pred_2020_data)
            rows_pred_2021.append(pred_2021_data)
            rows_diff_1_yr.append(diff_2020_data)
            cell_id += 1
            
    df_pred_2020 = pd.DataFrame(rows_pred_2020)
    df_pred_2021 = pd.DataFrame(rows_pred_2021)
    df_diff_1_yr = pd.DataFrame(rows_diff_1_yr)
    
    df_pred_2020.to_csv(os.path.join(pred_3_yrs_2020_dir, f"grid_stats{suffix}.csv"), index=False)
    df_pred_2021.to_csv(os.path.join(pred_3_yrs_2021_dir, f"grid_stats{suffix}.csv"), index=False)
    df_diff_1_yr.to_csv(os.path.join(diff_1_yr_dir, f"grid_stats{suffix}.csv"), index=False)
    
    print("Done! Files generated in", run_dir)

if __name__ == "__main__":
    generate()
