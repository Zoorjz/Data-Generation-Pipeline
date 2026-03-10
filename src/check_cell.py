import os
import argparse
import pandas as pd
import rasterio
import rasterio.mask
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Check information for a specific grid cell from the generated ML data.")
    parser.add_argument("--task", type=str, choices=["pred2020", "pred2021", "diff1yr"], required=True, help="Dataset task folder to check.")
    parser.add_argument("--row", type=int, required=True, help="Row index of the cell.")
    parser.add_argument("--col", type=int, required=True, help="Column index of the cell.")
    parser.add_argument("--data_dir", type=str, default="data/training_data", help="Directory where generated data is located.")
    parser.add_argument("--rgb", action="store_true", help="Display Sentinel-2 RGB as base.")
    parser.add_argument("--ndvi", action="store_true", help="Display NDVI as base.")
    parser.add_argument("--swir", action="store_true", help="Display SWIR as base.")
    parser.add_argument("--base_only", action="store_true", help="Only display the base raster map, suppressing ESA overlay and OSM vector data.")
    return parser.parse_args()

ESA_COLORS = {
    10: "#006400", 20: "#FFBB22", 30: "#FFFF4C", 40: "#F096FF", 
    50: "#FA0000", 60: "#B4B4B4", 70: "#F0F0F0", 
    80: "#0064C8", 90: "#0096A0", 95: "#00CF75", 100: "#FAE6A0"
}

def main():
    args = parse_args()
    
    if not (args.rgb or args.ndvi or args.swir):
        args.rgb = True  # Default fallback
        
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if args.task == "pred2020":
        data_dir = os.path.join(base_dir, args.data_dir, "Composition_prediction_in_3_years", "2020")
    elif args.task == "pred2021":
        data_dir = os.path.join(base_dir, args.data_dir, "Composition_prediction_in_3_years", "2021")
    elif args.task == "diff1yr":
        data_dir = os.path.join(base_dir, args.data_dir, "Composition_diff_in_one_year")
        
    csv_file = os.path.join(data_dir, "grid_stats.csv")
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return

    df = pd.read_csv(csv_file)
    cell_data = df[(df["row"] == args.row) & (df["col"] == args.col)]
    
    if cell_data.empty:
        print(f"Error: Cell at row {args.row}, col {args.col} not found in {csv_file}")
        return
        
    cell = cell_data.iloc[0]
    print(f"\n--- Information for Cell (Row: {args.row}, Col: {args.col}) ({args.task}) ---")
    for col in cell.index:
        print(f"  {col}: {cell[col]}")
        
    gpkg_file = os.path.join(data_dir, cell.get("gpkg_file", "training_data.gpkg"))
    tif_file = os.path.join(data_dir, cell.get("tif_file", "training_data.tif")) # fallback to expected format
    if not os.path.exists(gpkg_file) or not os.path.exists(tif_file):
        print(f"Error: Associated GPKG/TIF files not found: {gpkg_file} or {tif_file}")
        return
        
    box_4326 = box(cell["min_lon"], cell["min_lat"], cell["max_lon"], cell["max_lat"])
    bbox_gdf = gpd.GeoSeries([box_4326], crs="EPSG:4326")
    bbox_3857_geom = bbox_gdf.to_crs("EPSG:3857").iloc[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    print("\nLoading GTiff Raster Data...")
    try:
        with rasterio.open(tif_file) as src:
            out_image, out_transform = rasterio.mask.mask(src, [bbox_3857_geom], crop=True)
            
            # Legacy format check (4 bands) or new format check (10 bands)
            if out_image.shape[0] >= 4:
                extent = [out_transform.c, out_transform.c + out_transform.a * out_image.shape[2],
                          out_transform.f + out_transform.e * out_image.shape[1], out_transform.f]
                          
                if args.rgb:
                    rgb_img = np.transpose(out_image[:3], (1, 2, 0)) / 255.0
                    ax.imshow(rgb_img, extent=extent, zorder=1)
                elif args.ndvi and out_image.shape[0] >= 10:
                    ndvi_img = np.transpose(out_image[3:6], (1, 2, 0)) / 255.0
                    ax.imshow(ndvi_img, extent=extent, zorder=1)
                elif args.swir and out_image.shape[0] >= 10:
                    swir_img = np.transpose(out_image[6:9], (1, 2, 0)) / 255.0
                    ax.imshow(swir_img, extent=extent, zorder=1)
                elif (args.ndvi or args.swir) and out_image.shape[0] < 10:
                    print("This dataset only has 4 bands. Falling back to RGB.")
                    rgb_img = np.transpose(out_image[:3], (1, 2, 0)) / 255.0
                    ax.imshow(rgb_img, extent=extent, zorder=1)
                    
                if not args.base_only:
                    esa_idx = 9 if out_image.shape[0] >= 10 else 3
                    esa_data = out_image[esa_idx]
                    
                    # Plot ESA overlaid with some alpha
                    overlay = np.zeros((*esa_data.shape, 4), dtype=np.float32)
                    for cls_val, color in ESA_COLORS.items():
                        mask_arr = esa_data == cls_val
                        rgba_color = list(to_rgba(color))
                        rgba_color[3] = 0.5  # Alpha for ESA over base map
                        overlay[mask_arr] = rgba_color
                        
                    ax.imshow(overlay, extent=extent, zorder=2)
            else:
                print("Unexpected number of raster bands. Showing single band...")
                extent = [out_transform.c, out_transform.c + out_transform.a * out_image.shape[2],
                          out_transform.f + out_transform.e * out_image.shape[1], out_transform.f]
                ax.imshow(out_image[0], extent=extent, zorder=1)
                
    except Exception as e:
        print(f"! Failed to load raster from GTiff: {e}")
        
    if not args.base_only:
        print("\nLoading GeoPackage Vector Data...")
        try:
            lines = gpd.read_file(gpkg_file, layer="osm_infrastructure")
            lines = lines.to_crs("EPSG:3857")
            lines_clipped = gpd.clip(lines, bbox_3857_geom)
            if not lines_clipped.empty:
                lines_clipped.plot(ax=ax, color='red', linewidth=3, zorder=5, label="Infrastructure")
                print(f"  Found {len(lines_clipped)} infrastructure lines:")
                for idx, row in lines_clipped.iterrows():
                    name = row.get('name', 'Unnamed')
                    hw = row.get('highway')
                    rw = row.get('railway')
                    print(f"    - Name: {name}, Highway: {hw}, Railway: {rw}")
            else:
                print("  No infrastructure lines found in this cell.")
        except Exception as e:
            print(f"  ! Failed to load lines: {e}")
            
        try:
            stations = gpd.read_file(gpkg_file, layer="osm_stations")
            stations = stations.to_crs("EPSG:3857")
            stations_clipped = gpd.clip(stations, bbox_3857_geom)
            if not stations_clipped.empty:
                stations_clipped.plot(ax=ax, color='cyan', edgecolor='black', marker='*', markersize=200, zorder=6, label="Stations")
                print(f"  Found {len(stations_clipped)} stations:")
                for idx, row in stations_clipped.iterrows():
                    name = row.get('name', 'Unnamed')
                    rw = row.get('railway')
                    st = row.get('station')
                    print(f"    - Name: {name}, Railway: {rw}, Station Type: {st}")
            else:
                print("  No stations found in this cell.")
        except Exception as e:
            print(f"  ! Failed to load stations: {e}")
            
        try:
            border = gpd.read_file(gpkg_file, layer="nuremberg_border")
            border = border.to_crs("EPSG:3857")
            border_clipped = gpd.clip(border, bbox_3857_geom)
            if not border_clipped.empty:
                border_clipped.plot(ax=ax, color='none', edgecolor='blue', linewidth=4, zorder=7, label="Border")
                print("  Found Nuremberg City Border in this cell.")
        except Exception as e:
            pass
    base_str = "Sentinel-2 RGB"
    if args.ndvi: base_str = "NDVI"
    if args.swir: base_str = "SWIR"
    ax.set_title(f"Visualized Cell (Row: {args.row}, Col: {args.col}) ({args.task})\nBackground: {base_str} | Overlay: ESA WorldCover")
    ax.set_xlabel("EPSG:3857 Easting")
    ax.set_ylabel("EPSG:3857 Northing")
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
        
    print(f"Displaying graphical map plot for cell...")
    plt.show()

if __name__ == "__main__":
    main()
