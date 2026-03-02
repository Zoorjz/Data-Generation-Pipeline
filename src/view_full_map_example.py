"""
Example script demonstrating how to load and view full generated ML datasets visually.
It allows you to toggle specifically which layers you want to extract and render.

Examples:
    python src/view_full_map_example.py --task pred2021
    python src/view_full_map_example.py --task pred2021 --osm 
    python src/view_full_map_example.py --task diff1yr --labels --border
"""

import argparse
import os
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

# Identical Color mapping for ESA as the base Dashboard
ESA_COLORS = {
    10: "#006400", 20: "#FFBB22", 30: "#FFFF4C", 40: "#F096FF", 
    50: "#FA0000", 60: "#B4B4B4", 70: "#F0F0F0", 
    80: "#0064C8", 90: "#0096A0", 95: "#00CF75", 100: "#FAE6A0"
}

def parse_args():
    parser = argparse.ArgumentParser(description="View full ML training data map locally.")
    parser.add_argument("--task", type=str, choices=["pred2020", "pred2021", "diff1yr"], required=True, help="Task dataset to view.")
    
    # Toggles for explicit rendering
    parser.add_argument("--rgb", action="store_true", help="Display Sentinel-2 RGB Baseline Layer.")
    parser.add_argument("--ndvi", action="store_true", help="Display NDVI Baseline Layer.")
    parser.add_argument("--swir", action="store_true", help="Display SWIR Baseline Layer.")
    parser.add_argument("--labels", action="store_true", help="Display ESA WorldCover Labels Layer.")
    parser.add_argument("--osm", action="store_true", help="Display OSM Infrastructure & Stations Layer.")
    parser.add_argument("--border", action="store_true", help="Display Nuremberg Border Layer.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # If the user did not specify any specific flags, we automatically show all layers
    if not (args.rgb or args.ndvi or args.swir or args.labels or args.osm or args.border):
        args.rgb = args.labels = args.osm = args.border = True
        
    if args.task == "pred2020":
        data_dir = os.path.join("data", "training_data", "Composition_prediction_in_3_years", "2020")
    elif args.task == "pred2021":
        data_dir = os.path.join("data", "training_data", "Composition_prediction_in_3_years", "2021")
    elif args.task == "diff1yr":
        data_dir = os.path.join("data", "training_data", "Composition_diff_in_one_year")
        
    gpkg_file = os.path.join(data_dir, "training_data.gpkg")
    tif_file = os.path.join(data_dir, "training_data.tif")
    
    if not os.path.exists(gpkg_file) or not os.path.exists(tif_file):
        print(f"Error: Could not find {gpkg_file} or {tif_file}")
        return
        
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Full Dataset Visualisation - {args.task}")
    ax.set_xlabel("EPSG:3857 Easting")
    ax.set_ylabel("EPSG:3857 Northing")
    
    # --- RASTER RENDERING ---
    # GTiff raster arrays represent the full extent of the map natively.
    if args.rgb or args.ndvi or args.swir or args.labels:
        print("Loading GTiff Raster Matrices...")
        with rasterio.open(tif_file) as src:
            data = src.read()
            # Extent formats the array borders mapping exactly onto the Matplotlib X and Y axis limits
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            
            if args.rgb:
                print(" -> Rendering Sentinel-2 RGB (Bands 1-3)")
                rgb_data = data[:3]
                # Convert from 0-255 uint8 integers into 0.0-1.0 floats required by matplotlib
                rgb_img = np.transpose(rgb_data, (1, 2, 0)) / 255.0
                ax.imshow(rgb_img, extent=extent, zorder=1)
                
            if args.ndvi:
                print(" -> Rendering NDVI (Bands 4-6)")
                ndvi_data = data[3:6]
                ndvi_img = np.transpose(ndvi_data, (1, 2, 0)) / 255.0
                ax.imshow(ndvi_img, extent=extent, zorder=1)
                
            if args.swir:
                print(" -> Rendering SWIR (Bands 7-9)")
                swir_data = data[6:9]
                swir_img = np.transpose(swir_data, (1, 2, 0)) / 255.0
                ax.imshow(swir_img, extent=extent, zorder=1)
                
            if args.labels:
                print(" -> Rendering ESA Labels (Band 10)")
                # If the dataset has 10 bands, ESA is index 9. Otherwise, 3 (fallback).
                esa_idx = 9 if data.shape[0] >= 10 else 3
                esa_data = data[esa_idx]
                
                # Apply custom colors to map indices, converting strictly to uint8 to prevent backend downsampling bugs
                overlay = np.zeros((*esa_data.shape, 4), dtype=np.uint8)
                for cls_val, color in ESA_COLORS.items():
                    mask_arr = esa_data == cls_val
                    rgba_float = to_rgba(color)
                    rgba_color = [int(c * 255) for c in rgba_float]
                    # If any base is visible beneath it, make labels semi-transparent. Else make them opaque.
                    rgba_color[3] = 150 if (args.rgb or args.ndvi or args.swir) else 255
                    overlay[mask_arr] = rgba_color
                ax.imshow(overlay, extent=extent, zorder=2, interpolation='nearest') 
                
        # Force the axis bounds so the map renderer never gets confused by transparency
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
                
    # --- VECTOR RENDERING --- 
    # geopandas intrinsically supports slicing explicit layers right out of .gpkg
    if args.osm:
        print("Loading GPKG Vector Data (OSM)...")
        try:
            lines = gpd.read_file(gpkg_file, layer="osm_infrastructure")
            lines = lines.to_crs("EPSG:3857")
            lines.plot(ax=ax, color='green', linewidth=0.3, zorder=5, label="Infrastructure")
            print(f" -> Rendered {len(lines)} road/rail geometry lines")
        except Exception as e:
            print(f" -> Failed to read OSM lines: {e}")
            
        try:
            stations = gpd.read_file(gpkg_file, layer="osm_stations")
            stations = stations.to_crs("EPSG:3857")
            stations.plot(ax=ax, color='green', marker='*', markersize=5, zorder=6, label="Stations")
            print(f" -> Rendered {len(stations)} station coordinates")
        except Exception:
            pass
            
    if args.border:
        print("Loading GPKG Vector Data (Border)...")
        try:
            border_gdf = gpd.read_file(gpkg_file, layer="nuremberg_border")
            border_gdf = border_gdf.to_crs("EPSG:3857")
            border_gdf.plot(ax=ax, color='none', edgecolor='blue', linewidth=2, zorder=7, label="Border")
            print(" -> Rendered Nuremberg Administrative Boundary")
        except Exception as e:
            print(f" -> Failed to read border: {e}")
            
    print("\nDisplaying interactive matplotlib map...")
    plt.show()

if __name__ == "__main__":
    main()
