"""
Example script demonstrating how to load and use the generated Machine Learning datasets.
This script shows how to:
1. Load the CSV containing the grid definitions and statistics.
2. Read the corresponding GPKG to extract the Sentinel-2 and ESA rasters for a specific cell.
3. Extract the OSM vector data for that cell.
"""

import os
import pandas as pd
import rasterio
import rasterio.mask
import geopandas as gpd
from shapely.geometry import box
import numpy as np

def main():
    # 1. Define paths and load the CSV
    data_dir = os.path.join("data", "training_data")
    csv_path = os.path.join(data_dir, "grid_stats_2021.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Did you run generate_training_data.py?")
        return
        
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 2. Pick a random cell that actually contains some Built-up area
    # Example: Find cells where 'Built-up %' is greater than 10%
    if "Built-up %" in df.columns:
        builtup_cells = df[df["Built-up %"] > 10.0]
        if not builtup_cells.empty:
            target_cell = builtup_cells.iloc[0]
        else:
            target_cell = df.iloc[0]
    else:
        target_cell = df.iloc[0]
        
    print(f"\n--- Selected Cell (Row: {target_cell['row']}, Col: {target_cell['col']}) ---")
    print(f"Built-up Percentage: {target_cell.get('Built-up %', 'N/A')}%")
    print(f"Tree cover Percentage: {target_cell.get('Tree cover %', 'N/A')}%")
    
    # 3. Create the bounding box geometry for extracting data
    cell_box = box(
        target_cell["min_lon"], target_cell["min_lat"], 
        target_cell["max_lon"], target_cell["max_lat"]
    )
    # The GPKG stores raster/vector data in EPSG:3857, so we must project our EPSG:4326 box
    bbox_gdf = gpd.GeoSeries([cell_box], crs="EPSG:4326")
    bbox_3857 = bbox_gdf.to_crs("EPSG:3857").iloc[0]
    
    # 4. Open the GeoTIFF and GPKG files referenced by this cell
    gpkg_path = os.path.join(data_dir, target_cell["gpkg_file"])
    tif_path = os.path.join(data_dir, target_cell.get("tif_file", f"training_data_{target_cell['gpkg_file'][-9:-5]}.tif"))
    
    print(f"\nExtracting raster matrix from: {tif_path}")
    try:
        with rasterio.open(tif_path) as src:
            # Mask (crop) the large raster down to just our specific cell
            out_image, out_transform = rasterio.mask.mask(src, [bbox_3857], crop=True)
            
            # The matrix shape is (Bands, Height, Width)
            print(f"Successfully loaded tensor of shape: {out_image.shape}")
            
            if out_image.shape[0] >= 10:
                print(" - Band 1-3: Sentinel-2 RGB (uint8, 0-255)")
                print(" - Band 4-6: NDVI (uint8, 0-255)")
                print(" - Band 7-9: SWIR (uint8, 0-255)")
                print(" - Band 10: ESA WorldCover class labels")
                
                # Split into ML inputs and targets
                ml_input_rgb = out_image[:3] 
                ml_input_ndvi = out_image[3:6]
                ml_input_swir = out_image[6:9]
                ml_target_mask = out_image[9]
                
                print(f"RGB Input Shape: {ml_input_rgb.shape}")
                print(f"NDVI Input Shape: {ml_input_ndvi.shape}")
                print(f"Target Mask Shape: {ml_target_mask.shape}")
                
                # Example: Normalize RGB array for a Neural Network (0.0 to 1.0)
                normalized_input = ml_input_rgb.astype(np.float32) / 255.0
                
    except Exception as e:
        print(f"Error loading raster: {e}")
        
    # 5. Extracting corresponding OSM Vector Data (e.g. for Graph Neural Networks)
    print("\nExtracting vector data...")
    try:
        # Load infrastructure lines
        lines = gpd.read_file(gpkg_path, layer="osm_infrastructure")
        lines = lines.to_crs("EPSG:3857")
        # Clip lines strictly to the cell boundaries
        lines_clipped = gpd.clip(lines, bbox_3857)
        
        print(f"Found {len(lines_clipped)} road/rail geometry segments.")
        if not lines_clipped.empty:
            print("Types of infrastructure present:", lines_clipped['highway'].dropna().unique())
            
    except Exception as e:
        print(f"Error loading vectors: {e}")

if __name__ == "__main__":
    main()
