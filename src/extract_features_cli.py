import os
import glob
import argparse
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.windows import from_bounds

def extract_features(csv_path, tif_path, out_csv, tag):
    df = pd.read_csv(csv_path)
    rows = []

    with rasterio.open(tif_path) as src:
        total = len(df)
        for idx, cell in df.iterrows():
            if idx % 100 == 0: print(f"Processing cell {idx}/{total} [{tag}]...")
            
            bbox_4326 = box(cell["min_lon"], cell["min_lat"], cell["max_lon"], cell["max_lat"])
            bbox = gpd.GeoSeries([bbox_4326], crs="EPSG:4326").to_crs(src.crs).iloc[0]
            minx, miny, maxx, maxy = bbox.bounds
            win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)

            data = src.read(list(range(1, 10)), window=win).astype(np.float32)
            flat = data.reshape(9, -1)
            valid = ~(np.all(flat == 0, axis=0))
            if valid.any(): flat = flat[:, valid]

            feat = {
                "cell_id": int(cell["cell_id"]),
                "row": int(cell["row"]),
                "col": int(cell["col"]),
            }

            band_names = ["R", "G", "B", "NDVI_1", "NDVI_2", "NDVI_3", "SWIR_1", "SWIR_2", "SWIR_3"]
            for i in range(9):
                prefix = band_names[i]
                feat[f"{prefix}_mean_{tag}"] = float(np.mean(flat[i]))
                feat[f"{prefix}_std_{tag}"] = float(np.std(flat[i]))
            rows.append(feat)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} | shape={out.shape}")

def auto_extract(base_dir="."):
    training_data_dir = os.path.join(base_dir, "data", "training_data")
    if not os.path.exists(training_data_dir):
        print(f"Error: {training_data_dir} not found.")
        return
        
    # Find latest run
    runs = glob.glob(os.path.join(training_data_dir, "run_*"))
    if not runs:
        print(f"No run folders found in {training_data_dir}")
        return
        
    latest_run = max(runs, key=os.path.getctime)
    print(f"Auto-detecting latest run: {latest_run}")
    
    # We must support both denoised and with_noise naming dynamically
    suffix = "_denoised" if "denoised" in latest_run else "_with_noise"
    
    jobs = [
        # Diff Extraction
        {
            "csv": os.path.join(latest_run, "Composition_diff_in_one_year", f"grid_stats{suffix}.csv"),
            "tif": os.path.join(latest_run, "Composition_diff_in_one_year", f"training_data{suffix}.tif"),
            "out": os.path.join(base_dir, "Features", f"features{suffix}_diff.csv"),
            "tag": "diff"
        },
        # 2020 Extraction 
        {
            "csv": os.path.join(latest_run, "Composition_prediction_in_3_years", "2020", f"grid_stats{suffix}.csv"),
            "tif": os.path.join(latest_run, "Composition_prediction_in_3_years", "2020", f"training_data{suffix}.tif"),
            "out": os.path.join(base_dir, "Features", f"features{suffix}_2020.csv"),
            "tag": "2020"
        },
        # 2021 Extraction
        {
            "csv": os.path.join(latest_run, "Composition_prediction_in_3_years", "2021", f"grid_stats{suffix}.csv"),
            "tif": os.path.join(latest_run, "Composition_prediction_in_3_years", "2021", f"training_data{suffix}.tif"),
            "out": os.path.join(base_dir, "Features", f"features{suffix}_2021.csv"),
            "tag": "2021"
        }
    ]
    
    for job in jobs:
        if os.path.exists(job["csv"]) and os.path.exists(job["tif"]):
            print(f"\n--- Extracting {job['tag']} Task ---")
            extract_features(job["csv"], job["tif"], job["out"], job["tag"])
        else:
            print(f"\nMissing files for {job['tag']}, skipping...")
            
    print("\nAll extractions complete! Results outputted to /Features/")

def main():
    parser = argparse.ArgumentParser(description="Extract ML features from Grid stats and TIF")
    parser.add_argument('--csv_path', help="Input Grid Stats CSV (optional if auto-detecting)")
    parser.add_argument('--tif_path', help="Input Satellite TIF Image")
    parser.add_argument('--out_csv', help="Path to save output features CSV")
    parser.add_argument('--tag', help="Suffix for columns (e.g. '2022' or 'diff')")
    parser.add_argument('--auto', action='store_true', help="Automatically find latest run and extract everything to Features/")
    args = parser.parse_args()

    # If no manual paths are given or --auto is invoked
    if args.auto or not args.csv_path:
        print("Running in Auto-Detection Mode...")
        # Since script runs from root or Data Gen Pipeline, base is one up if inside src 
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        auto_extract(base_dir)
    else:
        # Fallback to manual mode to keep orchestration scripts unbroken
        if not all([args.csv_path, args.tif_path, args.out_csv, args.tag]):
            parser.error("If not using --auto, you must provide --csv_path, --tif_path, --out_csv, and --tag")
        extract_features(args.csv_path, args.tif_path, args.out_csv, args.tag)

if __name__ == "__main__":
    main()