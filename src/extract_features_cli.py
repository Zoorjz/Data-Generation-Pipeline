import argparse
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.windows import from_bounds

def main():
    parser = argparse.ArgumentParser(description="Extract ML features from Grid stats and TIF")
    parser.add_argument('--csv_path', required=True, help="Input Grid Stats CSV")
    parser.add_argument('--tif_path', required=True, help="Input Satellite TIF Image")
    parser.add_argument('--out_csv', required=True, help="Path to save output features CSV")
    parser.add_argument('--tag', required=True, help="Suffix for columns (e.g. '2022' or 'diff')")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    rows = []

    with rasterio.open(args.tif_path) as src:
        total = len(df)
        for idx, cell in df.iterrows():
            if idx % 100 == 0: print(f"Processing cell {idx}/{total}...")
            
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
                feat[f"{prefix}_mean_{args.tag}"] = float(np.mean(flat[i]))
                feat[f"{prefix}_std_{args.tag}"] = float(np.std(flat[i]))
            rows.append(feat)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} | shape={out.shape}")

if __name__ == "__main__":
    main()