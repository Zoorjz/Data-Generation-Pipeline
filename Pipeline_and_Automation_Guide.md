# Pipeline and Automation Guide

## Overview
The data pipeline consists of two main stages represented by the two Jupyter notebooks:
1. **Feature Extraction (`FeatureExtraction.ipynb`)**: Converts raw GeoTIFF satellite imagery into tabular numerical features (band means and standard deviations) based on bounding box coordinates from a CSV.
2. **Modeling (`modeling.ipynb`)**: Uses those extracted features to train multiple machine learning models (Ridge Regression, Random Forest, Gradient Boosted Trees, MLP) to predict land composition (Task A) and land composition change (Task B) across different years.

---

## 1. Feature Extraction (`FeatureExtraction.ipynb`)

### What it does
The notebook calculates statistical features from `.tif` files. It reads a list of geographical cells defined in a `.csv` file (using `min_lon`, `min_lat`, `max_lon`, `max_lat`). For each cell, it crops the pixel data of 9 bands (RGB, NDVI, SWIR, etc.), excludes completely transparent zeros, and calculates 18 explicit features (Mean and Standard Deviation for all 9 bands).

### How to use it manually
Currently, the extraction is done by calling the `extract_features()` function in the notebook.
You must provide:
- `csv_path`: The `grid_stats.csv` which defines all grid cells.
- `tif_path`: The satellite `.tif` file you want to analyze.
- `out_csv`: The output file name where the extracted data will go (e.g., `features_clean_2022.csv`).
- `tag`: A string (like `"2022"`) appended to each band column name (e.g., `band1_mean_2022`).

---

## 2. Modeling (`modeling.ipynb`)

### What it does
It loads the outputs created in the Feature Extraction step and merges them with target `%` classes (like Built-up, Water). Once merged, it trains 4 models to accomplish two tasks:
1. **Task A (Composition Prediction)**: Given Sentinel imagery features from a specific year, what is the exact percentage breakdown of land coverage within that cell right now?
2. **Task B (Change Prediction)**: Given Sentinel imagery features + baseline percentages from year `X`, how much will the terrain change (`delta %`) in year `Y`?

### Is there a preferable model?
Yes, the notebook systematically evaluates all 4 models and the outputs conclude:
- **Best Model for Task A (Composition)**: **MLP** (Multi-Layer Perceptron neural network) achieved the highest R2 score and lowest Mean Absolute Error (MAE).
- **Best Model for Task B (Change)**: **Random Forest** achieved the best scores for predicting 1-year deltas.

### How to input new data and receive predictions
The notebooks are strictly tailored for model training and grid CV validation. The final cells of `modeling.ipynb` **save all trained models completely entirely to your disk** as `.joblib` files inside the `outputs/models/` directory.

To predict on completely new data, you **do not** need to rerun the modeling notebook. You only need to:
1. Extract features for the new year.
2. Load the preferred `.joblib` file.
3. Compute 4 minor engineered features `lat`, `lon`, `NDVI_veg_ratio`, and `SWIR_moisture_ratio`.
4. Run `predict()`.

---

## 3. Recommended Manual Pipeline Execution 
If you want to prepare and train data manually by clicking inside the notebook:

**Step 1: Extract Features**
1. Open `FeatureExtraction.ipynb`.
2. Duplicate the bottom cell that calls `extract_features(...)`.
3. Update the inputs to point to the new years (e.g., 2022 or 2023 grid boundaries).
4. Run the cell and locate the generated output CSV.

**Step 2: Train Models**
1. Open `modeling.ipynb`.
2. Locate the first execution cell and update paths `BASE`, `FEAT`, and `DATA` so the notebook can find your extracted CSV files and the target stats CSV files.
3. "Run All Cells" from the top.
4. The notebook will automatically calculate accuracy, visualize metrics, and dump `.joblib` models into `outputs/models`.

---

## 4. Automation Implementations (Zero Notebook Edits Required)
To automate producing features and predicting over new years, it is highly recommended to completely step outside of the `.ipynb` ecosystem via standalone Python driver scripts.

### Automation A: Command-Line Feature Extraction
Create a file named `extract_features_cli.py` and run it via terminal instead of opening the notebook.

```python
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

            for i in range(9):
                band_index = i + 1
                feat[f"band{band_index}_mean_{args.tag}"] = float(np.mean(flat[i]))
                feat[f"band{band_index}_std_{args.tag}"] = float(np.std(flat[i]))
            rows.append(feat)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} | shape={out.shape}")

if __name__ == "__main__":
    main()
```
**Usage**: `python extract_features_cli.py --csv_path grid_stats.csv --tif_path 2022_image.tif --out_csv features_2022.csv --tag 2022`

### Automation B: Predicting from Pre-Trained Models
Create a script `predict_new_data.py`. This reads your newly extracted characteristics, formats them exactly as the model expects, and prints predictions without retraining anything.

```python
import argparse
import pandas as pd
import joblib

def engineer_features(df):
    df['lat'] = (df['min_lat'] + df['max_lat']) / 2
    df['lon'] = (df['min_lon'] + df['max_lon']) / 2
    df['NDVI_veg_ratio'] = df['NDVI_1_mean'] / (df['R_mean'] + 1e-6)
    df['SWIR_moisture_ratio'] = df['SWIR_1_mean'] / (df['G_mean'] + 1e-6)
    return df

def predict_composition(features_csv, target_csv, model_path):
    print("Loading data...")
    feats = pd.read_csv(features_csv)
    targets = pd.read_csv(target_csv)[['cell_id', 'min_lon', 'min_lat', 'max_lon', 'max_lat']]
    
    # Clean the suffix tag dynamically
    features = feats.rename(columns=lambda c: c.split('_2')[0] if '_2' in c else c)
    features = features.rename(columns=lambda c: c.split('_clean')[0] if '_clean' in c else c)
    
    df = features.merge(targets, on='cell_id')
    df = engineer_features(df)
    
    # Must explicitly match the sequence the model was trained on
    SPECTRAL = ['R_mean', 'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std', 
                'NDVI_1_mean', 'NDVI_1_std', 'NDVI_2_mean', 'NDVI_2_std', 
                'SWIR_1_mean', 'SWIR_1_std', 'SWIR_2_mean', 'SWIR_2_std', 
                'SWIR_3_mean', 'SWIR_3_std'] # NDVI_3 is discarded in training
                
    FINAL_FEATURES = SPECTRAL + ['lat', 'lon', 'NDVI_veg_ratio', 'SWIR_moisture_ratio']
    X_new = df[FINAL_FEATURES].values
    
    print(f"Loading Model: {model_path}...")
    model = joblib.load(model_path)
    
    # Predict
    preds = np.clip(model.predict(X_new), 0, 100)
    df['Predicted Built-up %'] = preds[:, 0]
    df['Predicted Water %'] = preds[:, 1]
    df['Predicted Other %'] = 100 - df['Predicted Built-up %'] - df['Predicted Water %']
    
    out_file = "new_predictions.csv"
    df[['cell_id', 'Predicted Built-up %', 'Predicted Water %', 'Predicted Other %']].to_csv(out_file, index=False)
    print(f"Predictions saved successfully to {out_file}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_csv', required=True)
    parser.add_argument('--target_csv', required=True) # Required to establish lat/lons
    parser.add_argument('--model_path', required=True, help="Path to joblib (e.g., outputs/models/composition_mlp.joblib)")
    args = parser.parse_args()
    
    predict_composition(args.features_csv, args.target_csv, args.model_path)
```
**Usage**: `python predict_new_data.py --features_csv features_2022.csv --target_csv grid_stats_2022.csv --model_path outputs/models/composition_mlp.joblib`
