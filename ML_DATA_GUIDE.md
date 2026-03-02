# Machine Learning Data Processing Guide

This guide details how to verify, view, and integrate the standalone ML training datasets provided to you in the `data/training_data/` folder.

## 1. Working With The Data (Example Integration)
For ML models (like PyTorch / scikit-learn), you parse the `csv` for metadata and calculate loss masks dynamically off the `.gpkg` numpy matrices.

See `src/load_ml_data_example.py` for a working implementation of extracting exactly one grid cell's dataset.

---

## 2. Visual Verification (`check_cell.py`)
Provides an immediate visual and statistical overlay to verify the generated data for a specific cell coordinate from the generated grid records.

**Command:**
```bash
python src/check_cell.py --task diff1yr --row 20 --col 16
```
### Parameters:
- `--task`: The target dataset task folder (`pred2020`, `pred2021`, or `diff1yr`).
- `--row`: The target cell row.
- `--col`: The target cell column.
- `--rgb`: (Optional) Display Sentinel-2 RGB as base.
- `--ndvi`: (Optional) Display NDVI as base.
- `--swir`: (Optional) Display SWIR as base.
- `--base_only`: (Optional) Only display the base raster map, suppressing ESA overlay and OSM vector data.

### Behavior:
1. Validates the index against the `.csv` records and prints all statistical compositions directly for that specific ML task.
2. Extracts OSM vector strings from the `.gpkg` mathematically within the bounds and prints a terminal list of found infrastructure elements and stations.
3. Opens a visual Map Plot layering Sentinel-2 RGB baseline -> ESA Labels (Alpha 0.5) -> Infrastructure Vectors (Red Polygon bounds).

---

## 3. Viewing the Entire Data Area Map (`view_full_map_example.py`)
This script loads the massive Sentinel-2 & ESA arrays mathematically and overlays OSM/Nuremberg border vectors across the entire composite 4km+ array safely.

**Command:**
```bash
python src/view_full_map_example.py --task pred2021
```

### Parameters:
You can specifically choose which individual layers you want to extract by passing their arguments directly:
- `--task`: The target dataset task folder (`pred2020`, `pred2021`, or `diff1yr`).
- `--rgb`: Renders only the continuous Sentinel-2 RGB Image.
- `--ndvi`: Renders only the continuous NDVI Image.
- `--swir`: Renders only the continuous SWIR Image.
- `--labels`: Renders only the continuous ESA WorldCover classification data overlay.
- `--osm`: Isolates and dynamically renders hundreds of thousands of vector infrastructure nodes simultaneously in green.
- `--border`: Casts the physical boundary of Nuremberg in bright blue.

---

## 4. Viewing Raw `.tif` Band Layers Directly (`inspect_tif.py`)
A fast developer tool to immediately inspect raw tensors directly without projecting or map manipulations. Great for diagnosing models or quickly seeing isolated band masks.

**Command:**
```bash
python src/inspect_tif.py data/training_data/Composition_diff_in_one_year/training_data.tif --bands 1 2 3
```
- `--bands 1 2 3`: Will plot an RGB color mapped image natively utilizing standard matplotlib float matrices.
- `--bands 10`: Will visually slice exactly layer 10 (ESA Labels) exclusively as an independent grayscale plot.

---

## 5. Generating New Data (`generate_training_data.py`) - (Maintainer Only)

**⚠️ Note for ML Engineers:** The raw uncompiled source data (Gigabytes of raw Map GeoTIFs / ESA S2RGBNIR bounds) required to run this script **is not included** in the team data bundle to save space. 
*If you need entirely new datasets generated, different grid-sizes extracted, or custom ESA labels mapped, please contact me directly, and I will generate and send you a new zip bundle.*

This script independently pulls raw datasets and compiles them into three heavily structured predictive Machine Learning modeling tasks. It isolates Sentinel/OSM dates safely mapped across corresponding predictive Denoised WorldCover Labels (refer to `changes.md` for task boundaries).

**Command:**
```bash
python src/generate_training_data.py --grid_size 1000 --labels Built-up "Tree cover" Water
```

### Parameters:
- `--grid_size`: The math-perfect bounding box slice size in meters (e.g., `1000` = 1km x 1km cells).
- `--labels`: Space-separated list of ESA classes to calculate. *Strings with spaces must be quoted.*
- `--out_dir`: (Optional) Custom output directory. Default is `data/training_data`.

### Output Structure:
The command structurally extracts predictions natively into directories mapping `Composition_prediction_in_3_years` and `Composition_diff_in_one_year`:
- **`training_data.tif`**: GeoTIFFs storing the actual cell raster arrays. 
    - **10-Band Raster Matrix:** Band 1-3 = Sentinel-2 RGB, Band 4-6 = NDVI, Band 7-9 = SWIR, Band 10 = Selected Target ESA Labels.
- **`training_data.gpkg`**: Geospatial packages storing bounding geometry metadata.
    - **Vector layers:** `osm_infrastructure` (line geometries) and `osm_stations` (point geometries).
- **`grid_stats.csv`**: A structured database mapping every individual grid cell capturing `row`/`col`, `min/max lat/lon` bounding parameters mapped to `EPSG:4326`, `tif_file`/`gpkg_file` relative routing paths, and label representation inputs explicitly linked mapping prediction structures (`Built-up %: 43.1`, `delta Built-up %: +2.1`).

---

## 6. Sharing the ML Data (`create_ml_bundle.py`)
This script auto-compiles all isolated data structures, exclusively extracting the nested `Composition...` `.gpkg`/`.csv` matrices and all five required Python helper scripts directly into a portable compressed `.zip` payload you can pass safely offline.

**Command:**
```bash
python src/create_ml_bundle.py
```
This generates `ml_training_data_bundle_YYYYMMDD_HHMMSS.zip` directly in your project root ready to distribute!
