# Data Generation Pipeline

This repository contains the data generation pipeline for creating Machine Learning datasets to predict land cover changes in Nuremberg over time. 

The pipeline fetches satellite imagery from Google Earth Engine (Sentinel-2), downloads OpenStreetMap infrastructure data, and integrates it with ESA WorldCover ground truth maps to generate a clean, unified dataset ready for training ML models.

## 📊 Pipeline Output & Purpose

The ultimate objective of this pipeline is to structure raw imagery and map data into **Machine Learning-ready tabular datasets (CSV)** alongside aligned spatial maps (GeoTIFFs & GeoPackages).

When the pipeline completes successfully, it generates three data folders inside `data/training_data/`:
1. `Composition_prediction_in_3_years/2020` (Predicting 2020 layout from 2017 inputs)
2. `Composition_prediction_in_3_years/2021` (Predicting 2021 layout from 2018 inputs)
3. `Composition_diff_in_one_year/` (Learning the 1-year differential change between 2020-2021)

Inside each folder, you will find:
- **`grid_stats.csv`**: This is the core ML target file! It contains numerical tables where each row is a physical grid square (e.g., 1000m x 1000m). It calculates the exact percentage composition of chosen labels (like *Built-up* or *Permanent water bodies*) alongside coordinate bounding boxes.
- **`training_data.tif`**: A massive multi-band raster compiling Sentinel-2 (RGB, NIR, SWIR), NDVI vegetation metrics, and the ESA truth map cleanly overlaid into one file.
- **`training_data.gpkg`**: A vector representation containing the OpenStreetMap railways/highways, transport stations, and the Nuremberg city border intersecting the grids.

These artifacts are designed to be ingested by ML frameworks (PyTorch, TensorFlow, Scikit-Learn) to learn how infrastructure and spectral bands predict urbanization or environmental shifts.
## How to Run the Pipeline

To run the complete pipeline automatically, you can use the central orchestration script:

```bash
# Activate your virtual environment
.\venv\Scripts\Activate.ps1

# Run the master pipeline script
python run_pipeline.py --denoise
# or
python run_pipeline.py --with_noise
```

### What `run_pipeline.py` does:
1. **Validates Requirements:** Checks if the required ESA WorldCover base maps for 2020 and 2021 are downloaded and present in the `OriginalData` folder.
2. **Downloads Imagery:** Connects to Google Earth Engine and sequentially downloads Sentinel-2 cloud-free composites (S2RGBNIR, NDVI, SWIR) for the years 2017, 2018, 2020, and 2021. It includes an 8-minute timeout per range to prevent the script from hanging on Earth Engine API issues, and automatically retries if it fails.
3. **Generates Grids:** Slices the geographical data into a tabular dataset in `data/training_data`, calculates the percentage of "Built-up" and "Permanent water bodies" areas per grid cell, removes noise from the 2020 baseline, and structures the target labels for predicting 3 years into the future or 1 year into the future.

---

## 🌍 How to Access the Required WorldCover Data

Before running the pipeline, you **must manually download** the ESA WorldCover high-resolution maps for Nuremberg, as they are too large to host directly in Git.

1. Go to the ESA WorldCover Viewer: [https://viewer.esa-worldcover.org/worldcover/](https://viewer.esa-worldcover.org/worldcover/)
2. Register for **VITO’s Terrascope platform** (click the link in the top right to create an account).
3. Log in to the WorldCover viewer using your new Terrascope credentials.
4. Using the map or search function, navigate to **Nürnberg** (Nuremberg).
5. Use the download tools to download the following exact map tiles for the region:
   - `ESA_WorldCover_10m_2020_v100_N48E009` (2020 Version 100)
   - `ESA_WorldCover_10m_2021_v200_N48E009` (2021 Version 200)

Extract and place these map tiles inside your local `OriginalData/WORLDCOVER/` directory so the scripts can find them safely. *(Note: The pipeline automatically handles the Windows 260-character path limit when accessing these deep folder structures).*

---

## 🛠 Helpful Scripts & Flags

If you want to run specific parts of the pipeline manually or customize parameters, you can use the individual scripts.

### 1. Sentinel-2 Downloader
Downloads a median, cloud-free composite from Earth Engine for a specific date range.

**Command:**
```bash
python src/download_sentinel2_ee.py --start_date 2020-06-01 --end_date 2020-09-01
```

**Flags:**
* `--start_date`: Start date of the collection timeframe (default: `2020-06-01`)
* `--end_date`: End date of the collection timeframe (default: `2020-09-01`)

*Note: For Windows users, make sure your Earth Engine project is initialized and authenticated!*

### 2. Training Data Generator
Aligns the spatial maps, fetches dynamic OSM data, handles noise reduction, masks out labels, and exports the final numerical grid and GeoPackages.

**Command:**
```bash
python src/generate_training_data.py --grid_size 1000 --labels "Built-up" "Permanent water bodies" --denoise
```

**Flags:**
* `--grid_size`: Size of the resulting square grid blocks in meters (default: `1000`)
* `--labels`: Space-separated list of ESA WorldCover labels to compute percentages for. Example: `--labels "Built-up" "Tree cover" "Permanent water bodies"`
* `--out_dir`: Directory where the final datasets will be saved (default: `data/training_data`)
* `--denoise`: Apply denoising filtering to 2020 map for expanding infrastructure
* `--with_noise`: Choose to leave 2020 baseline maps strictly matching raw outputs
