# ML Pipeline Architecture & Dataset Changes

## ⚠️ Important: Band Order Changes
Please note that the band order inside the `.tif` raster arrays has been updated from previous versions and is structurally different. Based on the extraction logic in the dataset tools (like `src/check_cell.py`), the final 10-band structure is explicitly mapped as follows:
* **Bands 1-3:** Sentinel-2 RGB Baseline 
* **Bands 4-6:** NDVI (Normalized Difference Vegetation Index)
* **Bands 7-9:** SWIR (Short-Wave Infrared)
* **Band 10:** Target ESA WorldCover Labels (Denoised or Raw, depending on the prediction task)

## What Was Before
Previously, the data generation script (`src/generate_training_data.py`) produced a generic ML dataset mapping identical years passively. The process was roughly:
* Collect Sentinel-2 images and ESA WorldCover labels for 2020 exactly into a `2020` statistical array.
* Collect Sentinel-2 images and ESA WorldCover labels for 2021 exactly into a `2021` statistical array.

While this provided great map overlays, it left machine learning models without a clear objective, simply offering isolated identical-year data without built-in prediction boundaries or clear delta calculations.

## What Changed
The generation script has been completely refactored to compile isolated datasets representing three distinct **Machine Learning Prediction Tasks**. Instead of just storing files by their generic year, the system intentionally pairs historical input data directly with target future-outcome labels to create structured objective functions for specific models.

The new structure guarantees the following target datasets are exported directly to `data/training_data/`:

### 1. `Composition_prediction_in_3_years`
This task trains a model to predict landscape composition exactly 3 years into the future using ONLY old Sentinel satellite history.
* **`2020` Directory:** Maps 2017 Sentinel/OSM Input Data ➡️ Predicts 2020 Denoised ESA WorldCover Labels.
* **`2021` Directory:** Maps 2018 Sentinel/OSM Input Data ➡️ Predicts 2021 Raw ESA WorldCover Labels.

### 2. `Composition_diff_in_one_year`
This task trains a model to predict how much the landscape will actively grow or change construction over a single year.
* Maps 2020 Sentinel/OSM Data alongside the base 2020 Denoised Labels ➡️ Predicts the specific mathematical difference (delta %) mapping 2021 minus 2020 labels for every grid cell. 

*(To prevent learning "fake" city growth and noise, grass-to-building noise is conservatively deleted from the target predictions prior to delta calculations).*

These tasks are cleanly mapped into their respective `.tif` raster structures and geometry formats safely decoupled from raw raw Earth Engine noise.
