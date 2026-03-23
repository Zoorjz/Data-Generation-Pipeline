# Google Drive Data Archive Structure for Reproducibility

To ensure the examiner can easily evaluate and reproduce your entire pipeline without friction, your Google Drive archive should be structured into **three distinct tiers (checkpoints)**.

Because some steps in the pipeline require external API authentication (Google Earth Engine) or heavy computational time (Raster Feature Extraction), providing checkpoints allows the examiner to jump straight into any script or notebook instantly.

Here is the exact folder structure you should compile into a ZIP file for your Google Drive link, along with an explanation of why each file is there:

***

## Recommended Folder Structure for the Google Drive Archive

```text
📦 Data_Investigation_Archive
 ┣ 📂 1_Raw_Original_Data
 ┃ ┗ 📂 WORLDCOVER
 ┃   ┣ 📂 ESA_WORLDCOVER_10M_2020_V100
 ┃   ┃ ┗ 📂 MAP
 ┃   ┃   ┗ 📜 ESA_WorldCover_10m_2020_v100_N48E009_Map.tif
 ┃   ┗ 📂 ESA_WORLDCOVER_10M_2021_V200
 ┃     ┗ 📂 MAP
 ┃       ┗ 📜 ESA_WorldCover_10m_2021_v200_N48E009_Map.tif
 ┃
 ┣ 📂 2_Generated_Training_Data
 ┃ ┗ 📂 run_denoised_final
 ┃   ┣ 📂 Composition_prediction_in_3_years
 ┃   ┃ ┣ 📂 2020
 ┃   ┃ ┃ ┣ 📜 grid_stats_denoised.csv
 ┃   ┃ ┃ ┣ 📜 training_data_denoised.gpkg
 ┃   ┃ ┃ ┗ 📜 training_data_denoised.tif
 ┃   ┃ ┗ 📂 2021
 ┃   ┃   ┗ 📜 ... (grid, gpkg, tif)
 ┃   ┗ 📂 Composition_diff_in_one_year
 ┃     ┗ 📜 ... (grid, gpkg, tif)
 ┃
 ┗ 📂 3_ML_Features
   ┣ 📜 features_clean_2020_renamed.csv
   ┣ 📜 features_clean_2021_renamed.csv
   ┗ 📜 features_clean_diff_renamed.csv
   
 ┗ 📂 4_Trained_Models
   ┣ 📜 composition_random_forest.joblib
   ┣ 📜 change_random_forest.joblib
   ┗ 📜 ... (ridge, gradient_boosting, mlp .joblib files)
```

***

## What Each Tier Allows the Examiner to Do:

### Tier 1: `1_Raw_Original_Data` (The Starting Line)
*   **What it is:** The mandatory foundational data downloaded from the ESA website.
*   **Why include it:** If the examiner wants to test your *Data Collection script* (`generate_training_data.py`), they need these base files. Without them, the pipeline cannot calculate the composition percentages or the denoising mask.
*   **Pipeline Phase Supported:** `generate_training_data.py`

### Tier 2: `2_Generated_Training_Data` (The Extraction Checkpoint)
*   **What it is:** The output of your data generation script. This contains the heavily engineered `.tif` files (which have the Sentinel-2 bands, WorldCover classifications, and OSM data all stacked together) alongside the `grid_stats.csv` geospatial anchors.
*   **Why include it:** Many examiners will not want to authenticate their personal Google Accounts to Earth Engine just to test your feature extraction code. By providing this folder, the examiner can skip the download phase entirely and directly run your `FeatureExtraction.ipynb` notebook to see how your spatial math works.
*   **Pipeline Phase Supported:** `FeatureExtraction.ipynb`

### Tier 3: `3_ML_Features` (The Modeling Checkpoint)
*   **What it is:** The final, cleanly extracted, purely tabular CSV statistics matrices.
*   **Why include it:** This is the most important tier. Most examiners reviewing ML assignments just want to run the models and see the metric evaluations. By providing these lightweight CSVs, the examiner can open `modeling.ipynb`, hit "Run All", and instantly train the Random Forest algorithms to review your results in seconds, completely bypassing hours of GIS computations.
*   **Pipeline Phase Supported:** `modeling.ipynb` & Model Evaluation

### Tier 4: `4_Trained_Models` (The Deployment Checkpoint)
*   **What it is:** The serialized (`.joblib`) Machine Learning models that were successfully fit inside `modeling.ipynb`.
*   **Why include it:** Your pipeline contains `predict_new_data.py`, which is designed to predict land composition for entirely new, unseen years (like 2025). By providing the trained weights, the examiner/dashboard team doesn't actually have to rerun `modeling.ipynb` at all; they can just pipe the 2025 features straight into your `.joblib` files and instantly receive the predicted map masks!
*   **Pipeline Phase Supported:** `predict_new_data.py` & Dashboard Visualization

***

## Instructions for the Examiner (Add this to your README)
When you share the drive link, you can add a short instruction block like this:

> **Data Setup Instructions:**
> Download the provided `Data_Investigation_Archive.zip` from Google Drive. 
> 
> *   **To run predictions on new data immediately:** Use the models found in `4_Trained_Models` by passing them into `predict_new_data.py`.
> *   **To train the Models:** Simply place the contents of `3_ML_Features` into the `Features/` directory of the repository and run `modeling.ipynb`.
> *   **To test Feature Extraction:** Place the `2_Generated_Training_Data` folder into `data/training_data/` and run `FeatureExtraction.ipynb`.
> *   **To run the pipeline from scratch:** Place `1_Raw_Original_Data` into an `OriginalData/` folder at the root of the repository, log into Earth Engine, and execute `generate_training_data.py`.
