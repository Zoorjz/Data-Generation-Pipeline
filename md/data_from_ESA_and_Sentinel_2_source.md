# Sourcing Data from ESA and Sentinel-2

If you prefer to download the raw data straight from the source instead of using the pre-packed Google Drive zip, follow these instructions. 

## WorldCover Data

WorldCover data can be downloaded from the ESA website (account sign-in required).

- [Interactive Map Link](https://viewer.esa-worldcover.org/worldcover/?language=en&bbox=7.412657819210489,48.0074783190621,14.967628699151884,50.79749364599755&overlay=true&bgLayer=OSM&date=2026-03-20&layer=WORLDCOVER_2020_MAP)

**Required Data Products:**
* `ESA_WorldCover_10m_2021_v200_N48E009`
* `ESA_WorldCover_10m_2020_v100_N48E009`

### Folder Structure for WorldCover
It should be stored inside the `OriginalData` directory in the following hierarchical structure. Make sure you place the `.tif` maps in these exact nested paths so the pipeline can find them. Note that when the pipeline runs, it relies primarily on the structural information found in Sentinel's RGB and NIR, skipping unneeded layers:

```text
OriginalData/
└── WORLDCOVER/
    ├── ESA_WORLDCOVER_10M_2020_V100/
    │   └── MAP/
    │       └── ESA_WorldCover_10m_2020_v100_N48E009_Map/
    │           └── ESA_WorldCover_10m_2020_v100_N48E009_Map.tif
    │
    └── ESA_WORLDCOVER_10M_2021_V200/
        └── MAP/
            └── ESA_WorldCover_10m_2021_v200_N48E009_Map/
                └── ESA_WorldCover_10m_2021_v200_N48E009_Map.tif
```

## Sentinel-2 Data

Sentinel-2 data are handled programmatically and downloaded using the script `src/download_sentinel2_ee.py`.

### Google Earth Engine Authentication
For the data to be downloaded, the user needs to authenticate with Google Earth Engine:

1. If running on a new machine, a browser window will automatically open to authenticate with Google Earth Engine when the script starts. 
2. **Important:** Ensure you change the project name in `src/download_sentinel2_ee.py` to your own Google Cloud project ID if you are not a member of the original project team.

### Folder Structure after downloading Sentinel-2 Data
Once you run the pipeline (`run_pipeline.py`), the `download_sentinel2_ee.py` script will download the image data directly into the `data/` folder, sorting it into separate folders grouped by their respective date ranges.

Below is an overview of what the final `data/` workspace should look like. During the data generation and extraction steps, files that aren't strictly needed for the next layer of inference (such as `SWIR.tif` and `NDVI.tif`) are generally excluded from training and feature extraction, focusing the training process solely on `S2RGBNIR.tif` and similarly crucial files. Keep this in mind if modifying the pipeline code:

```text
data/
├── sentinel2_downloads_2017-06-01_2017-09-01/
│   └── Nuremberg_S2RGBNIR.tif
├── sentinel2_downloads_2018-06-01_2018-09-01/
│   └── Nuremberg_S2RGBNIR.tif
... continuing for other date ranges (2020, 2021, 2023, 2024, 2025)
│
└── training_data/
    └── run_denoised_<timestamp>/  (or run_<timestamp>)
        └── (Contains extracted patches, generated labels, and feature CSVs)
```

After completing these setup and structure requirements, you are ready to execute the main pipeline file `run_pipeline.py`.
