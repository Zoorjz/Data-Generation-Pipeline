1. need to describe how to reproduce the results
2. describe how to get the data. (worldCover, Sentinel-2)
3. how to run the data generation pipline
4. how to run featrue extraction (we have notebook maybe create a script?)
5. how to train the models (we have notebook)
6. how to evaluate the models (we don't have notebook)
7. how to visualize the results (we will have a web app)


WorldCover :
WorldCover data can be downloaded from the ESA website. (sign in required)

https://viewer.esa-worldcover.org/worldcover/?language=en&bbox=7.412657819210489,48.0074783190621,14.967628699151884,50.79749364599755&overlay=true&bgLayer=OSM&date=2026-03-20&layer=WORLDCOVER_2020_MAP

the required data :
ESA_WorldCover_10m_2021_v200_N48E009
ESA_WorldCover_10m_2020_v100_N48E009

This should be stored in OriginalData folder in this format:
--explain here how it should be stored based on the script @generate_training_data.py like what the tree structure of the folder etc but exclude the files that are not needed so like SWIR, NDVI, S2RGBNIR ---

Sentinel 2 

Sentinel 2 data are downloaded using the script src/download_sentinel2_ee.py

For the data to be downloaded the user needs to authenticate with Google Earth Engine. 

If running on a new machine, a browser window will open to authenticate with Google Earth Engine. Ensure you change the project name in 

download_sentinel2_ee.py
 to your own Google Cloud project ID if you are not a member of the original team.


**how to run the data generation pipline**

Explain how to run generate_training_data.py

**how to run featrue extraction**

Explain how to run the notebooks in the Features folder
FeaturesExtraction.ipynb

**