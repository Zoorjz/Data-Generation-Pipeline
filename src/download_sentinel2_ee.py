import ee
import os
import requests
import zipfile
import io
import sys
import os
import argparse
import numpy as np
from generate_training_data import get_nuremberg_bbox, get_align_params

def authenticate_and_initialize():
    try:
        ee.Initialize()
    except Exception as e:
        print("Authentication required...")
        ee.Authenticate()
        ee.Initialize(project = 'ml-nuremberg-project')

import rasterio
from rasterio.merge import merge as rio_merge
import glob

def get_chunked_transforms(crs_transform, dimensions, chunk_size=1024):
    """
    Given a global transform and dimensions, yields parameters for sub-chunks.
    """
    scale_x, shear_x, offset_x, shear_y, scale_y, offset_y = crs_transform
    width, height = dimensions
    
    chunks = []
    for start_y in range(0, height, chunk_size):
        for start_x in range(0, width, chunk_size):
            end_y = min(start_y + chunk_size, height)
            end_x = min(start_x + chunk_size, width)
            
            sub_width = end_x - start_x
            sub_height = end_y - start_y
            
            # Compute new offsets for this chunk
            sub_offset_x = offset_x + (start_x * scale_x) + (start_y * shear_x)
            sub_offset_y = offset_y + (start_x * shear_y) + (start_y * scale_y)
            
            sub_transform = [scale_x, shear_x, sub_offset_x, shear_y, scale_y, sub_offset_y]
            
            chunks.append({
                'row': start_y // chunk_size,
                'col': start_x // chunk_size,
                'transform': sub_transform,
                'dimensions': [sub_width, sub_height]
            })
    return chunks

def download_tiled_geometry(image, name, folder, params, default_crs, chunk_size=1024):
    """
    A helper function to download an image in chunks/tiles and reconstruct via rasterio.
    Works for both single bands and multiband visually exported images.
    """
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, f"{name}.tif")
    
    # We must have exact grid dimensions to tile precisely
    if 'crs_transform' not in params or 'dimensions' not in params:
        raise ValueError("crs_transform and dimensions are required for chunked precise downloads.")
        
    global_transform = params['crs_transform']
    global_dim = [int(v) for v in params['dimensions'].split('x')]
    
    chunks = get_chunked_transforms(global_transform, global_dim, chunk_size)
    print(f"Downloading '{name}' in {len(chunks)} tiles to respect size limits...")
    
    chunk_files = []
    for i, c in enumerate(chunks):
        chunk_path = os.path.join(folder, f"{name}_tile_{c['row']}_{c['col']}.tif")
        chunk_files.append(chunk_path)
        
        # Build GEE params for this chunk
        chunk_params = {
            'format': 'GEO_TIFF',
            'crs': params.get('crs', default_crs),
            'crs_transform': c['transform'],
            'dimensions': f"{c['dimensions'][0]}x{c['dimensions'][1]}"
        }
        
        print(f"  Downloading tile {i + 1}/{len(chunks)}...")
        try:
            url = image.getDownloadURL(chunk_params)
            response = requests.get(url)
            response.raise_for_status()
            
            if response.content[:4] == b'PK\x03\x04': 
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    tif_names = [n for n in z.namelist() if n.endswith('.tif')]
                    if tif_names:
                        with open(chunk_path, 'wb') as f:
                            f.write(z.read(tif_names[0]))
            else:
                with open(chunk_path, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            print(f"  Failed tile {i+1}: {e}")
            
    print(f"  Merging tiles into {name}...")
    try:
        src_files_to_mosaic = []
        for fp in chunk_files:
            if os.path.exists(fp):
                src = rasterio.open(fp)
                src_files_to_mosaic.append(src)
                
        if not src_files_to_mosaic:
            print(f"No valid tiles received for {name}.")
            return

        mosaic, out_trans = rio_merge(src_files_to_mosaic)
        
        # Copy the metadata
        out_meta = src_files_to_mosaic[0].meta.copy()
        
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": mosaic.shape[0] # Ensure number of output bands aligns mapping
        })
        
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(mosaic)
            
        print(f"Successfully saved merged image to {out_path}")
        
    except Exception as e:
        print(f"Error merging tiles for {name}: {e}")
    finally:
        # Cleanup
        for src in src_files_to_mosaic:
            src.close()
        for fp in chunk_files:
            if os.path.exists(fp):
                os.remove(fp)

def download_image_local(image, name, folder='data/sentinel2_downloads', crs='EPSG:3857', crs_transform=None, dimensions=None, region=None):
    params = {'format': 'GEO_TIFF'}
    if crs_transform and dimensions:
        params['crs_transform'] = crs_transform
        params['dimensions'] = f"{dimensions[0]}x{dimensions[1]}"
        params['crs'] = crs
    elif region:
       params['region'] = region
       params['scale'] = 10
       params['crs'] = crs
       
    # We call the tiled approach directly
    download_tiled_geometry(image, name, folder, params, crs)
    
def download_multiband_image_local(image, name, folder='data/sentinel2_downloads', crs='EPSG:3857', crs_transform=None, dimensions=None, region=None):
    """
    To prevent any payload errors, band properties are already managed within `download_tiled_geometry`
    as Earth Engine seamlessly handles multiband chunks when they comfortably fit in payload limits (2048x2048).
    So we don't even need to isolate bands individually anymore! Just invoke chunking natively.
    """
    params = {'format': 'GEO_TIFF'}
    if crs_transform and dimensions:
        params['crs_transform'] = crs_transform
        params['dimensions'] = f"{dimensions[0]}x{dimensions[1]}"
        params['crs'] = crs
        
    download_tiled_geometry(image, name, folder, params, crs)

def main():
    parser = argparse.ArgumentParser(description="Download precise Sentinel-2 from Earth Engine matching WorldCover format.")
    parser.add_argument("--start_date", type=str, default="2020-06-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2020-09-01", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    authenticate_and_initialize()
    out_folder = f"data/sentinel2_downloads_{args.start_date}_{args.end_date}"
    
    # Ensure Earth Engine aligns exactly to the training data generator's grid layout
    print("Aligning map with generate_training_data...")
    bbox_32632 = get_nuremberg_bbox()
    align_params = get_align_params(bbox_32632)
    dst_crs, dst_transform, dst_width, dst_height, bounds_3857 = align_params
    
    minx, miny, maxx, maxy = bounds_3857
    
    # Re-structure the affine transformation from Rasterio to Earth Engine explicit format
    # [scale_x, shear_x, offset_x, shear_y, scale_y, offset_y]
    ee_crs_transform = [
        dst_transform.a, dst_transform.b, dst_transform.c,
        dst_transform.d, dst_transform.e, dst_transform.f
    ]
    
    ee_dimensions = [dst_width, dst_height]
    
    # We still define the AOI for filtering the Sentinel-2 image collection
    aoi = ee.Geometry.Rectangle(
        coords=[minx, miny, maxx, maxy],
        proj='EPSG:3857',
        geodesic=False
    )
    
    # Load and filter Sentinel-2 L2A collection
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(aoi)
                     .filterDate(args.start_date, args.end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))
    
    # Create cloud-free median mosaic
    image = s2_collection.median().clip(aoi)
    
    print(f"--- Generating Extracted Maps for {args.start_date} to {args.end_date} ---")
    
    # 1. S2RGBNIR Native 4-band dataset (matches original ESA S2 data layout perfectly)
    # Bands: B4 (Red), B3 (Green), B2 (Blue), B8 (NIR). Sentinel stores naturally as uint16
    s2rgbnir = image.select(['B4', 'B3', 'B2', 'B8']).toUint16()
    download_image_local(s2rgbnir, 'Nuremberg_S2RGBNIR', folder=out_folder, crs=dst_crs, crs_transform=ee_crs_transform, dimensions=ee_dimensions, region=aoi)
    
    # 2. NDVI (Vegetation Index) as visual 3-band
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndvi_vis = ndvi.visualize(min=-0.2, max=0.8, palette=['red', 'yellow', 'green'])
    download_image_local(ndvi_vis, 'Nuremberg_NDVI', folder=out_folder, crs=dst_crs, crs_transform=ee_crs_transform, dimensions=ee_dimensions, region=aoi)
    
    # 3. SWIR (Short-Wave Infrared) Visual Combination as 3-band
    swir_vis = image.visualize(bands=['B12', 'B8A', 'B4'], min=0, max=3000)
    download_image_local(swir_vis, 'Nuremberg_SWIR', folder=out_folder, crs=dst_crs, crs_transform=ee_crs_transform, dimensions=ee_dimensions, region=aoi)
    
if __name__ == '__main__':
    main()
