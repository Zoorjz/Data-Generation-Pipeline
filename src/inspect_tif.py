import argparse
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Simple tool to view individual bands/layers of a .tif file")
    parser.add_argument("filepath", type=str, help="Path to the .tif file")
    parser.add_argument("--bands", type=int, nargs="+", default=[1], help="Bands to visualize (1-indexed). Example: --bands 1 2 3")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File '{args.filepath}' does not exist.")
        return

    with rasterio.open(args.filepath) as src:
        print(f"File: {args.filepath}")
        print(f"Dimensions: {src.width}x{src.height}")
        print(f"Number of bands: {src.count}")
        print(f"Data types: {src.dtypes}")
        
        # Read the requested bands
        bands_data = []
        for b in args.bands:
            if 1 <= b <= src.count:
                bands_data.append(src.read(b))
            else:
                print(f"Error: Band {b} is out of bounds (1 to {src.count}).")
                return
                
        if len(bands_data) == 1:
            # Single channel grayscale
            img = bands_data[0]
            plt.figure(figsize=(10, 10))
            if "float" in img.dtype.name:
                plt.imshow(img, cmap='viridis')
            else:
                plt.imshow(img, cmap='gray')
            plt.colorbar(label='Pixel Value')
            plt.title(f"Band {args.bands[0]}")
            
        elif len(bands_data) == 3:
            # RGB
            img = np.dstack(bands_data)
            # Normalize to 0-1 for plotting if float or larger than uint8
            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                p2, p98 = np.percentile(img, (2, 98))
                img = np.clip((img - p2) / (p98 - p2), 0, 1)
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title(f"RGB Composite of Bands: {args.bands}")
        else:
            print(f"--bands takes either exactly 1 or 3 band numbers to plot natively on screen! You provided {len(bands_data)} bands.")
            return
            
        plt.tight_layout()
        print("Opening interactive window viewer...")
        plt.show()

if __name__ == "__main__":
    main()
