import zipfile
import os
from datetime import datetime

def create_bundle():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f'ml_training_data_bundle_{timestamp}.zip'
    
    # Core scripts explicitly requested
    files_to_zip = [
        'src/check_cell.py',
        'src/load_ml_data_example.py',
        'src/view_full_map_example.py',
        'ML_DATA_GUIDE.md',
        'requirements.txt',
        'changes.md'
    ]

    # Gather generated ML Data Files
    data_dirs = [
        os.path.join('data', 'training_data', 'Composition_prediction_in_3_years'),
        os.path.join('data', 'training_data', 'Composition_diff_in_one_year')
    ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    # Include .csv, .gpkg, and the new .tif arrays
                    if file.endswith('.csv') or file.endswith('.gpkg') or file.endswith('.tif'):
                        files_to_zip.append(os.path.join(root, file))
        else:
            print(f"Warning: Data directory '{data_dir}' not found. Did you run the generation script?")

    # Compile the Zip archive directly
    print(f"Creating ML Bundle: {zip_filename}")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            if os.path.exists(file):
                print(f' -> Packing {file}...')
                # Maintain relative file structure inside the zip
                zipf.write(file, arcname=file)
            else:
                print(f' -> [!] Skipping missing file: {file}')

    print(f'\nSuccess! Bundle created at: {os.path.abspath(zip_filename)}')
    print("You can now send this file to your team.")

if __name__ == "__main__":
    create_bundle()
