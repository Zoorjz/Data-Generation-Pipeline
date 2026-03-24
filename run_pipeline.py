import os
import sys
import time
import subprocess
import argparse

def check_worldcover_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming this script is run from the root of the "Data Generation Pipeline" directory
    # If the OriginalData directory is outside, we adjust the path
    esa_2020_path = os.path.join(base_dir, "OriginalData", "WORLDCOVER", "ESA_WORLDCOVER_10M_2020_V100", "MAP", "ESA_WorldCover_10m_2020_v100_N48E009_Map", "ESA_WorldCover_10m_2020_v100_N48E009_Map.tif")
    esa_2021_path = os.path.join(base_dir, "OriginalData", "WORLDCOVER", "ESA_WORLDCOVER_10M_2021_V200", "MAP", "ESA_WorldCover_10m_2021_v200_N48E009_Map", "ESA_WorldCover_10m_2021_v200_N48E009_Map.tif")

    # Add Windows long path support to bypass the 260 character limit
    if os.name == 'nt':
        esa_2020_path = f"\\\\?\\{os.path.abspath(esa_2020_path)}"
        esa_2021_path = f"\\\\?\\{os.path.abspath(esa_2021_path)}"

    missing = False
    if not os.path.exists(esa_2020_path):
        print(f"Missing WorldCover 2020 data at: {esa_2020_path}")
        missing = True
    if not os.path.exists(esa_2021_path):
        print(f"Missing WorldCover 2021 data at: {esa_2021_path}")
        missing = True

    if missing:
        print("\nWARNING: Required WorldCover data is missing!")
        print("Please refer to the README.md or ML_DATA_GUIDE.md for instructions on how to download and place these files.")
        sys.exit(1)

def run_download_with_timeout(start_date, end_date, timeout_seconds=500):
    # Check if data already exists to skip massive redundant downloads
    expected_folder = os.path.join("data", f"sentinel2_downloads_{start_date}_{end_date}")
    expected_tif = os.path.join(expected_folder, "Nuremberg_S2RGBNIR.tif")
    if os.path.exists(expected_tif):
        print(f"\n--- Data for {start_date} to {end_date} already exists in {expected_folder} ---")
        print("Skipping download.")
        return True

    print(f"\n--- Starting download for {start_date} to {end_date} ---")
    
    cmd = [
        sys.executable, 
        os.path.join("src", "download_sentinel2_ee.py"),
        "--start_date", start_date,
        "--end_date", end_date
    ]
    
    process = subprocess.Popen(cmd)
    
    last_print_time = time.time()
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed
        
        # Check if process is done
        retcode = process.poll()
        if retcode is not None:
            if retcode == 0:
                print(f"\nDownload for {start_date} to {end_date} finished successfully in {int(elapsed)} seconds.")
                return True
            else:
                print(f"\nDownload for {start_date} to {end_date} failed with return code {retcode}.")
                return False
            
        if remaining <= 0:
            print(f"\nTimeout reached ({timeout_seconds}s). Terminating process for {start_date} to {end_date}...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            print("Process terminated due to timeout.")
            return False
            
        # Print a countdown status every 30 seconds
        if time.time() - last_print_time >= 30:
            print(f"\n[Countdown] {int(remaining)} seconds remaining for {start_date} to {end_date}...")
            last_print_time = time.time()
            
        time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Run the full data generation pipeline.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--denoise", action="store_true", help="Apply denoising to 2020 data (default)")
    group.add_argument("--with_noise", action="store_true", help="Do not apply denoising (leave noise)")
    args = parser.parse_args()

    print("1. Checking WorldCover data availability...")
    check_worldcover_data()
    print("WorldCover data is present.\n")
    
    print("2. Downloading Sentinel-2 data (8 minute timeout per range)...")
    ranges = [
        ("2017-06-01", "2017-09-01"),
        ("2018-06-01", "2018-09-01"),
        ("2020-06-01", "2020-09-01"),
        ("2021-06-01", "2021-09-01")
    ]
    
    for start_date, end_date in ranges:
        success = run_download_with_timeout(start_date, end_date, timeout_seconds=500)
        
        # Automatically optionally retry one more time
        if not success:
            print(f"\nAttempt 1 failed or timed out for {start_date} to {end_date}. Trying one more time...")
            success = run_download_with_timeout(start_date, end_date, timeout_seconds=500)
            
            if not success:
                print(f"\nWARNING: Download for {start_date} to {end_date} failed/froze twice in a row!")
                print("The pipeline execution has been stopped to prevent infinite hanging.")
                print("We recommend you execute the download for this specific range manually in your terminal:")
                print(f"  python src/download_sentinel2_ee.py --start_date {start_date} --end_date {end_date}")
                print("\nOnce that succeeds, you can run this pipeline script again.")
                sys.exit(1)
        
    print("\n3. Generating Training Data...")
    os.makedirs(os.path.join("data", "training_data"), exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join("src", "generate_training_data.py"),
        "--grid_size", "1000",
        "--labels", "Built-up", "Permanent water bodies"
    ]
    
    if args.with_noise:
        cmd.append("--with_noise")
    else:
        cmd.append("--denoise")

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print("\n4. Extracting ML Features (Auto-Detecting Latest Run)...")
    extract_cmd = [
        sys.executable,
        os.path.join("src", "extract_features_cli.py")
    ]
    print(f"Running command: {' '.join(extract_cmd)}")
    subprocess.run(extract_cmd, check=True)
    
    print("\n5. Training ML Models (Ridge, RF, GBT, MLP)...")
    train_cmd = [
        sys.executable,
        os.path.join("src", "train_models_cli.py")
    ]
    print(f"Running command: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)
    
    print("\n6. Running Batch Inference Engine (Predicting Data Cubes)...")
    predict_cmd = [
        sys.executable,
        os.path.join("src", "predict_new_data.py")
    ]
    print(f"Running command: {' '.join(predict_cmd)}")
    subprocess.run(predict_cmd, check=True)
    
    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
