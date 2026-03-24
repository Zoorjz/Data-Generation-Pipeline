import os
import glob
import re
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def strip_suffix(df, suffix):
    return df.rename(columns={c: c.replace(f'_{suffix}', '') for c in df.columns if f'_{suffix}' in c})

def engineer_features(df):
    df = df.copy()
    if 'min_lat' in df.columns and 'max_lat' in df.columns:
        df['lat'] = (df['min_lat'] + df['max_lat']) / 2
        df['lon'] = (df['min_lon'] + df['max_lon']) / 2
    
    # Mathematical Ratios
    df['NDVI_veg_ratio'] = df['NDVI_1_mean'] / (df['R_mean'] + 1e-6)
    df['SWIR_moisture_ratio'] = df['SWIR_1_mean'] / (df['G_mean'] + 1e-6)
    return df

def align_model_inputs(df):
    # Exactly matching the standard from train_models_cli.py zero-variance drops
    SPECTRAL = [
        'R_mean', 'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std',
        'NDVI_1_mean', 'NDVI_1_std', 'NDVI_2_mean', 'NDVI_2_std',
        'SWIR_1_mean', 'SWIR_1_std', 'SWIR_2_mean', 'SWIR_2_std',
        'SWIR_3_mean', 'SWIR_3_std'
    ]
    FEATURES_COMP = SPECTRAL + ['lat', 'lon', 'NDVI_veg_ratio', 'SWIR_moisture_ratio']
    return df[FEATURES_COMP].values, FEATURES_COMP

def discover_feature_folders(data_root):
    pattern = os.path.join(data_root, "**", "Features", "features_*.csv")
    files = glob.glob(pattern, recursive=True)
    
    tasks = {}
    for f in files:
        if 'diff' in os.path.basename(f) or 'clean' in os.path.basename(f): 
            continue # Ignore diff runs or legacy
            
        match = re.search(r'(\d{4})', os.path.basename(f))
        if match:
            year = int(match.group(1))
            # Fallback to absolute latest file if multiple runs exist for identical year
            if year not in tasks or os.path.getmtime(f) > os.path.getmtime(tasks[year]['feature']):
                grid_pattern = os.path.join(os.path.dirname(f), f"grid_{year}.csv")
                # If dynamic grid exists, use it; else fallback slightly up tree
                grid_f = grid_pattern if os.path.exists(grid_pattern) else os.path.join(os.path.dirname(os.path.dirname(f)), "grid_stats.csv")
                
                tasks[year] = {
                    'feature': f,
                    'grid': grid_f
                }
    return tasks

def load_models(models_dir):
    print("Loading Joblib models...")
    algorithms = ['random_forest', 'gradient_boosting', 'mlp']
    models = {'comp': {}, 'diff': {}}
    
    for alg in algorithms:
        comp_path = os.path.join(models_dir, f"composition_{alg}.joblib")
        diff_path = os.path.join(models_dir, f"change_{alg}.joblib")
        
        if os.path.exists(comp_path): models['comp'][alg] = joblib.load(comp_path)
        if os.path.exists(diff_path): models['diff'][alg] = joblib.load(diff_path)
            
    return models

def construct_composition_df(df, models_comp, X_comp, year):
    out = df[['cell_id', 'row', 'col', 'lat', 'lon']].copy()
    
    for alg, model in models_comp.items():
        preds = np.clip(model.predict(X_comp), 0, 100)
        out[f'{alg}_buildup'] = preds[:, 0]
        out[f'{alg}_water'] = preds[:, 1]
        out[f'{alg}_other'] = 100 - preds[:, 0] - preds[:, 1]
    
    out['features_year'] = year
    out['predicted_year'] = year + 3
    return out

def construct_diff_df(df, models_diff, X_comp, comp_predictions, year):
    out = df[['cell_id', 'row', 'col', 'lat', 'lon']].copy()
    
    for alg, model in models_diff.items():
        # Diff inputs strictly require ["Built-up Baseline %", "Permanent water bodies Baseline %"] at runtime
        # We synthesize baseline empirically from the exact counterpart structural composition prediction!
        baseline_buildup = comp_predictions[f'{alg}_buildup'].values.reshape(-1, 1)
        baseline_water = comp_predictions[f'{alg}_water'].values.reshape(-1, 1)
        
        X_diff = np.hstack([X_comp, baseline_buildup, baseline_water])
        
        preds = np.clip(model.predict(X_diff), -100, 100)
        out[f'{alg}_delta_buildup'] = preds[:, 0]
        out[f'{alg}_delta_water'] = preds[:, 1]
        out[f'{alg}_delta_other'] = -(preds[:, 0] + preds[:, 1])
        
    out['features_year'] = year
    out['predicted_change'] = "1 year"
    return out

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_gen_dir = base_dir
    
    tasks = discover_feature_folders(os.path.join(data_gen_dir, "data"))
    if not tasks:
        print("No feature outputs available to predict on!")
        return
        
    print(f"Discovered mapped features for inference years: {list(tasks.keys())}")
    models = load_models(os.path.join(data_gen_dir, "outputs", "models"))
    
    predictions_root = os.path.join(base_dir, "predictions")
    os.makedirs(predictions_root, exist_ok=True)
    
    for year, files in tasks.items():
        print(f"\n--- Predicting Datacube for Universe Year {year} ---")
        
        # Unify geometry
        feats = strip_suffix(pd.read_csv(files['feature']), str(year))
        feats = strip_suffix(feats, 'clean') # Just in case
        
        grids = []
        if os.path.exists(files['grid']):
            grids = pd.read_csv(files['grid'])
        else:
            # Fallback search if grid.csv isn't exactly bound
            print(f"Warning: Grid mapping not found dynamically at {files['grid']}, bypassing spatial map.")
            continue
            
        merged = feats.merge(grids[['cell_id', 'min_lat', 'max_lat', 'min_lon', 'max_lon']], on='cell_id', how='left')
        df = engineer_features(merged)
        
        X_comp, _ = align_model_inputs(df)
        
        # Pipeline 1: Native Commposition
        print(f"Inferencing Composition (Year + 3 => {year + 3})...")
        comp_df = construct_composition_df(df, models['comp'], X_comp, year)
        
        # Pipeline 2: Temporal Delta
        print(f"Inferencing Change (Year + 1 => {year + 1})...")
        diff_df = construct_diff_df(df, models['diff'], X_comp, comp_df, year)
        
        # Exact Dashboard File Formatting
        year_out_dir = os.path.join(predictions_root, str(year))
        os.makedirs(year_out_dir, exist_ok=True)
        
        comp_file = os.path.join(year_out_dir, f"3year_composition_{year}_to_{year+3}.csv")
        diff_file = os.path.join(year_out_dir, f"1year_delta_{year}_to_{year+1}.csv")
        
        comp_df.to_csv(comp_file, index=False)
        diff_df.to_csv(diff_file, index=False)
        
        print(f"SUCCESS: Ejected prediction bundle into root `{year_out_dir}`!")
        
if __name__ == "__main__":
    main()