import os
import glob
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

def get_latest_run_paths(base_dir):
    training_data_dir = os.path.join(base_dir, "data", "training_data")
    if not os.path.exists(training_data_dir):
        raise FileNotFoundError(f"Training data directory not found at {training_data_dir}")
        
    runs = glob.glob(os.path.join(training_data_dir, "run_*"))
    if not runs:
        raise FileNotFoundError(f"No run folders found in {training_data_dir}")
        
    latest_run = max(runs, key=os.path.getctime)
    print(f"Auto-detected latest run: {latest_run}")
    
    suffix = "_denoised" if "denoised" in latest_run else "_with_noise"
    
    paths = {
        'feat_2020': os.path.join(base_dir, "Features", f"features{suffix}_2020.csv"),
        'feat_2021': os.path.join(base_dir, "Features", f"features{suffix}_2021.csv"),
        'feat_diff': os.path.join(base_dir, "Features", f"features{suffix}_diff.csv"),
        'target_2020': os.path.join(latest_run, "Composition_prediction_in_3_years", "2020", f"grid_stats{suffix}.csv"),
        'target_2021': os.path.join(latest_run, "Composition_prediction_in_3_years", "2021", f"grid_stats{suffix}.csv"),
        'target_diff': os.path.join(latest_run, "Composition_diff_in_one_year", f"grid_stats{suffix}.csv")
    }
    
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")
            
    return paths

def strip_suffix(df, suffix):
    return df.rename(columns={c: c.replace(f'_{suffix}', '') for c in df.columns if f'_{suffix}' in c})

def engineer_features(df):
    df = df.copy()
    df['lat'] = (df['min_lat'] + df['max_lat']) / 2
    df['lon'] = (df['min_lon'] + df['max_lon']) / 2
    df['NDVI_veg_ratio'] = df['NDVI_1_mean'] / (df['R_mean'] + 1e-6)
    df['SWIR_moisture_ratio'] = df['SWIR_1_mean'] / (df['G_mean'] + 1e-6)
    return df

def build_models():
    return {
        'Ridge': make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=1.0))),
        'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
        'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)),
        'MLP': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42)),
    }

def evaluate(y_true, y_pred, target_names):
    rows = []
    for i, name in enumerate(target_names):
        rows.append({
            'Target': name,
            'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
            'RMSE': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            'R2': r2_score(y_true[:, i], y_pred[:, i]),
        })
    return pd.DataFrame(rows)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    
    print("\n--- [5/5] Training ML Models (Auto-Detecting Environment) ---")
    paths = get_latest_run_paths(base_dir)
    
    feat_2020 = strip_suffix(pd.read_csv(paths['feat_2020']), '2020')
    feat_2021 = strip_suffix(pd.read_csv(paths['feat_2021']), '2021')
    feat_diff = strip_suffix(pd.read_csv(paths['feat_diff']), 'diff')
    if 'clean' in feat_diff.columns[3]: # Handle old standard if applicable
        feat_diff = strip_suffix(feat_diff, 'clean')
        
    target_2020 = pd.read_csv(paths['target_2020'])
    target_2021 = pd.read_csv(paths['target_2021'])
    target_diff = pd.read_csv(paths['target_diff'])
    
    target_cols_comp = ['cell_id', 'min_lon', 'min_lat', 'max_lon', 'max_lat', 'Built-up %', 'Permanent water bodies %', 'Other %']
    target_cols_diff = ['cell_id', 'min_lon', 'min_lat', 'max_lon', 'max_lat', 'Built-up Baseline %', 'Permanent water bodies Baseline %', 'delta Built-up %', 'delta Permanent water bodies %', 'delta Other %']
    
    df_2020 = feat_2020.merge(target_2020[target_cols_comp], on='cell_id')
    df_2021 = feat_2021.merge(target_2021[target_cols_comp], on='cell_id')
    df_diff = feat_diff.merge(target_diff[target_cols_diff], on='cell_id')
    
    df_2020 = engineer_features(df_2020)
    df_2021 = engineer_features(df_2021)
    df_diff = engineer_features(df_diff)
    
    spectral = ['R_mean', 'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std',
                'NDVI_1_mean', 'NDVI_1_std', 'NDVI_2_mean', 'NDVI_2_std',
                'NDVI_3_mean', 'NDVI_3_std', 'SWIR_1_mean', 'SWIR_1_std',
                'SWIR_2_mean', 'SWIR_2_std', 'SWIR_3_mean', 'SWIR_3_std']
                
    SPECTRAL = [c for c in spectral if c in df_2020 and df_2020[c].std() > 1e-6]
    
    FEATURES_COMP = SPECTRAL + ['lat', 'lon', 'NDVI_veg_ratio', 'SWIR_moisture_ratio']
    FEATURES_DIFF = FEATURES_COMP + ['Built-up Baseline %', 'Permanent water bodies Baseline %']
    
    TARGETS_COMP = ['Built-up %', 'Permanent water bodies %']
    TARGETS_DIFF = ['delta Built-up %', 'delta Permanent water bodies %']
    
    print("\nTraining Phase A: Composition Regression")
    df_combined = pd.concat([df_2020, df_2021], ignore_index=True)
    X_comb = df_combined[FEATURES_COMP].values
    y_comb = df_combined[TARGETS_COMP].values
    
    df_combined['block_id'] = (df_combined['row'] // 10) * 4 + (df_combined['col'] // 10)
    groups = df_combined['block_id'].values
    gkf = GroupKFold(n_splits=4)
    
    cv_preds_comp = {}
    cv_metrics_comp = []
    final_models_comp = {}
    
    for model_name in ['Ridge', 'Random Forest', 'Gradient Boosting', 'MLP']:
        print(f"  Training {model_name}...")
        model_template = build_models()[model_name]
        
        # Spatial CV
        y_pred_cv = np.zeros_like(y_comb)
        for train_idx, test_idx in gkf.split(X_comb, y_comb, groups):
            m = clone(model_template)
            m.fit(X_comb[train_idx], y_comb[train_idx])
            y_pred_cv[test_idx] = np.clip(m.predict(X_comb[test_idx]), 0, 100)
            
        cv_preds_comp[model_name] = y_pred_cv
        metrics = evaluate(y_comb, y_pred_cv, TARGETS_COMP)
        metrics['Model'] = model_name
        cv_metrics_comp.append(metrics)
        
        # Final Full Fit
        m_final = build_models()[model_name]
        m_final.fit(X_comb, y_comb)
        final_models_comp[model_name] = m_final
        
    cv_metrics_comp_df = pd.concat(cv_metrics_comp, ignore_index=True)
    
    print("\nTraining Phase B: Change (Deltas) Regression")
    X_diff = df_diff[FEATURES_DIFF].values
    y_diff = df_diff[TARGETS_DIFF].values
    
    df_diff['block_id'] = (df_diff['row'] // 10) * 4 + (df_diff['col'] // 10)
    groups_diff = df_diff['block_id'].values
    gkf_diff = GroupKFold(n_splits=4)
    
    cv_preds_diff = {}
    cv_metrics_diff = []
    final_models_diff = {}
    
    for model_name in ['Ridge', 'Random Forest', 'Gradient Boosting', 'MLP']:
        print(f"  Training {model_name}...")
        model_template = build_models()[model_name]
        
        y_pred_cv = np.zeros_like(y_diff)
        for train_idx, test_idx in gkf_diff.split(X_diff, y_diff, groups_diff):
            m = clone(model_template)
            m.fit(X_diff[train_idx], y_diff[train_idx])
            y_pred_cv[test_idx] = np.clip(m.predict(X_diff[test_idx]), -100, 100)
            
        cv_preds_diff[model_name] = y_pred_cv
        metrics = evaluate(y_diff, y_pred_cv, TARGETS_DIFF)
        metrics['Model'] = model_name
        cv_metrics_diff.append(metrics)
        
        # Final Full Fit
        m_final = build_models()[model_name]
        m_final.fit(X_diff, y_diff)
        final_models_diff[model_name] = m_final
        
    cv_metrics_diff_df = pd.concat(cv_metrics_diff, ignore_index=True)
    
    print("\nExporting Prediction Logs and Summary Tables...")
    
    # ── Composition predictions 
    comp_export = df_combined[['cell_id', 'row', 'col', 'lat', 'lon', 'Built-up %', 'Permanent water bodies %']].copy()
    comp_export['Other %'] = 100 - comp_export['Built-up %'] - comp_export['Permanent water bodies %']
    
    for model_name, preds in cv_preds_comp.items():
        tag = model_name.lower().replace(' ', '_')
        comp_export[f'{tag}_buildup'] = preds[:, 0]
        comp_export[f'{tag}_water'] = preds[:, 1]
        comp_export[f'{tag}_other'] = 100 - preds[:, 0] - preds[:, 1]
        
    # RF Uncertainty (std of trees)
    rf_comp = final_models_comp['Random Forest']
    rf_buildup = rf_comp.estimators_[0]
    rf_water = rf_comp.estimators_[1]
    
    tree_preds_buildup = np.array([tree.predict(X_comb) for tree in rf_buildup.estimators_])
    tree_preds_water = np.array([tree.predict(X_comb) for tree in rf_water.estimators_])
    
    comp_export['rf_confidence_buildup'] = tree_preds_buildup.std(axis=0)
    comp_export['rf_confidence_water'] = tree_preds_water.std(axis=0)
    
    comp_export.to_csv(os.path.join(out_dir, "composition_predictions.csv"), index=False)
    
    # ── Change predictions
    diff_export = df_diff[['cell_id', 'row', 'col', 'lat', 'lon', 'delta Built-up %', 'delta Permanent water bodies %']].copy()
    diff_export['delta Other %'] = -(diff_export['delta Built-up %'] + diff_export['delta Permanent water bodies %'])
    
    for model_name, preds in cv_preds_diff.items():
        tag = model_name.lower().replace(' ', '_')
        diff_export[f'{tag}_delta_buildup'] = preds[:, 0]
        diff_export[f'{tag}_delta_water'] = preds[:, 1]
        diff_export[f'{tag}_delta_other'] = -(preds[:, 0] + preds[:, 1])
        
    diff_export.to_csv(os.path.join(out_dir, "change_predictions.csv"), index=False)
    
    # ── Evaluation summary
    eval_summary = pd.concat([
        cv_metrics_comp_df.assign(Task='Composition'),
        cv_metrics_diff_df.assign(Task='Change')
    ], ignore_index=True)
    eval_summary.to_csv(os.path.join(out_dir, "evaluation_summary.csv"), index=False)
    
    print("Serializing weights to .joblib files...")
    for name, model in final_models_comp.items():
        path = os.path.join(out_dir, "models", f"composition_{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, path)
    for name, model in final_models_diff.items():
        path = os.path.join(out_dir, "models", f"change_{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, path)
        
    print(f"\nSUCCESS! 8 Models fully trained. Weights outputted to {out_dir}/models/")

if __name__ == '__main__':
    main()
