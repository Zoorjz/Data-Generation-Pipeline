"""
Generate model predictions for new feature files (2020, 2023, 2024, 2025).
Uses trained .joblib models from outputs/models/ — no retraining.

Output:
    Models_output_for_2025/
        Composition_in_3_years/   composition_predictions_{year}.csv
        Difference_in_1_year/     change_predictions_{year}.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
ESA_2020 = (PROJECT_ROOT / "data" / "ml_tables"
            / "ml_training_data_bundle_20260302_200300"
            / "data" / "training_data"
            / "Composition_prediction_in_3_years" / "2020" / "grid_stats.csv")

DOWNLOADS = Path.home() / "Downloads"
FEATURE_SOURCES = {
    2020: DOWNLOADS / "sentinel2_downloads_2020-06-01_2020-09-01" / "Features",
    2023: DOWNLOADS / "sentinel2_downloads_2023-06-01_2023-09-01" / "Features",
    2024: DOWNLOADS / "sentinel2_downloads_2024-06-01_2024-09-01" / "Features",
    2025: DOWNLOADS / "sentinel2_downloads_2025-06-01_2025-09-01" / "Features",
}

OUT_BASE = SCRIPT_DIR / "Models_output_for_2025"
OUT_COMP = OUT_BASE / "Composition_in_3_years"
OUT_DIFF = OUT_BASE / "Difference_in_1_year"

# ── Feature lists (must match training order exactly) ──
FEATURES_COMP = [
    'R_mean', 'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std',
    'NDVI_1_mean', 'NDVI_1_std', 'NDVI_2_mean', 'NDVI_2_std',
    'SWIR_1_mean', 'SWIR_1_std', 'SWIR_2_mean', 'SWIR_2_std',
    'SWIR_3_mean', 'SWIR_3_std',
    'lat', 'lon', 'NDVI_veg_ratio', 'SWIR_moisture_ratio',
]

FEATURES_DIFF = FEATURES_COMP + [
    'Built-up Baseline %', 'Permanent water bodies Baseline %',
]

MODEL_NAMES = ['ridge', 'random_forest', 'gradient_boosting', 'mlp']


def load_and_prepare(year):
    """Load feature + grid CSVs, strip year suffix, merge, engineer features."""
    src = FEATURE_SOURCES[year]
    feat = pd.read_csv(src / f"features_{year}.csv")
    grid = pd.read_csv(src / f"grid_{year}.csv")

    suffix = str(year)
    feat = feat.rename(columns={
        c: c.replace(f'_{suffix}', '') for c in feat.columns if f'_{suffix}' in c
    })

    df = feat.merge(grid[['cell_id', 'min_lon', 'min_lat', 'max_lon', 'max_lat']], on='cell_id')

    df['lat'] = (df['min_lat'] + df['max_lat']) / 2
    df['lon'] = (df['min_lon'] + df['max_lon']) / 2
    df['NDVI_veg_ratio'] = df['NDVI_1_mean'] / (df['R_mean'] + 1e-6)
    df['SWIR_moisture_ratio'] = df['SWIR_1_mean'] / (df['G_mean'] + 1e-6)

    return df


def predict_composition(df, year):
    """Run all 4 composition models on df, return results DataFrame."""
    X = df[FEATURES_COMP].values
    out = df[['cell_id', 'row', 'col', 'lat', 'lon']].copy()

    for name in MODEL_NAMES:
        model = joblib.load(MODELS_DIR / f"composition_{name}.joblib")
        preds = np.clip(model.predict(X), 0, 100)
        out[f'{name}_buildup'] = preds[:, 0]
        out[f'{name}_water'] = preds[:, 1]
        out[f'{name}_other'] = np.clip(100 - preds[:, 0] - preds[:, 1], 0, 100)

    # RF confidence via tree-level std
    rf_model = joblib.load(MODELS_DIR / "composition_random_forest.joblib")
    rf_inner = rf_model.estimators_[0]
    tree_preds_bu = np.array([t.predict(X) for t in rf_inner.estimators_])
    tree_preds_w = np.array([
        t.predict(X) for t in rf_model.estimators_[1].estimators_
    ])
    out['rf_confidence_buildup'] = tree_preds_bu.std(axis=0)
    out['rf_confidence_water'] = tree_preds_w.std(axis=0)

    return out


def predict_change(df, year, comp_preds=None):
    """Run all 4 change models on df, return results DataFrame."""

    if year == 2020:
        esa = pd.read_csv(ESA_2020)
        df = df.merge(
            esa[['cell_id', 'Built-up %', 'Permanent water bodies %']],
            on='cell_id',
        )
        df = df.rename(columns={
            'Built-up %': 'Built-up Baseline %',
            'Permanent water bodies %': 'Permanent water bodies Baseline %',
        })
    else:
        # Use MLP composition prediction as baseline (best performing model)
        df = df.copy()
        df['Built-up Baseline %'] = comp_preds['mlp_buildup'].values
        df['Permanent water bodies Baseline %'] = comp_preds['mlp_water'].values

    X = df[FEATURES_DIFF].values
    out = df[['cell_id', 'row', 'col', 'lat', 'lon']].copy()

    for name in MODEL_NAMES:
        model = joblib.load(MODELS_DIR / f"change_{name}.joblib")
        preds = model.predict(X)
        out[f'{name}_delta_buildup'] = preds[:, 0]
        out[f'{name}_delta_water'] = preds[:, 1]
        out[f'{name}_delta_other'] = -(preds[:, 0] + preds[:, 1])

    return out


def main():
    os.makedirs(OUT_COMP, exist_ok=True)
    os.makedirs(OUT_DIFF, exist_ok=True)

    for year in [2020, 2023, 2024, 2025]:
        comp_target_year = year + 3
        diff_target_year = year + 1

        print(f"\n{'='*50}")
        print(f"  Features from {year}")
        print(f"  Composition predicted for: {comp_target_year}")
        print(f"  Change predicted for: {year} -> {diff_target_year}")
        print(f"{'='*50}")

        df = load_and_prepare(year)
        print(f"  Loaded {len(df)} cells, {len(df.columns)} columns")

        # Composition predictions
        comp = predict_composition(df, year)
        comp['features_year'] = year
        comp['predicted_year'] = comp_target_year
        comp_path = OUT_COMP / f"composition_predictions_{comp_target_year}.csv"
        comp.to_csv(comp_path, index=False)
        print(f"  Saved {comp_path.name}  ({len(comp)} rows)")

        # Change predictions
        change = predict_change(df, year, comp_preds=comp)
        change['features_year'] = year
        change['predicted_change'] = f"{year}-{diff_target_year}"
        change_path = OUT_DIFF / f"change_predictions_{year}_to_{diff_target_year}.csv"
        change.to_csv(change_path, index=False)
        print(f"  Saved {change_path.name}  ({len(change)} rows)")

    print(f"\nAll outputs in: {OUT_BASE}")


if __name__ == "__main__":
    main()
