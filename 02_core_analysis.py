#!/usr/bin/env python
"""
02_core_analysis.py — Spearman Correlation, Random Forest, and Temporal Analysis
=================================================================================

Interpretability analysis of AlphaEarth 64-dimensional satellite embeddings
against 26 environmental variables across CONUS (2017-2023, ~12.1M samples).

Analysis components:
  1. Spearman rank correlation (n=1,000,000): 64 x 26 correlation matrix
  2. Random Forest permutation importance (n=700,000): nonlinear relationships
  3. Spatial block cross-validation (2 deg blocks): generalization assessment
  4. Temporal stability analysis (n=300,000/year x 7 years)
  5. Dimension dictionary: per-dimension semantic labels
  6. CONUS map rasterization for spatial visualization

Sample sizes are chosen to balance statistical power with memory constraints
(32 GB RAM). Per-year loading avoids materializing the full 12M-row dataset.

Outputs (saved to results/ directory):
  spearman_matrix.csv, pval_matrix.csv, rf_importance_matrix.csv,
  rf_r2_scores.csv, rf_details.csv, cv_comparison.csv,
  dimension_dictionary.csv, temporal_stability.csv, temporal_series.csv,
  temporal_spearman_{year}.csv, clustering_orders.npz, conus_grids.npz,
  analysis_metadata.json

Usage:
  python 02_core_analysis.py                    # Full run (~30-45 min)
  python 02_core_analysis.py --skip-rf          # Skip RF, reuse cached results
  python 02_core_analysis.py --years 2021 2022  # Specific years only
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.inspection import permutation_importance
from tqdm import tqdm
import os
import gc
import sys
import json
import time
import warnings
import pyarrow.parquet as pq
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    DATA_DIR = '../data/unified_conus'
    RESULTS_DIR = 'results'
    YEARS = list(range(2017, 2024))

    AE_COLS = [f'A{i:02d}' for i in range(64)]
    META_COLS = ['longitude', 'latitude', 'year', 'point_id']

    # 26 continuous environmental variables
    ENV_VARS_CANDIDATES = [
        'elevation', 'slope', 'aspect',
        'soil_clay_pct', 'soil_organic_carbon', 'soil_ph', 'soil_water_capacity',
        'flow_acc_log', 'tree_cover_2000',
        'impervious_pct',
        'ndvi_mean', 'ndvi_max', 'evi_mean', 'lai_mean',
        'lst_day_c', 'lst_night_c', 'albedo',
        'precip_annual_mm', 'precip_max_month', 'temp_mean_c', 'temp_range_c',
        'soil_moisture', 'runoff_annual_mm', 'et_annual_mm',
        'nightlights', 'pop_density',
    ]

    ENV_LABELS = {
        'elevation': 'Elevation', 'slope': 'Slope', 'aspect': 'Aspect',
        'soil_clay_pct': 'Soil Clay %', 'soil_organic_carbon': 'Soil Organic C',
        'soil_ph': 'Soil pH', 'soil_water_capacity': 'Soil Water Cap.',
        'flow_acc_log': 'Flow Accum. (log)', 'tree_cover_2000': 'Tree Cover 2000',
        'impervious_pct': 'Impervious %',
        'ndvi_mean': 'NDVI (mean)', 'ndvi_max': 'NDVI (max)',
        'evi_mean': 'EVI (mean)', 'lai_mean': 'LAI (mean)',
        'lst_day_c': 'LST Day (C)', 'lst_night_c': 'LST Night (C)',
        'albedo': 'Albedo',
        'precip_annual_mm': 'Precip. (mm/yr)', 'precip_max_month': 'Max Monthly Precip.',
        'temp_mean_c': 'Temp. Mean (C)', 'temp_range_c': 'Dewpoint (C)',
        'soil_moisture': 'Soil Moisture', 'runoff_annual_mm': 'Runoff (mm/yr)',
        'et_annual_mm': 'ET (mm/yr)',
        'nightlights': 'Nightlights', 'pop_density': 'Pop. Density',
    }

    ENV_CATEGORY = {
        'elevation': 'Terrain', 'slope': 'Terrain', 'aspect': 'Terrain',
        'soil_clay_pct': 'Soil', 'soil_organic_carbon': 'Soil',
        'soil_ph': 'Soil', 'soil_water_capacity': 'Soil',
        'flow_acc_log': 'Hydrology', 'tree_cover_2000': 'Vegetation',
        'impervious_pct': 'Urban',
        'ndvi_mean': 'Vegetation', 'ndvi_max': 'Vegetation',
        'evi_mean': 'Vegetation', 'lai_mean': 'Vegetation',
        'lst_day_c': 'Temperature', 'lst_night_c': 'Temperature',
        'albedo': 'Radiation',
        'precip_annual_mm': 'Climate', 'precip_max_month': 'Climate',
        'temp_mean_c': 'Temperature', 'temp_range_c': 'Temperature',
        'soil_moisture': 'Hydrology', 'runoff_annual_mm': 'Hydrology',
        'et_annual_mm': 'Hydrology',
        'nightlights': 'Urban', 'pop_density': 'Urban',
    }

    CATEGORY_COLORS = {
        'Terrain':     '#8C564B',
        'Soil':        '#C49C6B',
        'Vegetation':  '#2CA02C',
        'Temperature': '#D62728',
        'Climate':     '#1F77B4',
        'Hydrology':   '#17BECF',
        'Urban':       '#7F7F7F',
        'Radiation':   '#BCBD22',
    }

    # Computation parameters
    SPEARMAN_SAMPLE = 1_000_000
    RF_SAMPLE = 700_000
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = 12
    RF_MIN_SAMPLES_LEAF = 50
    RF_MAX_FEATURES = 'sqrt'
    RF_N_JOBS = 12
    PERM_N_REPEATS = 10
    PERM_SAMPLE = 150_000
    N_FOLDS = 5
    SPATIAL_BLOCK_DEG = 2.0
    TOP_N_CV = 10
    MIN_VALID = 1000
    TEMPORAL_SAMPLE_PER_YEAR = 300_000

    MAP_YEAR = 2021
    GRID_SPACING = 0.025
    LON_RANGE = (-125.0, -66.5)
    LAT_RANGE = (24.5, 49.5)

    SEED = 42


def print_header(msg):
    print(f"\n{'='*70}")
    print(msg)
    print('='*70)


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def discover_env_vars(data_dir, years):
    """Read one parquet file schema to discover available environmental variables."""
    for yr in years:
        fp = f'{data_dir}/conus_{yr}_unified.parquet'
        if os.path.exists(fp):
            all_cols = pq.read_schema(fp).names
            available = [v for v in Config.ENV_VARS_CANDIDATES if v in all_cols]
            print(f"  Discovered {len(available)}/{len(Config.ENV_VARS_CANDIDATES)} "
                  f"env vars from {yr}")
            return available, all_cols
    raise FileNotFoundError("No data files found in " + data_dir)


def load_subsample(n_total, years, data_dir, columns, seed=42):
    """
    Memory-efficient data loading with per-year subsampling.
    Only needed columns are loaded; each year is subsampled independently
    to avoid materializing the full dataset.
    """
    print_header(f"LOADING DATA — target n={n_total:,}")

    rng = np.random.RandomState(seed)
    frames = []
    n_per_year = n_total // len(years)

    for yr in years:
        fp = f'{data_dir}/conus_{yr}_unified.parquet'
        if not os.path.exists(fp):
            print(f"  WARNING: {fp} not found, skipping")
            continue

        try:
            all_cols = pq.read_schema(fp).names
        except:
            df_peek = pd.read_parquet(fp, columns=['longitude'])
            all_cols = pd.read_parquet(fp).columns.tolist()

        cols_to_load = [c for c in columns if c in all_cols]

        for coord in ['longitude', 'latitude']:
            if coord in all_cols and coord not in cols_to_load:
                cols_to_load.append(coord)

        if not cols_to_load:
            print(f"  {yr}: no matching columns, skipping")
            continue

        df_yr = pd.read_parquet(fp, columns=cols_to_load)
        print(f"  {yr}: loaded {len(df_yr):,} rows x {len(cols_to_load)} cols")

        # Require valid AE embeddings
        ae_in_cols = [c for c in Config.AE_COLS if c in df_yr.columns]
        if ae_in_cols:
            mask = df_yr[ae_in_cols].notna().all(axis=1)
            df_yr = df_yr[mask].reset_index(drop=True)

        n_take = min(n_per_year, len(df_yr))
        if n_take > 0 and n_take < len(df_yr):
            idx = rng.choice(len(df_yr), size=n_take, replace=False)
            df_yr = df_yr.iloc[idx].copy()

        # float32 for memory savings
        for c in df_yr.columns:
            if df_yr[c].dtype == 'float64':
                df_yr[c] = df_yr[c].astype('float32')

        print(f"       -> subsampled to {len(df_yr):,}")
        frames.append(df_yr)

        gc.collect()

    if not frames:
        raise ValueError("No data files found or no valid data!")

    df = pd.concat(frames, ignore_index=True)
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"  Final: {len(df):,} samples ({mem_mb:.0f} MB)")

    return df


def validate_env_vars(df, env_candidates):
    """Keep only environmental variables with >5% non-NaN coverage."""
    env_cols = []
    for v in env_candidates:
        if v not in df.columns:
            continue
        coverage = df[v].notna().mean()
        if coverage < 0.05:
            print(f"  EXCLUDED (coverage {coverage:.1%}): {v}")
            continue
        env_cols.append(v)
    print(f"  Valid env vars: {len(env_cols)}/{len(env_candidates)}")
    return env_cols


# =============================================================================
# 2. SPEARMAN CORRELATION
# =============================================================================

def compute_spearman(df, ae_cols, env_cols):
    """Compute Spearman rho for all 64 x N_env pairs."""
    print_header(f"SPEARMAN CORRELATION — {len(ae_cols)} x {len(env_cols)} "
                 f"(n = {len(df):,})")

    corr_mat = np.full((len(ae_cols), len(env_cols)), np.nan, dtype=np.float64)
    pval_mat = np.full_like(corr_mat, np.nan)

    for j, ev in enumerate(tqdm(env_cols, desc="Spearman")):
        mask_e = df[ev].notna().values
        if mask_e.sum() < Config.MIN_VALID:
            continue
        ev_vals = df[ev].values

        for i, ad in enumerate(ae_cols):
            mask_a = df[ad].notna().values
            both = mask_e & mask_a
            if both.sum() < Config.MIN_VALID:
                continue
            rho, pval = spearmanr(df[ad].values[both], ev_vals[both])
            corr_mat[i, j] = rho
            pval_mat[i, j] = pval

    corr_df = pd.DataFrame(corr_mat, index=ae_cols, columns=env_cols)
    pval_df = pd.DataFrame(pval_mat, index=ae_cols, columns=env_cols)

    n_computed = np.isfinite(corr_mat).sum()
    n_05 = (np.abs(corr_mat[np.isfinite(corr_mat)]) > 0.5).sum()
    n_07 = (np.abs(corr_mat[np.isfinite(corr_mat)]) > 0.7).sum()
    print(f"\n  Computed: {n_computed:,} pairs")
    print(f"  |rho| > 0.5: {n_05},  |rho| > 0.7: {n_07}")

    # Report top-5 pairs
    flat = []
    for i, d in enumerate(ae_cols):
        for j, v in enumerate(env_cols):
            if np.isfinite(corr_mat[i, j]):
                flat.append((d, v, corr_mat[i, j]))
    flat.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\n  Top 5 correlations:")
    for d, v, rho in flat[:5]:
        print(f"    {d} x {Config.ENV_LABELS.get(v, v)}: rho = {rho:+.4f}")

    return corr_df, pval_df


# =============================================================================
# 3. RANDOM FOREST PERMUTATION IMPORTANCE
# =============================================================================

def compute_rf_importance(df, ae_cols, env_cols, rng):
    """
    Train one RF per environmental variable, extract permutation importance.
    Each model uses 64 AE dimensions as features.
    """
    print_header(f"RANDOM FOREST PERMUTATION IMPORTANCE (n = {len(df):,})")
    print(f"  n_estimators={Config.RF_N_ESTIMATORS}, max_depth={Config.RF_MAX_DEPTH}, "
          f"n_jobs={Config.RF_N_JOBS}")
    print(f"  Permutation importance: {Config.PERM_SAMPLE:,} samples, "
          f"{Config.PERM_N_REPEATS} repeats")

    imp_matrix = pd.DataFrame(np.nan, index=ae_cols, columns=env_cols)
    rf_r2 = {}
    rf_details = []

    for ev in tqdm(env_cols, desc="RF models"):
        cols = ae_cols + [ev]
        dv = df[cols].dropna()
        if len(dv) < 2000:
            rf_r2[ev] = np.nan
            continue

        X = dv[ae_cols].values.astype('float32')
        y = dv[ev].values.astype('float32')
        n_used = len(X)

        rf = RandomForestRegressor(
            n_estimators=Config.RF_N_ESTIMATORS,
            max_depth=Config.RF_MAX_DEPTH,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            max_features=Config.RF_MAX_FEATURES,
            n_jobs=Config.RF_N_JOBS,
            random_state=Config.SEED,
        )

        # Random k-fold CV
        cv = cross_val_score(rf, X, y, cv=Config.N_FOLDS,
                             scoring='r2', n_jobs=Config.RF_N_JOBS)
        rf_r2[ev] = cv.mean()

        # Fit on full data, then compute permutation importance on subset
        rf.fit(X, y)

        n_perm = min(Config.PERM_SAMPLE, len(X))
        perm_idx = rng.choice(len(X), size=n_perm, replace=False)
        perm = permutation_importance(
            rf, X[perm_idx], y[perm_idx],
            n_repeats=Config.PERM_N_REPEATS,
            random_state=Config.SEED,
            n_jobs=Config.RF_N_JOBS,
        )

        for i, ad in enumerate(ae_cols):
            imp_matrix.loc[ad, ev] = perm.importances_mean[i]

        top3 = np.argsort(perm.importances_mean)[::-1][:3]
        rf_details.append({
            'variable': ev, 'r2_cv': cv.mean(), 'r2_cv_std': cv.std(),
            'n_samples': n_used,
            'top1': ae_cols[top3[0]], 'top2': ae_cols[top3[1]], 'top3': ae_cols[top3[2]],
        })
        print(f"  {ev:25s}: R2={cv.mean():.3f}+/-{cv.std():.3f}  (n={n_used:,})  "
              f"Top: {ae_cols[top3[0]]}, {ae_cols[top3[1]]}, {ae_cols[top3[2]]}")

        del rf, X, y, dv, perm
        gc.collect()

    return imp_matrix, rf_r2, pd.DataFrame(rf_details)


# =============================================================================
# 4. SPATIAL BLOCK CROSS-VALIDATION
# =============================================================================

def compute_spatial_cv(df, ae_cols, env_cols, corr_df, rf_r2_random):
    """
    Spatial block CV using 2-degree blocks (~222 km) to assess
    generalization beyond spatial autocorrelation.
    """
    print_header("SPATIAL BLOCK CROSS-VALIDATION")
    bs = Config.SPATIAL_BLOCK_DEG
    print(f"  Block size: {bs} deg (~{bs*111:.0f} km), n={len(df):,}")

    df = df.copy()
    df['_blk'] = (
        np.floor(df['longitude'] / bs).astype(int).astype(str) + '_' +
        np.floor(df['latitude'] / bs).astype(int).astype(str)
    )
    n_blocks = df['_blk'].nunique()
    print(f"  Spatial blocks: {n_blocks}")

    # Select top variables by maximum absolute Spearman correlation
    top_vars = (
        corr_df.abs().max(axis=0)
        .sort_values(ascending=False)
        .head(Config.TOP_N_CV)
        .index.tolist()
    )
    print(f"  Variables ({len(top_vars)}): {top_vars}")

    rf_r2_spatial = {}

    for ev in tqdm(top_vars, desc="Spatial CV"):
        cols_needed = ae_cols + [ev, '_blk']
        dv = df[cols_needed].dropna(subset=ae_cols + [ev])
        if len(dv) < 2000:
            rf_r2_spatial[ev] = np.nan
            continue

        X = dv[ae_cols].values.astype('float32')
        y = dv[ev].values.astype('float32')
        groups = dv['_blk'].values

        rf = RandomForestRegressor(
            n_estimators=Config.RF_N_ESTIMATORS,
            max_depth=Config.RF_MAX_DEPTH,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            max_features=Config.RF_MAX_FEATURES,
            n_jobs=Config.RF_N_JOBS,
            random_state=Config.SEED,
        )
        scores = cross_val_score(
            rf, X, y,
            cv=GroupKFold(n_splits=Config.N_FOLDS),
            groups=groups,
            scoring='r2',
            n_jobs=Config.RF_N_JOBS,
        )
        rf_r2_spatial[ev] = scores.mean()

        rand_r2 = rf_r2_random.get(ev, np.nan)
        gap = rand_r2 - scores.mean()
        print(f"  {ev:25s}: Random={rand_r2:.3f}  Spatial={scores.mean():.3f}  "
              f"Gap={gap:+.3f}  (n={len(dv):,})")

        del rf, X, y, dv
        gc.collect()

    df.drop(columns='_blk', inplace=True, errors='ignore')

    cv_rows = []
    for v in top_vars:
        r = rf_r2_random.get(v, np.nan)
        s = rf_r2_spatial.get(v, np.nan)
        cv_rows.append({'variable': v, 'random_cv': r, 'spatial_cv': s,
                        'gap': r - s if np.isfinite(r) and np.isfinite(s) else np.nan})

    return rf_r2_spatial, top_vars, pd.DataFrame(cv_rows)


# =============================================================================
# 5. TEMPORAL ANALYSIS
# =============================================================================

def compute_temporal_spearman(ae_cols, env_cols, data_dir, years):
    """
    Compute Spearman rho independently per year to assess temporal stability.
    Each year uses an independent subsample; data is loaded one year at a time.
    """
    print_header(f"TEMPORAL ANALYSIS — {Config.TEMPORAL_SAMPLE_PER_YEAR:,}/year x {len(years)} years")

    rng = np.random.RandomState(Config.SEED + 100)
    yearly_corr = {}

    for yr in years:
        fp = f'{data_dir}/conus_{yr}_unified.parquet'
        if not os.path.exists(fp):
            print(f"  {yr}: file not found, skipping")
            continue

        needed = ['longitude', 'latitude'] + ae_cols + env_cols
        all_cols = pq.read_schema(fp).names
        cols_to_load = [c for c in needed if c in all_cols]

        df_yr = pd.read_parquet(fp, columns=cols_to_load)
        mask = df_yr[ae_cols].notna().all(axis=1)
        df_yr = df_yr[mask]

        n_sub = min(Config.TEMPORAL_SAMPLE_PER_YEAR, len(df_yr))
        idx = rng.choice(len(df_yr), size=n_sub, replace=False)
        df_yr = df_yr.iloc[idx]

        corr_yr = np.full((len(ae_cols), len(env_cols)), np.nan)
        for j, ev in enumerate(env_cols):
            if ev not in df_yr.columns:
                continue
            mask_e = df_yr[ev].notna().values
            if mask_e.sum() < 500:
                continue
            ev_vals = df_yr[ev].values
            for i, ad in enumerate(ae_cols):
                mask_a = df_yr[ad].notna().values
                both = mask_e & mask_a
                if both.sum() < 500:
                    continue
                rho, _ = spearmanr(df_yr[ad].values[both], ev_vals[both])
                corr_yr[i, j] = rho

        yearly_corr[yr] = pd.DataFrame(corr_yr, index=ae_cols, columns=env_cols)
        n_valid = np.isfinite(corr_yr).sum()
        print(f"  {yr}: {n_sub:,} samples -> {n_valid:,} correlations")

        del df_yr
        gc.collect()

    # Temporal stability: pairwise Pearson r between yearly correlation profiles
    print("\n  Computing temporal stability...")
    stability = []
    year_list = sorted(yearly_corr.keys())

    for d in ae_cols:
        profiles = []
        for yr in year_list:
            if yr in yearly_corr:
                profile = yearly_corr[yr].loc[d].values
                if np.isfinite(profile).sum() > 5:
                    profiles.append(profile)

        if len(profiles) < 2:
            stability.append({'dimension': d, 'mean_profile_corr': np.nan,
                              'std_profile_corr': np.nan, 'n_years': len(profiles)})
            continue

        pairwise = []
        for i in range(len(profiles)):
            for j_p in range(i+1, len(profiles)):
                mask = np.isfinite(profiles[i]) & np.isfinite(profiles[j_p])
                if mask.sum() > 3:
                    r = np.corrcoef(profiles[i][mask], profiles[j_p][mask])[0, 1]
                    pairwise.append(r)

        stability.append({
            'dimension': d,
            'mean_profile_corr': np.mean(pairwise) if pairwise else np.nan,
            'std_profile_corr': np.std(pairwise) if pairwise else np.nan,
            'min_profile_corr': np.min(pairwise) if pairwise else np.nan,
            'n_years': len(profiles),
        })

    stability_df = pd.DataFrame(stability)

    # Per-pair temporal series
    temporal_series = []
    for d in ae_cols:
        for v in env_cols:
            row = {'dimension': d, 'variable': v}
            for yr in year_list:
                if yr in yearly_corr:
                    row[f'rho_{yr}'] = yearly_corr[yr].loc[d, v]
            temporal_series.append(row)

    temporal_series_df = pd.DataFrame(temporal_series)

    mean_stab = stability_df['mean_profile_corr'].mean()
    min_stab = stability_df['mean_profile_corr'].min()
    print(f"\n  Temporal stability: mean={mean_stab:.4f}, min={min_stab:.4f}")

    return yearly_corr, stability_df, temporal_series_df


# =============================================================================
# 6. DIMENSION DICTIONARY
# =============================================================================

def build_dimension_dictionary(ae_cols, corr_df, imp_matrix):
    """
    Assign semantic labels to each embedding dimension based on convergence
    of Spearman correlation and RF permutation importance.
    """
    print_header("DIMENSION DICTIONARY — ALL 64 DIMENSIONS")

    dim_dict = []
    for d in ae_cols:
        sp_row = corr_df.loc[d].dropna()
        imp_row = imp_matrix.loc[d].dropna().astype(float)

        entry = {'dimension': d}

        if len(sp_row) > 0:
            sp_sorted = sp_row.abs().sort_values(ascending=False)
            sp_top = sp_sorted.index[0]
            entry['sp_primary'] = sp_top
            entry['sp_rho'] = sp_row[sp_top]
            entry['sp_abs_max'] = abs(sp_row[sp_top])
            entry['sp_category'] = Config.ENV_CATEGORY.get(sp_top, '?')
            for rank in range(min(3, len(sp_sorted))):
                var = sp_sorted.index[rank]
                entry[f'sp_top{rank+1}_var'] = var
                entry[f'sp_top{rank+1}_rho'] = sp_row[var]
        else:
            entry.update({'sp_primary': 'N/A', 'sp_rho': np.nan,
                          'sp_abs_max': 0, 'sp_category': '?'})

        if len(imp_row) > 0:
            rf_top = imp_row.idxmax()
            entry['rf_primary'] = rf_top
            entry['rf_imp'] = imp_row[rf_top]
            entry['rf_category'] = Config.ENV_CATEGORY.get(rf_top, '?')
        else:
            entry.update({'rf_primary': 'N/A', 'rf_imp': np.nan, 'rf_category': '?'})

        entry['agree'] = entry.get('sp_primary') == entry.get('rf_primary')
        dim_dict.append(entry)

    dd = pd.DataFrame(dim_dict).sort_values('sp_abs_max', ascending=False)
    n_agree = dd['agree'].sum()
    print(f"  Linear-RF agreement: {n_agree}/64 ({100*n_agree/64:.0f}%)")

    for _, r in dd.head(20).iterrows():
        tag = 'Y' if r['agree'] else 'N'
        sp_label = Config.ENV_LABELS.get(r['sp_primary'], r['sp_primary'])
        rf_label = Config.ENV_LABELS.get(r['rf_primary'], r['rf_primary'])
        print(f"    {r['dimension']:>3s}: Spearman->{sp_label:<18s} (rho={r['sp_rho']:+.3f})  "
              f"RF->{rf_label:<18s} [{tag}]")

    return dd


# =============================================================================
# 7. CONUS MAP DATA
# =============================================================================

def prepare_conus_map_data(ae_cols, dd, data_dir, map_year):
    """
    Rasterize top embedding dimensions and their associated environmental
    variables at full CONUS resolution for spatial visualization.
    """
    print_header(f"CONUS MAP DATA (year={map_year}, full resolution)")

    fp = f'{data_dir}/conus_{map_year}_unified.parquet'
    if not os.path.exists(fp):
        for yr in Config.YEARS:
            alt = f'{data_dir}/conus_{yr}_unified.parquet'
            if os.path.exists(alt):
                fp = alt
                print(f"  {map_year} not found, using {yr}")
                break

    top_dims = dd.head(6)['dimension'].tolist()
    dim_to_var = dict(zip(dd['dimension'], dd['sp_primary']))
    top_vars = list(set(dim_to_var[d] for d in top_dims if dim_to_var[d] != 'N/A'))

    cols_needed = ['longitude', 'latitude'] + top_dims + top_vars
    all_cols = pq.read_schema(fp).names
    cols_to_load = [c for c in cols_needed if c in all_cols]

    df_map = pd.read_parquet(fp, columns=cols_to_load)
    df_map = df_map.dropna(subset=top_dims[:1]).reset_index(drop=True)
    print(f"  Loaded {len(df_map):,} points ({len(cols_to_load)} columns)")

    lo, hi = Config.LON_RANGE
    la, ha = Config.LAT_RANGE
    sp = Config.GRID_SPACING
    nc = int(round((hi - lo) / sp))
    nr = int(round((ha - la) / sp))
    extent = [lo, hi, la, ha]

    def rasterize(df, col):
        grid = np.full((nr, nc), np.nan, dtype=np.float32)
        ci = np.round((df['longitude'].values - lo) / sp).astype(int)
        ri = np.round((df['latitude'].values - la) / sp).astype(int)
        ok = (ci >= 0) & (ci < nc) & (ri >= 0) & (ri < nr) & df[col].notna().values
        grid[ri[ok], ci[ok]] = df[col].values[ok]
        return grid

    grids = {}
    for d in top_dims:
        if d in df_map.columns:
            grids[f'dim_{d}'] = rasterize(df_map, d)
    for v in top_vars:
        if v in df_map.columns:
            grids[f'var_{v}'] = rasterize(df_map, v)

    print(f"  Rasterized {len(grids)} grids, shape=({nr}, {nc})")
    del df_map
    gc.collect()

    return grids, top_dims, dim_to_var, extent


# =============================================================================
# 8. HIERARCHICAL CLUSTERING
# =============================================================================

def compute_clustering(corr_df):
    """Hierarchical clustering of the Spearman correlation matrix."""
    corr_vals = corr_df.values.copy()
    corr_vals[np.isnan(corr_vals)] = 0
    col_link = linkage(pdist(corr_vals.T, 'correlation'), method='average')
    col_order = leaves_list(col_link)
    row_link = linkage(pdist(corr_vals, 'correlation'), method='average')
    row_order = leaves_list(row_link)
    return row_order, col_order, row_link, col_link


# =============================================================================
# 9. SAVE RESULTS
# =============================================================================

def save_results(results_dir, **kwargs):
    """Save analysis outputs in appropriate formats."""
    os.makedirs(results_dir, exist_ok=True)
    for name, obj in kwargs.items():
        if isinstance(obj, pd.DataFrame):
            path = f'{results_dir}/{name}.csv'
            obj.to_csv(path, index=True if name.endswith('_matrix') else False)
            print(f"  Saved: {path} ({len(obj)} rows)")
        elif isinstance(obj, dict) and all(isinstance(v, np.ndarray) for v in obj.values()):
            path = f'{results_dir}/{name}.npz'
            np.savez_compressed(path, **obj)
            print(f"  Saved: {path}")
        elif isinstance(obj, dict):
            path = f'{results_dir}/{name}.json'
            serializable = {}
            for k, v in obj.items():
                if isinstance(v, (np.integer, np.int64)):
                    serializable[str(k)] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)):
                    serializable[str(k)] = float(v)
                elif isinstance(v, np.ndarray):
                    serializable[str(k)] = v.tolist()
                else:
                    serializable[str(k)] = v
            with open(path, 'w') as f:
                json.dump(serializable, f, indent=2)
            print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()
    skip_rf = '--skip-rf' in sys.argv

    year_args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if year_args:
        Config.YEARS = year_args
        print(f"Running for years: {Config.YEARS}")

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    print_header("CORE ANALYSIS — AlphaEarth Dimension Interpretability")
    print(f"  Dataset: {len(Config.YEARS)} years x ~1.73M samples/year = ~12.1M total")
    print(f"  Spearman sample:  {Config.SPEARMAN_SAMPLE:>10,}")
    print(f"  RF sample:        {Config.RF_SAMPLE:>10,}")
    print(f"  Temporal sample:  {Config.TEMPORAL_SAMPLE_PER_YEAR:>10,} per year")
    print(f"  Data: {Config.DATA_DIR}")
    print(f"  Output: {Config.RESULTS_DIR}")

    env_candidates, all_cols = discover_env_vars(Config.DATA_DIR, Config.YEARS)

    needed_cols = Config.AE_COLS + env_candidates + ['longitude', 'latitude']
    df_sp = load_subsample(Config.SPEARMAN_SAMPLE, Config.YEARS,
                           Config.DATA_DIR, needed_cols, seed=Config.SEED)
    env_cols = validate_env_vars(df_sp, env_candidates)

    corr_df, pval_df = compute_spearman(df_sp, Config.AE_COLS, env_cols)
    del df_sp
    gc.collect()

    print("  Saving Spearman results...")
    corr_df.to_csv(f"{Config.RESULTS_DIR}/spearman_matrix.csv")
    pval_df.to_csv(f"{Config.RESULTS_DIR}/pval_matrix.csv")

    if skip_rf and os.path.exists(f'{Config.RESULTS_DIR}/rf_importance_matrix.csv'):
        print("\n  Loading cached RF results...")
        imp_matrix = pd.read_csv(f'{Config.RESULTS_DIR}/rf_importance_matrix.csv',
                                 index_col=0)
        rf_r2 = {}
        if os.path.exists(f'{Config.RESULTS_DIR}/rf_r2_scores.csv'):
            rf_r2_df = pd.read_csv(f'{Config.RESULTS_DIR}/rf_r2_scores.csv')
            rf_r2 = dict(zip(rf_r2_df['variable'], rf_r2_df['r2_cv']))
        rf_details_df = pd.DataFrame()

        if os.path.exists(f'{Config.RESULTS_DIR}/cv_comparison.csv'):
            cv_df = pd.read_csv(f'{Config.RESULTS_DIR}/cv_comparison.csv')
            rf_r2_spatial = dict(zip(cv_df['variable'], cv_df['spatial_cv']))
            top_vars_cv = cv_df['variable'].tolist()
        else:
            df_cv = load_subsample(Config.RF_SAMPLE, Config.YEARS,
                                   Config.DATA_DIR, needed_cols, seed=Config.SEED + 1)
            rf_r2_spatial, top_vars_cv, cv_df = compute_spatial_cv(
                df_cv, Config.AE_COLS, env_cols, corr_df, rf_r2)
            del df_cv; gc.collect()
    else:
        df_rf = load_subsample(Config.RF_SAMPLE, Config.YEARS,
                               Config.DATA_DIR, needed_cols, seed=Config.SEED + 1)
        rng_rf = np.random.RandomState(Config.SEED)
        imp_matrix, rf_r2, rf_details_df = compute_rf_importance(
            df_rf, Config.AE_COLS, env_cols, rng_rf)

        print("  Saving RF results...")
        imp_matrix.to_csv(f"{Config.RESULTS_DIR}/rf_importance_matrix.csv")
        pd.DataFrame([{"variable": k, "r2_cv": v} for k, v in rf_r2.items()]).to_csv(
            f"{Config.RESULTS_DIR}/rf_r2_scores.csv", index=False)
        if len(rf_details_df) > 0:
            rf_details_df.to_csv(f"{Config.RESULTS_DIR}/rf_details.csv", index=False)

        rf_r2_spatial, top_vars_cv, cv_df = compute_spatial_cv(
            df_rf, Config.AE_COLS, env_cols, corr_df, rf_r2)
        del df_rf; gc.collect()

    print("  Saving Spatial CV results...")
    cv_df.to_csv(f"{Config.RESULTS_DIR}/cv_comparison.csv", index=False)

    dd = build_dimension_dictionary(Config.AE_COLS, corr_df, imp_matrix)

    print("  Saving Dimension Dictionary...")
    dd.to_csv(f"{Config.RESULTS_DIR}/dimension_dictionary.csv", index=False)

    yearly_corr, stability_df, temporal_series_df = compute_temporal_spearman(
        Config.AE_COLS, env_cols, Config.DATA_DIR, Config.YEARS)

    grids, top_dims, dim_to_var, extent = prepare_conus_map_data(
        Config.AE_COLS, dd, Config.DATA_DIR, Config.MAP_YEAR)

    row_order, col_order, _, _ = compute_clustering(corr_df)

    # Save all results
    print_header("SAVING RESULTS")

    save_results(
        Config.RESULTS_DIR,
        spearman_matrix=corr_df,
        pval_matrix=pval_df,
        rf_importance_matrix=imp_matrix,
        dimension_dictionary=dd,
        cv_comparison=cv_df,
        temporal_stability=stability_df,
        temporal_series=temporal_series_df,
        conus_grids=grids,
    )

    if rf_r2:
        r2_df = pd.DataFrame([{'variable': k, 'r2_cv': v} for k, v in rf_r2.items()])
        r2_df.to_csv(f'{Config.RESULTS_DIR}/rf_r2_scores.csv', index=False)
        print(f"  Saved: rf_r2_scores.csv")

    if len(rf_details_df) > 0:
        rf_details_df.to_csv(f'{Config.RESULTS_DIR}/rf_details.csv', index=False)

    for yr, cdf in yearly_corr.items():
        cdf.to_csv(f'{Config.RESULTS_DIR}/temporal_spearman_{yr}.csv')
    print(f"  Saved: {len(yearly_corr)} temporal Spearman matrices")

    np.savez(f'{Config.RESULTS_DIR}/clustering_orders.npz',
             row_order=row_order, col_order=col_order)

    meta = {
        'env_cols': env_cols,
        'ae_cols': Config.AE_COLS,
        'top_dims': top_dims,
        'dim_to_var': dim_to_var,
        'extent': extent,
        'years': [int(y) for y in sorted(yearly_corr.keys())],
        'n_samples_spearman': int(Config.SPEARMAN_SAMPLE),
        'n_samples_rf': int(Config.RF_SAMPLE),
        'n_samples_temporal_per_year': int(Config.TEMPORAL_SAMPLE_PER_YEAR),
        'n_total_dataset': int(len(Config.YEARS) * 1_729_840),
        'spatial_block_deg': Config.SPATIAL_BLOCK_DEG,
    }
    save_results(Config.RESULTS_DIR, analysis_metadata=meta)

    # Summary
    elapsed = time.time() - start_time
    print_header("SUMMARY")
    n_agree = dd['agree'].sum()
    print(f"  Full dataset: ~{meta['n_total_dataset']:,} samples across {len(Config.YEARS)} years")
    print(f"  Spearman:  n={Config.SPEARMAN_SAMPLE:,} -> "
          f"{np.isfinite(corr_df.values).sum():,} pairs")
    print(f"  RF:        n={Config.RF_SAMPLE:,} -> {len(rf_r2)} models, "
          f"mean R2={np.nanmean(list(rf_r2.values())):.3f}")
    print(f"  Spatial CV gap: mean={cv_df['gap'].mean():.4f}")
    print(f"  Linear-RF agreement: {n_agree}/64 ({100*n_agree/64:.0f}%)")
    print(f"  Temporal stability: mean r={stability_df['mean_profile_corr'].mean():.4f}")
    print(f"  Top dim: {dd.iloc[0]['dimension']} -> "
          f"{Config.ENV_LABELS.get(dd.iloc[0]['sp_primary'], dd.iloc[0]['sp_primary'])} "
          f"(rho = {dd.iloc[0]['sp_rho']:+.3f})")
    print(f"  Duration: {elapsed/60:.1f} minutes")
    print(f"\n  All results saved to: {Config.RESULTS_DIR}/")


if __name__ == '__main__':
    main()
