#!/usr/bin/env python
"""
03_transformer_analysis.py — TabTransformer Interpretability
=============================================================

Multi-task TabTransformer trained on GPU to predict environmental variables
from AlphaEarth 64-dimensional embeddings. Complements the linear (Spearman)
and tree-based (RF) analyses in 02_core_analysis.py by capturing nonlinear
and cross-dimension interactions.

Architecture:
  Each of the 64 embedding dimensions is treated as a token, projected to
  d_model via a learned linear layer with positional encoding. A 4-layer
  Transformer encoder captures inter-dimension dependencies. Task-specific
  MLP heads predict each of 26 environmental variables simultaneously.

Analysis outputs:
  1. Gradient-based importance matrix (64 x N_env): |d_yhat_j / d_x_i|
  2. Attention weight patterns (64 x 64): learned inter-dimension interactions
  3. R-squared scores with random and spatial cross-validation

Training uses 5M samples (stratified across 7 years), bf16 mixed precision,
cosine learning rate schedule with warmup, and early stopping.

Inputs:
  data/unified_conus/conus_{year}_unified.parquet
  results/analysis_metadata.json  (from 02_core_analysis.py)

Outputs:
  transformer_importance_matrix.csv, transformer_attention_matrix.csv,
  transformer_r2_scores.csv, transformer_training_log.csv,
  transformer_best.pt

Usage:
  python 03_transformer_analysis.py
  python 03_transformer_analysis.py --epochs 80 --batch-size 4096
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import gc
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    DATA_DIR = '../data/unified_conus'
    RESULTS_DIR = 'results'
    YEARS = list(range(2017, 2024))

    AE_COLS = [f'A{i:02d}' for i in range(64)]

    # Model architecture
    D_MODEL = 128
    N_HEADS = 8
    N_LAYERS = 4
    DROPOUT = 0.1
    FF_DIM = 512

    # Training
    TRAIN_SAMPLE = 5_000_000
    BATCH_SIZE = 2048
    EPOCHS = 60
    LR = 1e-4
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.05
    PATIENCE = 8

    # Evaluation
    SPATIAL_BLOCK_DEG = 2.0
    N_FOLDS = 5
    GRADIENT_SAMPLE = 200_000
    CV_EPOCHS = 25

    SEED = 42

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


def print_header(msg):
    print(f"\n{'='*70}")
    print(msg)
    print('='*70)


# =============================================================================
# 1. MODEL ARCHITECTURE
# =============================================================================

class AlphaEarthTransformer(nn.Module):
    """
    Multi-task TabTransformer for predicting environmental variables
    from AlphaEarth embedding dimensions.

    Each dimension is treated as a token (scalar -> d_model projection).
    Transformer encoder captures cross-dimension interactions.
    Independent MLP heads produce predictions for each target variable.
    """

    def __init__(self, n_dims=64, n_targets=22, d_model=128,
                 n_heads=8, n_layers=4, dropout=0.1, ff_dim=512):
        super().__init__()
        self.n_dims = n_dims
        self.n_targets = n_targets
        self.d_model = d_model

        self.value_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.pos_encoding = nn.Parameter(torch.randn(1, n_dims, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pool_norm = nn.LayerNorm(d_model)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            for _ in range(n_targets)
        ])

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        tokens = self.value_proj(x.unsqueeze(-1))
        tokens = tokens + self.pos_encoding

        if return_attention:
            attn_weights = []
            out = tokens
            for layer in self.transformer.layers:
                out2, attn = layer.self_attn(
                    layer.norm1(out), layer.norm1(out), layer.norm1(out),
                    need_weights=True, average_attn_weights=False
                )
                out = out + layer.dropout1(out2)
                out = out + layer._ff_block(layer.norm2(out))
                attn_weights.append(attn.detach())
            encoded = out
            attn_tensor = torch.stack(attn_weights, dim=1)
        else:
            encoded = self.transformer(tokens)
            attn_tensor = None

        pooled = self.pool_norm(encoded.mean(dim=1))
        preds = torch.cat([head(pooled) for head in self.heads], dim=-1)

        if return_attention:
            return preds, attn_tensor
        return preds


# =============================================================================
# 2. DATASET
# =============================================================================

class AlphaEarthDataset(Dataset):
    """Dataset wrapper handling NaN masking for multi-task training."""

    def __init__(self, X, Y, coords=None, blocks=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.coords = coords
        self.blocks = blocks
        self.mask = torch.tensor(np.isfinite(Y).astype(np.float32))
        self.Y = torch.nan_to_num(self.Y, nan=0.0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mask[idx]


# =============================================================================
# 3. DATA LOADING
# =============================================================================

def load_data(env_cols):
    """Load stratified subsample across all years."""
    print_header(f"LOADING DATA — {Config.TRAIN_SAMPLE:,} samples")

    rng = np.random.RandomState(Config.SEED)
    n_per_year = Config.TRAIN_SAMPLE // len(Config.YEARS)
    remainder = Config.TRAIN_SAMPLE - n_per_year * len(Config.YEARS)
    frames = []

    for i_yr, yr in enumerate(Config.YEARS):
        fp = f'{Config.DATA_DIR}/conus_{yr}_unified.parquet'
        if not os.path.exists(fp):
            continue

        needed = Config.AE_COLS + env_cols + ['longitude', 'latitude']
        import pyarrow.parquet as pq; all_cols = pq.read_schema(fp).names
        cols_to_load = [c for c in needed if c in all_cols]

        df_yr = pd.read_parquet(fp, columns=cols_to_load)

        # Require all AE dimensions valid
        ae_present = [c for c in Config.AE_COLS if c in df_yr.columns]
        mask_ae = df_yr[ae_present].notna().all(axis=1)

        # Require >=75% of environmental variables valid
        env_in_df = [c for c in env_cols if c in df_yr.columns]
        if env_in_df:
            y_valid_frac = df_yr[env_in_df].notna().mean(axis=1)
            mask_env = y_valid_frac >= 0.75
            mask = mask_ae & mask_env
        else:
            mask = mask_ae

        df_yr = df_yr[mask].reset_index(drop=True)
        n_available = len(df_yr)

        if n_available == 0:
            print(f"  {yr}: 0 valid samples, skipping")
            continue

        n_this = n_per_year + (remainder if i_yr == 0 else 0)
        n_this = min(n_this, n_available)

        idx = rng.choice(n_available, size=n_this, replace=False)
        df_yr = df_yr.iloc[idx]
        frames.append(df_yr)
        print(f"  {yr}: {n_this:>9,} / {n_available:>10,}")

    df = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')

    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"  Total: {len(df):,}  ({mem_mb:.0f} MB in RAM)")
    return df


def prepare_data(df, env_cols):
    """Standardize features and targets, assign spatial blocks."""
    X = df[Config.AE_COLS].values.astype(np.float32)
    Y = df[env_cols].values.astype(np.float32)
    coords = df[['longitude', 'latitude']].values.astype(np.float32)

    # Standardize X
    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X).astype(np.float32)

    # Standardize Y per variable (handling NaN)
    y_scalers = {}
    Y_scaled = np.full_like(Y, np.nan)
    for j, v in enumerate(env_cols):
        mask = np.isfinite(Y[:, j])
        if mask.sum() > 100:
            scaler = StandardScaler()
            Y_scaled[mask, j] = scaler.fit_transform(Y[mask, j:j+1]).ravel()
            y_scalers[v] = scaler

    # Spatial blocks for spatial CV
    bs = Config.SPATIAL_BLOCK_DEG
    blocks = (
        np.floor(coords[:, 0] / bs).astype(int).astype(str) + '_' +
        np.floor(coords[:, 1] / bs).astype(int).astype(str)
    )

    return X, Y_scaled, coords, blocks, x_scaler, y_scalers


# =============================================================================
# 4. TRAINING UTILITIES
# =============================================================================

def masked_mse_loss(pred, target, mask):
    """MSE loss computed only over non-NaN entries (masked multi-task loss)."""
    diff = (pred - target) ** 2
    masked = diff * mask
    loss_per_task = masked.sum(dim=0) / (mask.sum(dim=0) + 1e-8)
    return loss_per_task.mean()


def get_lr_scheduler(optimizer, total_steps, warmup_ratio):
    """Cosine annealing with linear warmup."""
    warmup_steps = int(total_steps * warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(model, train_loader, val_loader, device, n_epochs):
    """Train with early stopping, return model and training log."""
    optimizer = optim.AdamW(model.parameters(),
                            lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    total_steps = len(train_loader) * n_epochs
    scheduler = get_lr_scheduler(optimizer, total_steps, Config.WARMUP_RATIO)

    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float32
    if use_amp:
        print(f"  Mixed precision: {'bf16' if amp_dtype == torch.bfloat16 else 'fp32'}")

    best_val_loss = float('inf')
    patience_counter = 0
    training_log = []

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for X_batch, Y_batch, M_batch in pbar:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            M_batch = M_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                pred = model(X_batch)
                loss = masked_mse_loss(pred, Y_batch, M_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, Y_batch, M_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                Y_batch = Y_batch.to(device, non_blocking=True)
                M_batch = M_batch.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    pred = model(X_batch)
                    loss = masked_mse_loss(pred, Y_batch, M_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        lr = scheduler.get_last_lr()[0]
        training_log.append({'epoch': epoch+1, 'train_loss': train_loss,
                             'val_loss': val_loss, 'lr': lr})

        print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{Config.RESULTS_DIR}/transformer_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(f'{Config.RESULTS_DIR}/transformer_best.pt',
                                     weights_only=True))
    return model, pd.DataFrame(training_log)


# =============================================================================
# 5. IMPORTANCE AND ATTENTION EXTRACTION
# =============================================================================

def compute_gradient_importance(model, dataset, device, env_cols):
    """
    Compute mean absolute gradient |d_yhat_j / d_x_i| over a large sample.
    Produces a 64 x N_env importance matrix analogous to RF permutation importance.
    """
    print_header(f"GRADIENT IMPORTANCE (n={Config.GRADIENT_SAMPLE:,})")

    model.eval()
    n_samples = min(Config.GRADIENT_SAMPLE, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=512, shuffle=False,
                        num_workers=0, pin_memory=True)

    grad_sum = torch.zeros(64, len(env_cols), device=device)
    n_valid = torch.zeros(len(env_cols), device=device)

    for X_batch, Y_batch, M_batch in tqdm(loader, desc="Gradient importance"):
        X_batch = X_batch.to(device).requires_grad_(True)
        M_batch = M_batch.to(device)

        pred = model(X_batch)

        for j in range(len(env_cols)):
            valid = M_batch[:, j] > 0.5
            if valid.sum() == 0:
                continue

            grad_output = torch.zeros_like(pred)
            grad_output[:, j] = valid.float()

            if X_batch.grad is not None:
                X_batch.grad.zero_()

            pred.backward(grad_output, retain_graph=(j < len(env_cols) - 1))

            if X_batch.grad is not None:
                grad_sum[:, j] += X_batch.grad[valid].abs().sum(dim=0)
                n_valid[j] += valid.sum()

        X_batch = X_batch.detach()

    n_valid = n_valid.clamp(min=1)
    grad_avg = (grad_sum / n_valid.unsqueeze(0)).cpu().numpy()

    imp_df = pd.DataFrame(grad_avg, index=Config.AE_COLS, columns=env_cols)
    print(f"  Shape: {imp_df.shape}")
    return imp_df


def extract_attention_patterns(model, dataset, device):
    """Extract mean attention weights (64 x 64) across samples."""
    print_header("ATTENTION PATTERN EXTRACTION")

    model.eval()
    n_samples = min(20000, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False,
                        num_workers=0, pin_memory=True)

    attn_sum = None
    count = 0

    with torch.no_grad():
        for X_batch, _, _ in tqdm(loader, desc="Attention"):
            X_batch = X_batch.to(device)
            _, attn = model(X_batch, return_attention=True)
            attn_batch = attn.mean(dim=0)

            if attn_sum is None:
                attn_sum = attn_batch.cpu()
            else:
                attn_sum += attn_batch.cpu()
            count += 1

    attn_avg = attn_sum / count
    attn_all = attn_avg.mean(dim=(0, 1)).numpy()

    attn_df = pd.DataFrame(attn_all, index=Config.AE_COLS, columns=Config.AE_COLS)
    print(f"  Shape: {attn_df.shape}, range: [{attn_all.min():.4f}, {attn_all.max():.4f}]")
    return attn_df


# =============================================================================
# 6. R-SQUARED EVALUATION AND CROSS-VALIDATION
# =============================================================================

def evaluate_r2(model, dataset, device, env_cols):
    """Compute per-variable R-squared on a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=2048, shuffle=False,
                        num_workers=0, pin_memory=True)

    all_preds, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for X_batch, Y_batch, M_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu()
            all_preds.append(pred)
            all_targets.append(Y_batch)
            all_masks.append(M_batch)

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    masks = torch.cat(all_masks, dim=0).numpy()

    r2_scores = {}
    for j, v in enumerate(env_cols):
        valid = masks[:, j] > 0.5
        if valid.sum() < 100:
            r2_scores[v] = np.nan
            continue
        y_true = targets[valid, j]
        y_pred = preds[valid, j]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2_scores[v] = 1 - ss_res / (ss_tot + 1e-8)

    return r2_scores


def _train_cv_model(X_train, Y_train, X_val, Y_val, device, n_targets):
    """Train a single CV fold model with reduced epochs."""
    train_ds = AlphaEarthDataset(X_train, Y_train)
    val_ds = AlphaEarthDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True,
                              drop_last=True)

    model = AlphaEarthTransformer(
        n_dims=64, n_targets=n_targets,
        d_model=Config.D_MODEL, n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS, dropout=Config.DROPOUT,
        ff_dim=Config.FF_DIM,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR,
                            weight_decay=Config.WEIGHT_DECAY)
    total_steps = len(train_loader) * Config.CV_EPOCHS
    scheduler = get_lr_scheduler(optimizer, total_steps, Config.WARMUP_RATIO)

    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float32

    model.train()
    for epoch in range(Config.CV_EPOCHS):
        for Xb, Yb, Mb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            Yb = Yb.to(device, non_blocking=True)
            Mb = Mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                pred = model(Xb)
                loss = masked_mse_loss(pred, Yb, Mb)
            loss.backward()
            optimizer.step()
            scheduler.step()

    return model, val_ds


def cross_validate_transformer(X, Y, blocks, env_cols, device):
    """Random and spatial 5-fold cross-validation."""
    print_header(f"TRANSFORMER CROSS-VALIDATION (n={len(X):,}, {Config.CV_EPOCHS} epochs/fold)")

    n_targets = len(env_cols)

    # Random CV
    print("  Random 5-fold CV...")
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    random_r2_folds = {v: [] for v in env_cols}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"    Fold {fold+1}/{Config.N_FOLDS}")
        model, val_ds = _train_cv_model(
            X[train_idx], Y[train_idx], X[val_idx], Y[val_idx], device, n_targets)
        r2 = evaluate_r2(model, val_ds, device, env_cols)
        for v in env_cols:
            if np.isfinite(r2.get(v, np.nan)):
                random_r2_folds[v].append(r2[v])
        del model, val_ds
        torch.cuda.empty_cache(); gc.collect()

    random_r2 = {v: np.mean(s) if s else np.nan for v, s in random_r2_folds.items()}

    # Spatial block CV
    print("\n  Spatial block CV...")
    unique_blocks = np.unique(blocks)
    block_to_int = {b: i for i, b in enumerate(unique_blocks)}
    block_ids = np.array([block_to_int[b] for b in blocks])

    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    spatial_r2_folds = {v: [] for v in env_cols}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=block_ids)):
        print(f"    Fold {fold+1}/{Config.N_FOLDS}")
        model, val_ds = _train_cv_model(
            X[train_idx], Y[train_idx], X[val_idx], Y[val_idx], device, n_targets)
        r2 = evaluate_r2(model, val_ds, device, env_cols)
        for v in env_cols:
            if np.isfinite(r2.get(v, np.nan)):
                spatial_r2_folds[v].append(r2[v])
        del model, val_ds
        torch.cuda.empty_cache(); gc.collect()

    spatial_r2 = {v: np.mean(s) if s else np.nan for v, s in spatial_r2_folds.items()}

    # Report
    print(f"\n  {'Variable':25s} {'Random':>8s} {'Spatial':>8s} {'Gap':>8s}")
    print(f"  {'-'*51}")
    for v in sorted(env_cols, key=lambda k: random_r2.get(k, 0), reverse=True):
        r = random_r2.get(v, np.nan)
        s = spatial_r2.get(v, np.nan)
        g = r - s if np.isfinite(r) and np.isfinite(s) else np.nan
        print(f"  {Config.ENV_LABELS.get(v, v):25s} {r:8.4f} {s:8.4f} {g:+8.4f}")

    return random_r2, spatial_r2


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    for i, arg in enumerate(sys.argv):
        if arg == '--epochs' and i + 1 < len(sys.argv):
            Config.EPOCHS = int(sys.argv[i + 1])
        if arg == '--batch-size' and i + 1 < len(sys.argv):
            Config.BATCH_SIZE = int(sys.argv[i + 1])

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU detected. Training will be slow.")

    # Load metadata from prior analysis
    meta_path = f'{Config.RESULTS_DIR}/analysis_metadata.json'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        env_cols = meta['env_cols']
        print(f"Loaded metadata: {len(env_cols)} env vars from 02_core_analysis.py")
    else:
        print("WARNING: No metadata found. Discovering env vars from data...")
        for yr in Config.YEARS:
            fp = f'{Config.DATA_DIR}/conus_{yr}_unified.parquet'
            if os.path.exists(fp):
                import pyarrow.parquet as pq; cols = pq.read_schema(fp).names
                env_cols = [v for v in Config.ENV_LABELS.keys() if v in cols]
                break

    print(f"Environmental variables ({len(env_cols)})")

    # Load data
    df = load_data(env_cols)

    print("\n  Preparing arrays...")
    X, Y, coords, blocks, x_scaler, y_scalers = prepare_data(df, env_cols)
    del df; gc.collect()

    print(f"  X: {X.shape} ({X.nbytes/1e6:.0f} MB)")
    print(f"  Y: {Y.shape} ({Y.nbytes/1e6:.0f} MB)")
    print(f"  NaN fraction in Y: {np.isnan(Y).mean():.3f}")

    # Train/val split
    rng = np.random.RandomState(Config.SEED)
    n = len(X)
    idx = rng.permutation(n)
    n_train = int(0.85 * n)

    train_ds = AlphaEarthDataset(X[idx[:n_train]], Y[idx[:n_train]])
    val_ds = AlphaEarthDataset(X[idx[n_train:]], Y[idx[n_train:]])

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train: {n_train:,}, Val: {n - n_train:,}")

    # Build and train model
    model = AlphaEarthTransformer(
        n_dims=64, n_targets=len(env_cols),
        d_model=Config.D_MODEL, n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS, dropout=Config.DROPOUT,
        ff_dim=Config.FF_DIM,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters")

    model, training_log = train_model(model, train_loader, val_loader,
                                       device, Config.EPOCHS)

    # Validation R-squared
    print_header("VALIDATION R-SQUARED")
    val_r2 = evaluate_r2(model, val_ds, device, env_cols)
    for v in sorted(val_r2, key=lambda k: val_r2.get(k, 0), reverse=True):
        if np.isfinite(val_r2[v]):
            print(f"  {Config.ENV_LABELS.get(v, v):25s}: R2 = {val_r2[v]:.4f}")

    training_log.to_csv(f'{Config.RESULTS_DIR}/transformer_training_log.csv', index=False)

    # Gradient importance
    full_ds = AlphaEarthDataset(X, Y)
    trans_imp = compute_gradient_importance(model, full_ds, device, env_cols)
    trans_imp.to_csv(f'{Config.RESULTS_DIR}/transformer_importance_matrix.csv')

    # Attention patterns
    attn_df = extract_attention_patterns(model, full_ds, device)
    attn_df.to_csv(f'{Config.RESULTS_DIR}/transformer_attention_matrix.csv')

    # Cross-validation
    random_r2, spatial_r2 = cross_validate_transformer(
        X, Y, blocks, env_cols, device)

    # Save all results
    print_header("SAVING RESULTS")

    trans_imp.to_csv(f'{Config.RESULTS_DIR}/transformer_importance_matrix.csv')
    print(f"  Saved: transformer_importance_matrix.csv")

    attn_df.to_csv(f'{Config.RESULTS_DIR}/transformer_attention_matrix.csv')
    print(f"  Saved: transformer_attention_matrix.csv")

    training_log.to_csv(f'{Config.RESULTS_DIR}/transformer_training_log.csv', index=False)
    print(f"  Saved: transformer_training_log.csv")

    r2_rows = []
    for v in env_cols:
        r2_rows.append({
            'variable': v,
            'val_r2': val_r2.get(v, np.nan),
            'random_cv_r2': random_r2.get(v, np.nan),
            'spatial_cv_r2': spatial_r2.get(v, np.nan),
            'gap': random_r2.get(v, 0) - spatial_r2.get(v, 0),
        })
    r2_df = pd.DataFrame(r2_rows)
    r2_df.to_csv(f'{Config.RESULTS_DIR}/transformer_r2_scores.csv', index=False)
    print(f"  Saved: transformer_r2_scores.csv")

    # Update dimension dictionary with transformer results
    dd_path = f'{Config.RESULTS_DIR}/dimension_dictionary.csv'
    if os.path.exists(dd_path):
        dd = pd.read_csv(dd_path)
        for _, row in dd.iterrows():
            d = row['dimension']
            if d in trans_imp.index:
                top_var = trans_imp.loc[d].idxmax()
                dd.loc[dd['dimension'] == d, 'trans_primary'] = top_var
                dd.loc[dd['dimension'] == d, 'trans_imp'] = trans_imp.loc[d, top_var]
                dd.loc[dd['dimension'] == d, 'trans_category'] = \
                    Config.ENV_CATEGORY.get(top_var, '?')

        dd['agree_3way'] = (
            (dd['sp_primary'] == dd['rf_primary']) &
            (dd['sp_primary'] == dd.get('trans_primary', ''))
        )
        dd.to_csv(dd_path, index=False)
        print(f"  Updated: dimension_dictionary.csv")

    # Summary
    elapsed = time.time() - start_time
    print_header("SUMMARY")
    print(f"  Training samples: {Config.TRAIN_SAMPLE:,}")
    print(f"  Model: d={Config.D_MODEL}, h={Config.N_HEADS}, L={Config.N_LAYERS}, "
          f"params={n_params:,}")
    print(f"  Best val loss: {training_log['val_loss'].min():.4f}")
    print(f"  Mean R2 random CV: {np.nanmean(list(random_r2.values())):.4f}")
    print(f"  Mean R2 spatial CV: {np.nanmean(list(spatial_r2.values())):.4f}")
    print(f"  Duration: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
