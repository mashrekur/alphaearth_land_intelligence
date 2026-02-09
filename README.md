# AlphaEarth Embedding Interpretability

Analysis code for the paper:

> **Physically Interpretable Satellite Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence**  
> Mashrekur Rahman  
> Dartmouth College

## Overview

This repository contains the analysis pipeline for interpreting Google AlphaEarth satellite foundation model embeddings. The 64-dimensional annual embeddings are systematically evaluated against 26 environmental variables across the Continental United States (CONUS) using 12.1 million co-located samples spanning 2017–2023 at 0.025° grid spacing (~2.75 km).

Three complementary analytical methods are applied:

- **Spearman rank correlation** (n = 1,000,000): linear monotonic relationships between each embedding dimension and environmental variable
- **Random Forest permutation importance** (n = 700,000): nonlinear predictive relationships with spatial block cross-validation
- **Multi-task TabTransformer** (n = 5,000,000): deep learning-based gradient importance and cross-dimension attention patterns

## Repository Structure

```
├── 01_data_extraction.py        # Google Earth Engine data extraction pipeline
├── 02_core_analysis.py          # Spearman, Random Forest, spatial CV, temporal analysis
├── 03_transformer_analysis.py   # TabTransformer training, gradient importance, attention
├── 04_analysis_figures.ipynb    # Publication-quality figure generation
├── requirements.txt             # Python dependencies
└── README.md
```

## Pipeline

### Step 1: Data Extraction (`01_data_extraction.py`)

Extracts co-located AlphaEarth embeddings and environmental variables from Google Earth Engine across a regular CONUS grid. Requires an authenticated GEE account and project with compute quota.

**Data sources:**

| Category | Source | Variables |
|----------|--------|-----------|
| Embeddings | Google AlphaEarth V1 Annual | 64 dimensions (A00–A63) |
| Terrain | USGS SRTM 30m | Elevation, slope, aspect |
| Soil | OpenLandMap | Clay fraction, organic carbon, pH, water capacity |
| Hydrology | HydroSHEDS, ERA5-Land | Flow accumulation, soil moisture, runoff, ET |
| Vegetation | MODIS (MOD13A2, MOD15A2H) | NDVI, EVI, LAI |
| Temperature | MODIS (MOD11A2), PRISM | LST day/night, mean temperature, dewpoint |
| Climate | PRISM, ERA5-Land | Annual/max monthly precipitation |
| Land surface | MODIS (MCD43A3) | Albedo |
| Land cover | USGS NLCD | Impervious surface percentage |
| Urban | VIIRS DNB, GPWv4 | Nighttime lights, population density |
| Forest | Hansen GFC | Tree cover circa 2000 |

```bash
# Extract all years (2017–2023)
python 01_data_extraction.py

# Extract specific years only
python 01_data_extraction.py 2020 2021
```

### Step 2: Core Analysis (`02_core_analysis.py`)

Runs Spearman correlation, Random Forest permutation importance, spatial block cross-validation (2° blocks, ~222 km), temporal stability analysis (7 independent yearly subsamples), and dimension dictionary construction.

```bash
# Full analysis
python 02_core_analysis.py

# Skip RF if cached results exist
python 02_core_analysis.py --skip-rf
```

**Key outputs:**
- `spearman_matrix.csv`: 64 × 26 Spearman ρ matrix
- `rf_importance_matrix.csv`: 64 × 26 RF permutation importance matrix
- `cv_comparison.csv`: Random vs. spatial block CV R² comparison
- `temporal_stability.csv`: Per-dimension inter-year profile correlation
- `dimension_dictionary.csv`: Per-dimension semantic labels with method agreement

### Step 3: Transformer Analysis (`03_transformer_analysis.py`)

Trains a multi-task TabTransformer (4-layer, 8-head, d=128) on 5M samples with bf16 mixed precision. Extracts gradient-based importance and attention weight matrices. Runs both random and spatial 5-fold cross-validation.

```bash
# Default training (60 epochs, batch size 2048)
python 03_transformer_analysis.py

# Custom settings
python 03_transformer_analysis.py --epochs 80 --batch-size 4096
```

**Requires GPU.** Tested on NVIDIA RTX 5090 (32 GB VRAM).

**Key outputs:**
- `transformer_importance_matrix.csv`: 64 × 26 gradient importance matrix
- `transformer_attention_matrix.csv`: 64 × 64 mean attention weights
- `transformer_r2_scores.csv`: Per-variable R² with random and spatial CV

### Step 4: Figures (`04_analysis_figures.ipynb`)

Generates all publication figures from the CSV outputs of Steps 2–3. No data extraction or model training required — operates entirely on saved results.

## Hardware Requirements

- **Steps 1**: Google Earth Engine account with compute quota
- **Step 2**: 32 GB RAM recommended (subsampling parameters are configurable)
- **Step 3**: CUDA-capable GPU with ≥16 GB VRAM
- **Step 4**: Standard workstation

## Data Availability

The extracted dataset (~12.1M samples) and all intermediate analysis results (correlation matrices, importance matrices, dimension dictionary, etc.) are available upon request from the corresponding author.

The raw input data are publicly available through Google Earth Engine and require no special access beyond a standard GEE account. All Earth Engine asset identifiers and extraction parameters are documented in `01_data_extraction.py`.

## Citation

If you use this code, please cite:

```bibtex
@article{rahman2025alphaearth,
  title={Physically Interpretable Satellite Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence},
  author={Rahman, Mashrekur},
  journal={Remote Sensing of Environment},
  year={2025}
}
```

## License

This project is released under the MIT License.
