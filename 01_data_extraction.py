#!/usr/bin/env python
"""
01_data_extraction.py — Unified CONUS Data Extraction
=====================================================

Extracts co-located AlphaEarth satellite foundation model embeddings and
environmental variables across the contiguous United States (CONUS) for
multiple years using Google Earth Engine.

For each grid point and year, the script extracts:
  - 64-dimensional AlphaEarth annual embeddings
  - Static terrain, soil, and hydrological variables (SRTM, OpenLandMap, HydroSHEDS)
  - Annual vegetation indices (MODIS NDVI, EVI, LAI)
  - Annual climate variables (PRISM temperature, precipitation)
  - Annual land surface temperature and albedo (MODIS LST, MCD43A3)
  - Annual hydro-meteorological variables (ERA5-Land soil moisture, runoff, ET)
  - Nighttime lights (VIIRS DNB) and population density (GPWv4)
  - Land cover and impervious surface (NLCD)

Grid: 0.025 deg spacing (~2.75 km), yielding ~2.3M points per year.
Extraction uses buffered point sampling (500 m radius) with batch processing
and checkpointing for fault tolerance.

NLCD year mapping:
  - 2017-2020: NLCD 2016 from USGS/NLCD_RELEASES/2019_REL
  - 2021-2023: NLCD 2021 from USGS/NLCD_RELEASES/2021_REL

Usage:
  python 01_data_extraction.py              # Extract all years (2017-2023)
  python 01_data_extraction.py 2019 2020    # Extract specific years only

Requirements:
  - Google Earth Engine Python API (authenticated)
  - Access to a GEE project with compute quota
"""

import ee
import pandas as pd
import numpy as np
import os
import sys
import time
import json
import logging
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_extraction.log'),
        logging.StreamHandler()
    ]
)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Google Earth Engine project ID — replace with your own
    PROJECT_ID = 'your-gee-project-id'

    CONUS_BBOX = {
        'west': -125.0,
        'south': 24.5,
        'east': -66.5,
        'north': 49.5
    }

    GRID_SPACING = 0.025  # ~2.75 km
    BATCH_SIZE = 400
    CHECKPOINT_EVERY = 50
    EXTRACTION_SCALE = 1000
    POINT_BUFFER = 500

    YEARS = list(range(2017, 2024))

    OUTPUT_DIR = 'data/unified_conus'
    CHECKPOINT_DIR = 'data/unified_conus/checkpoints'

    SLEEP_BETWEEN_BATCHES = 0.2

    AE_COLLECTION = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
    AE_BANDS = [f'A{i:02d}' for i in range(64)]


# =============================================================================
# EARTH ENGINE INITIALIZATION
# =============================================================================

def initialize_ee():
    """Initialize Earth Engine with project credentials."""
    try:
        ee.Initialize(project=Config.PROJECT_ID)
    except:
        ee.Authenticate()
        ee.Initialize(project=Config.PROJECT_ID)

    logging.info(f"Earth Engine initialized: {Config.PROJECT_ID}")
    return True


# =============================================================================
# GRID GENERATION
# =============================================================================

def generate_sample_grid():
    """Generate a regular lat/lon grid covering CONUS."""
    logging.info("Generating CONUS sampling grid...")

    lons = np.arange(Config.CONUS_BBOX['west'], Config.CONUS_BBOX['east'], Config.GRID_SPACING)
    lats = np.arange(Config.CONUS_BBOX['south'], Config.CONUS_BBOX['north'], Config.GRID_SPACING)

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = pd.DataFrame({
        'longitude': lon_grid.flatten(),
        'latitude': lat_grid.flatten(),
        'point_id': range(lon_grid.size)
    })

    logging.info(f"Generated {len(points):,} grid points")
    return points


# =============================================================================
# CHECKPOINTING
# =============================================================================

def get_checkpoint_path(year):
    return f"{Config.CHECKPOINT_DIR}/checkpoint_{year}.json"


def load_checkpoint(year):
    """Resume extraction from the last saved checkpoint for a given year."""
    checkpoint_path = get_checkpoint_path(year)

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            last_batch = checkpoint.get('last_batch_idx', 0)
            records_file = checkpoint.get('records_file')

            if records_file and os.path.exists(records_file):
                records_df = pd.read_parquet(records_file)
                records = records_df.to_dict('records')
                logging.info(f"  Resuming from batch {last_batch}, {len(records):,} records loaded")
                return last_batch, records
        except Exception as e:
            logging.warning(f"  Failed to load checkpoint: {e}")

    return 0, []


def save_checkpoint(year, batch_idx, records):
    """Save extraction progress to disk for fault tolerance."""
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    records_file = f"{Config.CHECKPOINT_DIR}/records_{year}_partial.parquet"
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(records_file, compression='snappy', index=False)

    checkpoint = {
        'year': year,
        'last_batch_idx': batch_idx,
        'n_records': len(records),
        'records_file': records_file,
        'timestamp': datetime.now().isoformat()
    }

    with open(get_checkpoint_path(year), 'w') as f:
        json.dump(checkpoint, f, indent=2)


def clear_checkpoint(year):
    """Remove checkpoint files after successful year completion."""
    for f in [get_checkpoint_path(year), f"{Config.CHECKPOINT_DIR}/records_{year}_partial.parquet"]:
        if os.path.exists(f):
            os.remove(f)


# =============================================================================
# IMAGE COMPOSITES
# =============================================================================

def create_static_environmental_composite():
    """
    Create composite of time-invariant environmental variables.

    Sources:
      - Terrain: USGS SRTM 30m (elevation, slope, aspect)
      - Soil: OpenLandMap (clay fraction, organic carbon, pH, water capacity)
      - Hydrology: HydroSHEDS 15-arcsec flow accumulation (log-transformed)
      - Forest: Hansen Global Forest Change (tree cover circa 2000)
    """
    logging.info("Creating static environmental composite...")

    # Terrain (SRTM)
    srtm = ee.Image('USGS/SRTMGL1_003')
    elevation = srtm.select('elevation').rename('elevation')
    terrain = ee.Terrain.products(srtm)
    slope = terrain.select('slope').rename('slope')
    aspect = terrain.select('aspect').rename('aspect')

    # Soil properties (OpenLandMap)
    soil_clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02') \
        .select('b0').rename('soil_clay_pct')
    soil_oc = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02') \
        .select('b0').rename('soil_organic_carbon')
    soil_ph = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02') \
        .select('b0').divide(10).rename('soil_ph')
    soil_awc = ee.Image('OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01') \
        .select('b0').rename('soil_water_capacity')

    # Hydrology (HydroSHEDS)
    flow_acc = ee.Image('WWF/HydroSHEDS/15ACC') \
        .select('b1').log10().rename('flow_acc_log')

    # Tree cover baseline
    tree_cover = ee.Image('UMD/hansen/global_forest_change_2024_v1_12') \
        .select('treecover2000').rename('tree_cover_2000')

    static = (elevation
              .addBands(slope)
              .addBands(aspect)
              .addBands(soil_clay)
              .addBands(soil_oc)
              .addBands(soil_ph)
              .addBands(soil_awc)
              .addBands(flow_acc)
              .addBands(tree_cover))

    bands = static.bandNames().getInfo()
    logging.info(f"  Static bands ({len(bands)}): {bands}")
    return static


def create_annual_environmental_composite(year):
    """
    Create composite of annual environmental variables for a given year.

    Sources:
      - NLCD land cover and impervious surface (year-matched)
      - MODIS vegetation indices (MOD13A2), LAI (MOD15A2H)
      - MODIS land surface temperature (MOD11A2)
      - MODIS broadband albedo (MCD43A3)
      - PRISM monthly climate normals (precipitation, temperature)
      - ERA5-Land monthly aggregates (soil moisture, runoff, ET)
      - VIIRS nighttime lights (DNB monthly composites)
      - GPWv4 population density (2020 snapshot)
    """
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'

    bands_list = []

    # NLCD — year-matched to available releases
    try:
        if year <= 2020:
            nlcd = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2016')
        else:
            nlcd = ee.Image('USGS/NLCD_RELEASES/2021_REL/NLCD/2021')

        landcover = nlcd.select('landcover').rename('nlcd_landcover')
        impervious = nlcd.select('impervious').rename('impervious_pct')
        bands_list.extend([landcover, impervious])

    except Exception as e:
        logging.warning(f"  NLCD not available for {year}: {e}")

    # MODIS vegetation indices (MOD13A2, 1 km, 16-day)
    modis_veg = ee.ImageCollection('MODIS/061/MOD13A2').filterDate(start_date, end_date)
    ndvi_mean = modis_veg.select('NDVI').mean().multiply(0.0001).rename('ndvi_mean')
    ndvi_max = modis_veg.select('NDVI').max().multiply(0.0001).rename('ndvi_max')
    evi_mean = modis_veg.select('EVI').mean().multiply(0.0001).rename('evi_mean')
    bands_list.extend([ndvi_mean, ndvi_max, evi_mean])

    # MODIS LAI (MOD15A2H, 500 m, 8-day)
    modis_lai = ee.ImageCollection('MODIS/061/MOD15A2H').filterDate(start_date, end_date)
    lai_mean = modis_lai.select('Lai_500m').mean().multiply(0.1).rename('lai_mean')
    bands_list.append(lai_mean)

    # MODIS LST (MOD11A2, 1 km, 8-day)
    modis_lst = ee.ImageCollection('MODIS/061/MOD11A2').filterDate(start_date, end_date)
    lst_day = modis_lst.select('LST_Day_1km').mean().multiply(0.02).subtract(273.15).rename('lst_day_c')
    lst_night = modis_lst.select('LST_Night_1km').mean().multiply(0.02).subtract(273.15).rename('lst_night_c')
    bands_list.extend([lst_day, lst_night])

    # MODIS albedo (MCD43A3, 500 m, daily)
    modis_albedo = ee.ImageCollection('MODIS/061/MCD43A3').filterDate(start_date, end_date)
    albedo = modis_albedo.select('Albedo_WSA_shortwave').mean().multiply(0.001).rename('albedo')
    bands_list.append(albedo)

    # PRISM monthly climate (4 km)
    prism = ee.ImageCollection('OREGONSTATE/PRISM/AN81m').filterDate(start_date, end_date)
    precip_annual = prism.select('ppt').sum().rename('precip_annual_mm')
    precip_max = prism.select('ppt').max().rename('precip_max_month')
    temp_mean = prism.select('tmean').mean().rename('temp_mean_c')
    temp_range = prism.select('tdmean').mean().rename('temp_range_c')
    bands_list.extend([precip_annual, precip_max, temp_mean, temp_range])

    # ERA5-Land monthly aggregates (~11 km)
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filterDate(start_date, end_date)
    soil_moisture = era5.select('volumetric_soil_water_layer_1').mean().rename('soil_moisture')
    runoff = era5.select('surface_runoff_sum').sum().multiply(1000).rename('runoff_annual_mm')
    et = era5.select('total_evaporation_sum').sum().multiply(-1000).rename('et_annual_mm')
    bands_list.extend([soil_moisture, runoff, et])

    # VIIRS nighttime lights (500 m, monthly)
    viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').filterDate(start_date, end_date)
    nightlights = viirs.select('avg_rad').mean().rename('nightlights')
    bands_list.append(nightlights)

    # Population density (GPWv4, ~1 km, 2020 snapshot)
    pop = ee.ImageCollection('CIESIN/GPWv411/GPW_Population_Density') \
        .filterDate('2020-01-01', '2020-12-31').first()
    pop_density = pop.select('population_density').rename('pop_density')
    bands_list.append(pop_density)

    # Combine all bands
    annual = bands_list[0]
    for band in bands_list[1:]:
        annual = annual.addBands(band)

    return annual


def get_alphaearth_image(year):
    """Retrieve AlphaEarth annual embedding mosaic for a given year."""
    dataset = ee.ImageCollection(Config.AE_COLLECTION)
    image = dataset.filterDate(f'{year}-01-01', f'{year}-12-31').mosaic()
    return image


# =============================================================================
# UNIFIED EXTRACTION
# =============================================================================

def extract_batch_unified(points_batch, ae_image, env_image, year):
    """Extract AlphaEarth embeddings and environmental data for a batch of points."""
    features = []
    for _, row in points_batch.iterrows():
        point = ee.Geometry.Point([row['longitude'], row['latitude']])
        buffered = point.buffer(Config.POINT_BUFFER)
        features.append(ee.Feature(buffered, {
            'point_id': int(row['point_id']),
            'lon': row['longitude'],
            'lat': row['latitude']
        }))

    fc = ee.FeatureCollection(features)
    combined = ae_image.addBands(env_image)

    try:
        sampled = combined.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=Config.EXTRACTION_SCALE,
            tileScale=4
        ).getInfo()['features']

        records = []
        for feat in sampled:
            props = feat['properties']

            # Require valid AlphaEarth data
            if props.get('A00') is None:
                continue

            record = {
                'point_id': props['point_id'],
                'longitude': props['lon'],
                'latitude': props['lat'],
                'year': year
            }

            for key, val in props.items():
                if key not in ['point_id', 'lon', 'lat']:
                    record[key] = val

            records.append(record)

        return records

    except Exception as e:
        logging.warning(f"Batch extraction failed: {str(e)[:100]}")
        return []


def extract_year_unified(grid_points, static_env, year):
    """Extract all data for a single year with checkpointing."""
    logging.info(f"\n{'='*60}")
    logging.info(f"EXTRACTING YEAR {year}")
    logging.info(f"{'='*60}")

    output_file = f"{Config.OUTPUT_DIR}/conus_{year}_unified.parquet"

    # Check if already complete
    if os.path.exists(output_file):
        existing = pd.read_parquet(output_file)
        if len(existing) > len(grid_points) * 0.5:
            logging.info(f"Skipping {year} — already complete ({len(existing):,} samples)")
            return
        else:
            logging.info(f"Existing file has only {len(existing):,} samples, re-extracting...")

    # Load checkpoint
    start_batch_idx, all_records = load_checkpoint(year)

    # Get images
    ae_image = get_alphaearth_image(year)
    annual_env = create_annual_environmental_composite(year)
    env_image = static_env.addBands(annual_env)

    # Calculate batches
    n_total_batches = (len(grid_points) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    start_idx = start_batch_idx * Config.BATCH_SIZE

    logging.info(f"  Total batches: {n_total_batches:,}")
    logging.info(f"  Starting from batch: {start_batch_idx}")
    logging.info(f"  Records already collected: {len(all_records):,}")

    batch_num = start_batch_idx

    for i in tqdm(range(start_idx, len(grid_points), Config.BATCH_SIZE),
                  desc=f"Year {year}",
                  initial=start_batch_idx,
                  total=n_total_batches):

        batch = grid_points.iloc[i:i + Config.BATCH_SIZE]
        records = extract_batch_unified(batch, ae_image, env_image, year)
        all_records.extend(records)

        batch_num += 1

        if batch_num % Config.CHECKPOINT_EVERY == 0:
            save_checkpoint(year, batch_num, all_records)
            logging.info(f"  Checkpoint saved: batch {batch_num}, {len(all_records):,} records")

        time.sleep(Config.SLEEP_BETWEEN_BATCHES)

    # Save final output
    if all_records:
        df = pd.DataFrame(all_records)
        ae_valid = df[Config.AE_BANDS].notna().all(axis=1).sum()

        df.to_parquet(output_file, compression='snappy', index=False)
        clear_checkpoint(year)

        logging.info(f"\nYear {year} complete:")
        logging.info(f"  Total records: {len(df):,}")
        logging.info(f"  Valid AE embeddings: {ae_valid:,} ({100*ae_valid/len(df):.1f}%)")
        logging.info(f"  Columns: {len(df.columns)}")
        logging.info(f"  Saved: {output_file}")
    else:
        logging.error(f"No data extracted for {year}")


def combine_all_years():
    """Combine all yearly parquet files into a single dataset."""
    logging.info("\n" + "="*60)
    logging.info("COMBINING ALL YEARS")
    logging.info("="*60)

    all_dfs = []
    for year in Config.YEARS:
        f = f'{Config.OUTPUT_DIR}/conus_{year}_unified.parquet'
        if os.path.exists(f):
            df = pd.read_parquet(f)
            if len(df) > 0:
                all_dfs.append(df)
                logging.info(f"  {year}: {len(df):,} samples")
            else:
                logging.warning(f"  {year}: Empty file, skipping")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        output_path = f'{Config.OUTPUT_DIR}/conus_all_years_unified.parquet'
        combined.to_parquet(output_path, compression='snappy', index=False)

        logging.info(f"\nCombined dataset saved:")
        logging.info(f"  File: {output_path}")
        logging.info(f"  Total samples: {len(combined):,}")
        logging.info(f"  Years: {sorted(combined['year'].unique())}")
        logging.info(f"  Columns: {len(combined.columns)}")

        return combined

    return None


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_extraction_approach():
    """Test extraction at a sample point to validate the pipeline."""
    logging.info("\n" + "="*60)
    logging.info("VERIFYING EXTRACTION APPROACH")
    logging.info("="*60)

    test_lon, test_lat = -100.0, 39.0

    point = ee.Geometry.Point([test_lon, test_lat])
    buffered = point.buffer(Config.POINT_BUFFER)
    fc = ee.FeatureCollection([ee.Feature(buffered, {'lon': test_lon, 'lat': test_lat})])

    ae = ee.ImageCollection(Config.AE_COLLECTION)
    ae_image = ae.filterDate('2020-01-01', '2020-12-31').mosaic()

    try:
        result = ae_image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=Config.EXTRACTION_SCALE
        ).getInfo()['features']

        if result and result[0]['properties'].get('A00') is not None:
            props = result[0]['properties']
            logging.info(f"Verification passed:")
            logging.info(f"  Test point: ({test_lon}, {test_lat})")
            logging.info(f"  A00 = {props.get('A00'):.4f}")
            valid_dims = sum(1 for k, v in props.items()
                           if k.startswith('A') and len(k) == 3 and v is not None)
            logging.info(f"  Valid dimensions: {valid_dims}/64")
            return True
        else:
            logging.error("Extraction returned None")
            return False

    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = datetime.now()

    # Parse command line args for specific years
    if len(sys.argv) > 1:
        try:
            years_to_run = [int(y) for y in sys.argv[1:]]
            Config.YEARS = years_to_run
            logging.info(f"Running specific years: {years_to_run}")
        except ValueError:
            logging.error("Invalid year arguments. Usage: python script.py 2019 2020")
            return

    logging.info("="*70)
    logging.info("UNIFIED CONUS DATA EXTRACTION")
    logging.info(f"Started: {start_time}")
    logging.info("="*70)

    logging.info(f"\nConfiguration:")
    logging.info(f"  Grid spacing: {Config.GRID_SPACING} deg ({Config.GRID_SPACING * 111:.2f} km)")
    logging.info(f"  Years to process: {Config.YEARS}")
    logging.info(f"  Output directory: {Config.OUTPUT_DIR}")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    if not initialize_ee():
        return

    if not verify_extraction_approach():
        logging.error("Verification failed. Aborting.")
        return

    # Load or generate grid
    grid_file = f'{Config.OUTPUT_DIR}/sample_grid.parquet'
    if os.path.exists(grid_file):
        logging.info(f"Loading existing grid from {grid_file}")
        grid_points = pd.read_parquet(grid_file)
    else:
        grid_points = generate_sample_grid()
        grid_points.to_parquet(grid_file, index=False)

    static_env = create_static_environmental_composite()

    for year in Config.YEARS:
        extract_year_unified(grid_points, static_env, year)

    combine_all_years()

    logging.info(f"\n{'='*70}")
    logging.info(f"COMPLETE — Duration: {datetime.now() - start_time}")
    logging.info(f"{'='*70}")


if __name__ == "__main__":
    main()
