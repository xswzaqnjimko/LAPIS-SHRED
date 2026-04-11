"""
LAPIS-NDSI Data Generation: MODIS Snow Cover (Sierra Nevada)
=============================================================

Downloads MODIS daily snow-cover NDSI for a FIXED region across multiple years.

Key difference from NDVI tile approach:
  - SAME location, DIFFERENT years → eliminates spatial domain shift.
  - Sim years: 2020–2024 (each year = one "simulation").
  - GT year : 2025 (same location, only early-season observed by LAPIS).

MODIS snow products used:
  - MOD10A1.061: Terra Snow Cover Daily (500 m)   → NDSI_Snow_Cover band
  - MYD10A1.061: Aqua  Snow Cover Daily (500 m)   → NDSI_Snow_Cover band

Output structure:
    LAPIS-SHRED/
    ├── data/
    │   ├── data_generation_ndsi.py      ← this script
    │   ├── sim_years/
    │   │   ├── ndsi_2020.npy   (T, H, W)  float32
    │   │   ├── ndsi_2021.npy
    │   │   ├── ...
    │   │   └── ndsi_2024.npy
    │   ├── gt/
    │   │   └── ndsi_2025.npy   (T, H, W)  float32
    │   ├── metadata.json
    │   └── visualization/
    │       └── data_preview.png
    └── model/
        └── lapis_ndsi.py + shred_jax/

Usage:
    python data_generation_ndsi.py
    python data_generation_ndsi.py --project_id YOUR_GEE_PROJECT
    python data_generation_ndsi.py --sim_years 2019 2020 2021 2022 2023 --gt_year 2024

Requirements:
    pip install earthengine-api numpy matplotlib
"""

import ee
import numpy as np
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import urllib.request
import tempfile
import os


# CONFIGURATION

PROJECT_ID = 'YOUR_GEE_PROJECT_ID'

# Region: Sierra Nevada / Tahoe (fixed across all years)
CENTER_LAT = 38.90
CENTER_LON = -120.10
BOX_LAT = (38.83264, 38.96736)
BOX_LON = (-120.18670, -120.01330)

# Spatial
GRID_SIZE = 64          # output pixels per side
SCALE = 500             # MODIS 500 m resolution

# Temporal
SEASON_START = "12-01"  # Dec 1
SEASON_END   = "05-31"  # May 31 (next calendar year)
TARGET_IMAGES = 180      # target frames per season

# Year split
SIM_YEARS = [2020, 2021, 2022, 2023, 2024]
GT_YEAR   = 2025

# Quality
MAX_CLOUD_PCT = 30      # max % cloud/missing pixels allowed per frame

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
DATA_DIR   = BASE_DIR / "data"


# GEE INITIALISATION

def init_gee(project_id):
    try:
        ee.Initialize(project=project_id)
        print(f"✓ GEE initialized (project: {project_id})")
    except Exception:
        print("Running authentication …")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("✓ Authenticated and initialized")


# MODIS SNOW COLLECTION

def get_study_region():
    """Fixed bounding box around Sierra Nevada."""
    return ee.Geometry.Rectangle([BOX_LON[0], BOX_LAT[0], BOX_LON[1], BOX_LAT[1]])


def get_sample_region(region):
    """Get exact region for target grid size."""
    centroid = region.centroid(maxError=1)
    half = GRID_SIZE * SCALE / 2
    return centroid.buffer(half, maxError=1).bounds()


def mask_bad_snow(image):
    """
    MOD10A1 NDSI_Snow_Cover encoding:
      0–100 : NDSI snow cover %
      200   : missing data
      201   : no decision
      211   : night
      237   : inland water
      239   : ocean
      250   : cloud
      254   : detector saturated
      255   : fill

    Keep only 0–100 (valid NDSI); mask everything else.
    """
    ndsi = image.select('NDSI_Snow_Cover')
    valid = ndsi.lte(100)
    # Scale 0–100 → 0.0–1.0
    ndsi_scaled = ndsi.toFloat().divide(100.0).rename('NDSI')
    return image.addBands(ndsi_scaled).updateMask(valid)


def get_snow_collection(region, start_date, end_date):
    """Get combined Terra + Aqua MODIS daily snow cover."""
    terra = (ee.ImageCollection('MODIS/061/MOD10A1')
             .filterBounds(region)
             .filterDate(start_date, end_date))

    aqua = (ee.ImageCollection('MODIS/061/MYD10A1')
            .filterBounds(region)
            .filterDate(start_date, end_date))

    combined = terra.merge(aqua)
    processed = combined.map(mask_bad_snow)
    return processed.sort('system:time_start')


def season_dates(year):
    """
    Return (start, end) for a snow season.
    Season = Dec 1 of (year-1) → May 31 of year.
    So "season 2024" means Dec 2023 → May 2024.
    """
    start = f"{year - 1}-{SEASON_START}"
    end   = f"{year}-{SEASON_END}"
    return start, end


# DOWNLOAD

def download_as_numpy(image, region, band='NDSI'):
    url = image.select(band).clip(region).getDownloadURL({
        'scale': SCALE,
        'region': region,
        'format': 'NPY',
        'crs': 'EPSG:4326',
    })
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        temp_path = f.name
    try:
        urllib.request.urlretrieve(url, temp_path)
        arr = np.load(temp_path, allow_pickle=True)
        if hasattr(arr, 'dtype') and arr.dtype.names is not None:
            arr = arr[band].astype(np.float32)
        else:
            arr = arr.astype(np.float32)
        return arr
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def download_season(region, sample_region, year, target_images=TARGET_IMAGES):
    """Download one snow season worth of NDSI frames."""
    start, end = season_dates(year)
    label = f"Season {year} ({start} → {end})"
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    collection = get_snow_collection(region, start, end)
    n_total = collection.size().getInfo()
    print(f"  Total images in collection: {n_total}")

    if n_total == 0:
        print("  ⚠ No images found!")
        return None, []

    img_list = collection.toList(n_total)

    # Equally spaced candidate indices (oversample ×1.5)
    n_cand = min(int(target_images * 1.5), n_total)
    cand_idx = np.unique(np.linspace(0, n_total - 1, n_cand, dtype=int))

    print(f"  Sampling {len(cand_idx)} candidates, target {target_images} valid …")

    data_list, dates_list = [], []
    expected = GRID_SIZE * GRID_SIZE

    for i in cand_idx:
        if len(data_list) >= target_images:
            break
        try:
            img = ee.Image(img_list.get(int(i)))
            date = img.date().format('YYYY-MM-dd').getInfo()

            # Check coverage
            cnt = img.select('NDSI').reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=sample_region,
                scale=SCALE,
                maxPixels=1e9,
            ).get('NDSI').getInfo()
            cov = (cnt or 0) / expected * 100

            if cov >= (100 - MAX_CLOUD_PCT):
                arr = download_as_numpy(img, sample_region, 'NDSI')
                # Resize to GRID_SIZE × GRID_SIZE if needed
                if arr.shape[0] != GRID_SIZE or arr.shape[1] != GRID_SIZE:
                    new = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)
                    h, w = min(arr.shape[0], GRID_SIZE), min(arr.shape[1], GRID_SIZE)
                    new[:h, :w] = arr[:h, :w]
                    arr = new

                data_list.append(arr)
                dates_list.append(date)
                if len(data_list) % 10 == 0:
                    print(f"    ✓ {len(data_list)}/{target_images}")

            time.sleep(0.3)
        except Exception as e:
            print(f"    [{i}] error: {str(e)[:60]}")
            continue

    if not data_list:
        print("  ✗ No valid images!")
        return None, []

    data = np.stack(data_list, axis=0).astype(np.float32)
    print(f"  ✓ {len(data_list)} images, shape {data.shape}")
    print(f"    dates: {dates_list[0]} → {dates_list[-1]}")
    return data, dates_list


# POST-PROCESSING

def fill_nan_temporal(data):
    """Fill NaN with temporal interpolation, pixel by pixel."""
    if data is None:
        return None
    out = data.copy()
    T, H, W = data.shape
    for i in range(H):
        for j in range(W):
            s = data[:, i, j]
            if np.any(np.isnan(s)):
                v = np.where(~np.isnan(s))[0]
                if len(v) == 0:
                    out[:, i, j] = np.nanmean(data)
                elif len(v) < T:
                    out[:, i, j] = np.interp(np.arange(T), v, s[v])
    return out


# VISUALISATION  (preview: sim-years-mean, GT, NDSI distribution)

def create_preview(sim_dict, gt_data, gt_dates, vis_dir):
    """
    Preview figure: one row per year (5 sim + 1 GT), 5 temporal snapshots each.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable, skipping preview)")
        return

    n_cols = 5
    years_sorted = sorted(sim_dict.keys())
    all_rows = [(str(yr), sim_dict[yr], "Sim") for yr in years_sorted]
    # Infer GT year from the last sim year + 1, or use 2025 as default
    gt_year = years_sorted[-1] + 1 if years_sorted else 2025
    all_rows.append((str(gt_year), gt_data, "GT"))
    n_rows = len(all_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    vmin, vmax = 0.0, 1.0
    cmap = "Blues_r"

    for row_i, (label, arr, kind) in enumerate(all_rows):
        T = arr.shape[0]
        t_idxs = np.linspace(0, T - 1, n_cols, dtype=int)
        for col, t in enumerate(t_idxs):
            ax = axes[row_i, col]
            ax.imshow(arr[t], cmap=cmap, vmin=vmin, vmax=vmax)
            if col == 0:
                ax.text(
                    0.02, 0.98, f"{kind} {label}\nT={T}",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
                )
            ax.set_title(f"t={t}", fontsize=8)
            ax.axis("off")
        # Row label on the left
        axes[row_i, 0].set_ylabel(f"{kind} {label}\n(T={arr.shape[0]})",
                                   fontsize=9, rotation=0, labelpad=55,
                                   va="center")

    plt.tight_layout()
    plt.suptitle("LAPIS-NDSI — Per-Year Snow Cover Preview", fontsize=13, y=1.02)
    vis_dir.mkdir(parents=True, exist_ok=True)
    out = vis_dir / "data_preview.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {out}")


# MAIN

def main():
    parser = argparse.ArgumentParser(description="LAPIS-NDSI data generation")
    parser.add_argument("--project_id", type=str, default=PROJECT_ID)
    parser.add_argument("--sim_years", type=int, nargs="+", default=SIM_YEARS)
    parser.add_argument("--gt_year",   type=int, default=GT_YEAR)
    parser.add_argument("--target_images", type=int, default=TARGET_IMAGES)
    parser.add_argument("--base_dir",  type=str, default=str(BASE_DIR))
    args = parser.parse_args()

    base = Path(args.base_dir)
    data_dir = base / "data"
    sim_dir  = data_dir / "sim_years"
    gt_dir   = data_dir / "gt"
    vis_dir  = data_dir / "visualization"
    for d in [sim_dir, gt_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LAPIS-NDSI Data Generation (MODIS Snow Cover)")
    print("=" * 60)
    print(f"  Region : Sierra Nevada ({CENTER_LAT}, {CENTER_LON})")
    print(f"  Grid   : {GRID_SIZE}×{GRID_SIZE} @ {SCALE} m")
    print(f"  Season : {SEASON_START} → {SEASON_END}")
    print(f"  Sim yrs: {args.sim_years}")
    print(f"  GT year: {args.gt_year}")

    init_gee(args.project_id)

    region = get_study_region()
    sample_region = get_sample_region(region)

    #  Download simulation years (skip if already downloaded) 
    sim_dict = {}
    for yr in args.sim_years:
        npy_path = sim_dir / f"ndsi_{yr}.npy"
        if npy_path.exists():
            print(f"  ✓ ndsi_{yr}.npy already exists, loading …")
            sim_dict[yr] = np.load(npy_path)
            continue
        data, dates = download_season(region, sample_region, yr, args.target_images)
        if data is not None:
            data = fill_nan_temporal(data)
            np.save(npy_path, data)
            print(f"  ✓ Saved ndsi_{yr}.npy  {data.shape}")
            sim_dict[yr] = data

    #  Download GT year (skip if already downloaded) 
    gt_npy_path = gt_dir / f"ndsi_{args.gt_year}.npy"
    gt_dates = []
    if gt_npy_path.exists():
        print(f"  ✓ ndsi_{args.gt_year}.npy already exists, loading …")
        gt_data = np.load(gt_npy_path)
    else:
        gt_data, gt_dates = download_season(region, sample_region, args.gt_year,
                                            args.target_images)
        if gt_data is not None:
            gt_data = fill_nan_temporal(gt_data)
            np.save(gt_npy_path, gt_data)
            print(f"  ✓ Saved ndsi_{args.gt_year}.npy  {gt_data.shape}")

    #  Metadata 
    meta = {
        "experiment": "LAPIS-NDSI",
        "region": {
            "center_lat": CENTER_LAT, "center_lon": CENTER_LON,
            "box_lat": list(BOX_LAT), "box_lon": list(BOX_LON),
        },
        "grid_size": GRID_SIZE, "scale_m": SCALE,
        "season": f"{SEASON_START} → {SEASON_END}",
        "sim_years": args.sim_years,
        "gt_year": args.gt_year,
        "sim_shapes": {str(yr): list(sim_dict[yr].shape) for yr in sim_dict},
        "gt_shape": list(gt_data.shape) if gt_data is not None else None,
        "gt_dates": gt_dates,
        "created": datetime.now().isoformat(),
    }
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Saved metadata.json")

    #  Preview 
    if gt_data is not None and sim_dict:
        create_preview(sim_dict, gt_data, gt_dates, vis_dir)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Output: {data_dir}")
    print(f"  Sim: {len(sim_dict)} years saved")
    if gt_data is not None:
        print(f"  GT:  {gt_data.shape}")
    print(f"\n  Next: python lapis_ndsi.py --base_dir {base}")


if __name__ == "__main__":
    main()
