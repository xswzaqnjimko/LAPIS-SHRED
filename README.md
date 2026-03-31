# LAPIS-SHRED

**LAtent Phase Inference from Short time sequences using Shallow REcurrent Decoders**

**LAPIS-SHRED** reconstructs or forecasts complete spatiotemporal dynamics from sparse sensor measurements confined to a short temporal window. It operates through a three-stage pipeline: 

(i) a SHRED model is pre-trained entirely on simulation data to map sensor time-histories into a structured latent space, 

(ii) a temporal sequence model, also trained on simulation-derived latent trajectories, learns to propagate latent states forward or backward in time to span unobserved temporal regions from short observational time windows, and 

(iii) at deployment, only a short observation window of hyper-sparse sensor measurements from the true system is provided, from which the frozen SHRED model and the temporal model jointly reconstruct or forecast the complete spatiotemporal trajectory.

The framework supports bidirectional (in time) inference, inherits data assimilation and multiscale reconstruction capabilities from its modular structure, and accommodates extreme observational constraints including single-frame terminal inputs. 

## Repository Structure

```
LAPIS-SHRED/
├── README.md
├── LICENSE
├── data/
│   ├── data_generation_2dks.py
│   ├── data_generation_2dkf.py
│   ├── data_generation_2dkvs.py
│   └── data_generation_ndsi.py
├── model/
│   ├── shred_jax/                   # Shared JAX/Flax ML library
│   │   ├── shred.py                     - SHRED models, losses, metrics
│   │   ├── datasets.py                  - ensemble datasets (sensor_extract_fn hook)
│   │   ├── temporal_models.py           - Forward/BackwardFromWindow, BackwardFromTerminal
│   │   ├── training.py                  - all training loops
│   │   ├── inference.py                 - LAPIS inference pipelines + baselines
│   │   └── utils.py                     - sensors, logging, JSON
│   ├── visualizations/              # Plotting utilities
│   │   ├── results_grid.py              - generic 4×5 comparison
│   │   ├── timeseries.py                - per-sensor time-series
│   │   ├── ndsi_plots.py                - SCAF, snow anims, Fig1 sensors
│   │   └── pde_plots.py                 - SymLogNorm/LogNorm figures, KVS 3×5
│   ├── lapis_2dks.py
│   ├── lapis_2dkf.py
│   ├── lapis_2dkvs.py
│   └── lapis_ndsi.py
├── quick_startup/NDSI_demo          # NDSI snow melt example: run lapis_ndsi_demo.py
└── demo_videos/                     # NDSI snow melt example visualizations
```

## Requirements

```
jax
jaxlib
flax
optax
numpy
scikit-learn
scipy
matplotlib
earthengine-api   # for NDSI data download only
```

## Usage - e.g. NDSI

**1. Download data** (requires Google Earth Engine account):
```bash
cd data
python data_generation_ndsi.py --project_id YOUR_GEE_PROJECT
```

**2. Run experiment:**
```bash
cd model
python lapis_ndsi.py                             # backward (default)
python lapis_ndsi.py --inference_mode forward    # forward
python lapis_ndsi.py --shred_mode frame          # frame-by-frame SHRED
```

All configuration is set at the top of `lapis_ndsi.py`.

**3. Output Files**

Results are saved to `results_forward/` or `results_backward/`:

| File | Description |
|------|-------------|
| `pred_lapis.npy` | LAPIS reconstructed fields |
| `pred_shred.npy` | SHRED baseline fields |
| `lapis_ndsi_metrics.json` | RMSE, SSIM, NRMSE |
| `lapis_ndsi_results.png` | GT / LAPIS / SHRED comparison |
| `timeseries_comparison.png` | Per-sensor time-series |
| `scaf_diagnostics.png` | SCAF endpoint cutting |
| `gifs/`, `videos/` | Reconstruction animations |

## Demo Videos

![NDSI Forward Reconstruction Demo](demo_videos/lapis_forward_combined.gif)
![NDSI Backward Reconstruction Demo](demo_videos/lapis_backward_combined.gif)

## Citation

```
Paper under review.
```
