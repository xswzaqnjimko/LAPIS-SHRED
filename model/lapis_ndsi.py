#!/usr/bin/env python3
"""
lapis_ndsi.py — LAPIS-SHRED on MODIS Snow Cover (Sierra Nevada)

Adapts the LAPIS architecture for NDSI snow dynamics: same location across
different years eliminates spatial domain shift.  Supports both forward
inference (initial window -> predict melt) and backward inference (terminal
frame -> reconstruct full season).

Directory layout:
    LAPIS-SHRED/
    ├── model/
    │   ├── shred_jax/
    │   ├── visualizations/
    │   └── lapis_ndsi.py        <- this script
    └── data/
        ├── sim_years/           <- ndsi_2020.npy ... ndsi_2024.npy
        ├── gt/                  <- ndsi_2025.npy
        └── metadata.json

Usage:
    python lapis_ndsi.py
    python lapis_ndsi.py --shred_mode frame
    python lapis_ndsi.py --inference_mode forward
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import jax
import jax.numpy as jnp

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from shred_jax import (
    Seq2SeqSHRED, FrameSHRED,
    EnsembleSeq2SeqDataset, EnsembleFrameDataset,
    ForwardFromWindow, BackwardFromTerminal,
    train_ensemble_shred, train_ensemble_frame_shred,
    train_forward_model, train_backward_model,
    extract_latent_trajectories_seq2seq, extract_latent_trajectories_frame,
    lapis_forward_inference_seq2seq, lapis_backward_inference_terminal_seq2seq,
    shred_baseline_seq2seq, shred_baseline_frame,
    compute_metrics, place_sensors, to_json_safe,
)
from visualizations import save_results_grid, save_timeseries
from visualizations.ndsi_plots import (
    plot_scaf_diagnostics, save_cut_data_preview,
    generate_fig1_sensor_plots, save_gt_video, save_ndsi_video,
)


# Configuration

class Config:
    """All tunables in one place. Edit here for PyCharm runs."""

    # Paths — resolved from script location
    BASE_DIR: Path = SCRIPT_DIR.parent
    DATA_DIR: Path = None
    SIM_DIR:  Path = None
    GT_DIR:   Path = None
    RESULTS_DIR: Path = None

    # Year split
    SIM_YEARS = [2020, 2021, 2022, 2023, 2024]
    GT_YEAR   = 2025

    # Observation budget (~7% of a 6-month season)
    GT_OBS_FRACTION = 0.07

    # Sensors
    N_SENSORS = 64
    SENSOR_STRATEGY = "stratified"
    SEED = 42

    # SHRED architecture
    SHRED_MODE     = "seq2seq"    # "frame" or "seq2seq"
    SEQ2SEQ_HIDDEN = 64           # BiLSTM hidden -> latent_dim = 128
    NUM_LAYERS     = 2
    DROPOUT_RATE   = 0.1
    LAGS           = 5            # time-delay embedding (frame mode)
    DECODER_LAYERS = (256, 256)   # MLP decoder (frame mode)

    # Temporal model
    FORWARD_HIDDEN = 64
    FORWARD_LAYERS = 2

    # Training
    BATCH_SIZE      = 4
    EPOCHS_SHRED    = 300
    LR_SHRED        = 1e-3
    EPOCHS_FORWARD  = 500
    LR_FORWARD      = 1e-3
    LAMBDA_RECON    = 2.0
    LAMBDA_ANCHOR   = 2.0
    LAMBDA_SHAPE    = 30.0
    WEIGHT_DECAY    = 1e-5

    # Loss weighting
    ACTIVE_WEIGHT    = 1.0
    ACTIVE_THRESHOLD = 0.0

    # SCAF endpoint cutting
    TAU           = 0.4
    RHO           = 0.25
    K_CONSEC      = 3
    SMOOTH_WINDOW = 5

    # Inference direction: "forward" or "backward"
    INFERENCE_MODE    = "backward"
    STATIC_PAD_LENGTH = 10

    # Video / animation
    VIDEO_FPS       = 5
    VIDEO_DPI       = 100
    GENERATE_VIDEOS = False # True → also save MP4
    GENERATE_FIG1   = True

    @classmethod
    def initialize(cls, base_dir: Path = None):
        if base_dir is not None:
            cls.BASE_DIR = base_dir
        cls.DATA_DIR = cls.BASE_DIR / "data"
        cls.SIM_DIR  = cls.DATA_DIR / "sim_years"
        cls.GT_DIR   = cls.DATA_DIR / "gt"
        return cls

    @classmethod
    def finalize_results_dir(cls):
        suffix = "results_forward" if cls.INFERENCE_MODE == "forward" else "results_backward"
        cls.RESULTS_DIR = cls.BASE_DIR / suffix
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls


# NDSI data helpers

def clean_ndsi(arr: np.ndarray) -> np.ndarray:
    """Clean raw NDSI data: handle fill values, scale to [-1, 1], forward-fill NaNs."""
    x = arr.astype(np.float32)
    x[x <= -10000] = np.nan
    finite_max = np.nanmax(x) if np.isfinite(np.nanmax(x)) else 0.0
    finite_min = np.nanmin(x) if np.isfinite(np.nanmin(x)) else 0.0
    if finite_max > 2.0 or finite_min < -2.0:
        x = x / 10000.0
    x = np.clip(x, -1.0, 1.0)
    for t in range(1, x.shape[0]):
        nan_mask = np.isnan(x[t])
        if np.any(nan_mask):
            x[t][nan_mask] = x[t - 1][nan_mask]
    x[0][np.isnan(x[0])] = 0.0
    return x


def compute_scaf(ndsi: np.ndarray, tau: float = 0.4) -> np.ndarray:
    """Compute Snow-Covered Area Fraction time-series."""
    T = ndsi.shape[0]
    scaf = np.zeros(T)
    for t in range(T):
        frame = ndsi[t]
        valid = np.isfinite(frame)
        n_valid = valid.sum()
        scaf[t] = (frame[valid] > tau).sum() / n_valid if n_valid > 0 else 0.0
    return scaf


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return x.copy()
    return np.convolve(x, np.ones(window) / window, mode='same')


def find_t_end(scaf_smooth: np.ndarray, rho: float = 0.25,
               k_consec: int = 3) -> int:
    """Find first time after peak where SCAF stays below rho for k consecutive frames."""
    t_peak = int(np.argmax(scaf_smooth))
    for t in range(t_peak, len(scaf_smooth) - k_consec + 1):
        if np.all(scaf_smooth[t:t + k_consec] < rho):
            return t
    return len(scaf_smooth) - 1


def cut_sequences_to_endpoints(arrays: List[np.ndarray], config,
                               labels=None):
    """Apply SCAF endpoint cutting to a list of NDSI grids.
    Returns cut grids and endpoint info dicts.
    """
    cut_grids, ep_info = [], []
    for i, arr in enumerate(arrays):
        scaf = compute_scaf(arr, tau=config.TAU)
        scaf_s = moving_average(scaf, window=config.SMOOTH_WINDOW)
        t_peak = int(np.argmax(scaf_s))
        t_end = find_t_end(scaf_s, rho=config.RHO, k_consec=config.K_CONSEC)
        t_start = min(t_peak, t_end)
        cut = arr[t_start:t_end + 1]
        cut_grids.append(cut)
        info = {"t_peak": t_peak, "t_end": t_end, "t_start": t_start,
                "T_orig": arr.shape[0], "T_cut": cut.shape[0]}
        if labels:
            info["label"] = labels[i]
        ep_info.append(info)
        label = labels[i] if labels else str(i)
        print(f"    {label}: T_orig={arr.shape[0]}, t_peak={t_peak}, "
              f"t_end={t_end}, T_cut={cut.shape[0]}")
    return cut_grids, ep_info


def load_years(config: Config):
    """Load simulation and ground-truth NDSI arrays."""
    sim_grids = []
    for yr in config.SIM_YEARS:
        p = config.SIM_DIR / f"ndsi_{yr}.npy"
        arr = clean_ndsi(np.load(p))
        sim_grids.append(arr)
        print(f"    Sim {yr}: {arr.shape}")

    gt = clean_ndsi(np.load(config.GT_DIR / f"ndsi_{config.GT_YEAR}.npy"))
    print(f"    GT {config.GT_YEAR}: {gt.shape}")

    meta = {}
    meta_path = config.DATA_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return sim_grids, gt, meta


# Main

def main():
    parser = argparse.ArgumentParser(description="LAPIS-NDSI")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--shred_mode", type=str, default=None, choices=["frame", "seq2seq"])
    parser.add_argument("--sensor_strategy", type=str, default=None,
                        choices=["stratified", "grid", "random"])
    parser.add_argument("--inference_mode", type=str, default=None,
                        choices=["forward", "backward"])
    parser.add_argument("--generate_videos", action="store_true")
    args = parser.parse_args()

    # Apply CLI overrides (Config defaults take priority for PyCharm runs)
    if args.base_dir is not None:
        config = Config.initialize(Path(args.base_dir))
    else:
        config = Config.initialize()
    if args.shred_mode is not None:
        config.SHRED_MODE = args.shred_mode
    if args.sensor_strategy is not None:
        config.SENSOR_STRATEGY = args.sensor_strategy
    if args.inference_mode is not None:
        config.INFERENCE_MODE = args.inference_mode
    if args.generate_videos:
        config.GENERATE_VIDEOS = True
    config.finalize_results_dir()

    use_backward = config.INFERENCE_MODE.lower() == "backward"
    mode_label = ("Backward Reconstruction (terminal -> full)" if use_backward
                  else "Forward Inference (initial -> future)")
    print(f"\n  LAPIS-NDSI: {mode_label}\n")

    rng = jax.random.PRNGKey(config.SEED)

    # [1] Load data
    print("[1] Loading data ...")
    sim_grids_raw, gt_grid_raw, meta = load_years(config)
    H, W = gt_grid_raw.shape[1], gt_grid_raw.shape[2]

    # [1b] SCAF endpoint cutting
    print("\n[1b] SCAF endpoint cutting ...")
    sim_labels = [str(yr) for yr in config.SIM_YEARS[:len(sim_grids_raw)]]
    sim_grids, sim_ep_info = cut_sequences_to_endpoints(sim_grids_raw, config, labels=sim_labels)
    gt_cut_list, gt_ep_list = cut_sequences_to_endpoints([gt_grid_raw], config,
                                                         labels=[str(config.GT_YEAR)])
    gt_grid = gt_cut_list[0]
    gt_ep_info = gt_ep_list[0]
    T_total = gt_grid.shape[0]
    obs_len = max(config.LAGS, int(T_total * config.GT_OBS_FRACTION))
    print(f"  GT: T_cut={T_total}, obs_len={obs_len} ({config.GT_OBS_FRACTION*100:.0f}%)")

    # Diagnostic plots
    plot_scaf_diagnostics(sim_grids_raw, gt_grid_raw, sim_ep_info, gt_ep_info, config,
                          config.RESULTS_DIR / "scaf_diagnostics.png")
    save_cut_data_preview(sim_grids, gt_grid, sim_ep_info, gt_ep_info, config,
                          config.RESULTS_DIR / "cut_data_preview.png")
    save_gt_video(gt_grid, config, config.RESULTS_DIR / "gt", "ground_truth")

    # [2] Sensors & dataset
    print("\n[2] Placing sensors & building dataset ...")
    sensors = place_sensors(sim_grids, H, W, config.N_SENSORS,
                            strategy=config.SENSOR_STRATEGY, seed=config.SEED)
    print(f"  {len(sensors)} sensors ({config.SENSOR_STRATEGY}), mode={config.SHRED_MODE}")

    active_mask = np.ones((H, W), dtype=bool)
    use_frame = config.SHRED_MODE.lower() == "frame"

    if use_frame:
        dataset = EnsembleFrameDataset(sim_grids, sensors, lags=config.LAGS, fit=True)
        print(f"  Dataset: {len(dataset)} frame samples")

        # [3] Train Frame SHRED
        print(f"\n[3] Training Frame SHRED ...")
        shred_model = FrameSHRED(
            n_sensors=config.N_SENSORS, hidden_size=config.SEQ2SEQ_HIDDEN,
            num_layers=config.NUM_LAYERS, decoder_layers=config.DECODER_LAYERS,
            state_dim=H * W, dropout_rate=config.DROPOUT_RATE)
        rng, srng = jax.random.split(rng)
        shred_state = train_ensemble_frame_shred(shred_model, dataset, active_mask, config, srng)

        # [4] Extract latents & train temporal model
        latent_trajectories, z_inits = extract_latent_trajectories_frame(
            shred_state, dataset, sim_grids, sensors, config)
        T_originals = [t.shape[0] for t in latent_trajectories]
        latent_dim = latent_trajectories[0].shape[1]

        # NOTE: frame-by-frame forward/backward inference uses the same
        # temporal models but requires frame-level encode/decode at inference.
        # For brevity, frame-mode inference reuses the seq2seq temporal pipeline
        # with the frame-extracted latent trajectories.
        # Full frame-mode inference (as in the original script) can be added
        # as needed for specific experiments.

    else:
        dataset = EnsembleSeq2SeqDataset(sim_grids, sensors, initial_pad=10, fit=True)
        print(f"  Dataset: {len(dataset)} sequences, T_padded={dataset.T_padded}")

        # [3] Train Seq2Seq SHRED
        print(f"\n[3] Training Seq2Seq SHRED ...")
        latent_dim = config.SEQ2SEQ_HIDDEN * 2
        shred_model = Seq2SeqSHRED(
            n_sensors=config.N_SENSORS, hidden_size=config.SEQ2SEQ_HIDDEN,
            num_layers=config.NUM_LAYERS, state_dim=H * W,
            dropout_rate=config.DROPOUT_RATE)
        rng, srng = jax.random.split(rng)
        shred_state = train_ensemble_shred(shred_model, dataset, active_mask, config, srng)

        # [4] Extract latent trajectories
        latent_trajectories, z_inits = extract_latent_trajectories_seq2seq(shred_state, dataset, config)
        T_originals = dataset.T_originals

    if use_backward:
        print("\n[4] Training Backward Model ...")
        backward_model = BackwardFromTerminal(
            latent_dim=latent_dim, hidden_dim=config.FORWARD_HIDDEN,
            num_layers=config.FORWARD_LAYERS, dropout_rate=config.DROPOUT_RATE)
        rng, brng = jax.random.split(rng)
        backward_state = train_backward_model(
            backward_model, latent_trajectories, T_originals, obs_len, config, brng,
            z_inits=z_inits)

        print(f"\n[5] LAPIS Backward Inference (terminal frame only) ...")
        pred_lapis = lapis_backward_inference_terminal_seq2seq(
            shred_state, backward_state, gt_grid, sensors, dataset, config)
    else:
        print("\n[4] Training Forward Model ...")
        forward_model = ForwardFromWindow(
            latent_dim=latent_dim, hidden_dim=config.FORWARD_HIDDEN,
            num_layers=config.FORWARD_LAYERS, dropout_rate=config.DROPOUT_RATE)
        rng, frng = jax.random.split(rng)
        forward_state = train_forward_model(
            forward_model, latent_trajectories, T_originals, obs_len, config, frng)

        print(f"\n[5] LAPIS Forward Inference (first {obs_len} frames) ...")
        pred_lapis = lapis_forward_inference_seq2seq(
            shred_state, forward_state, gt_grid, sensors, dataset, obs_len, config)

    # [6] SHRED baseline
    print("\n[6] SHRED baseline (full GT sensor time-series) ...")
    if use_frame:
        pred_shred = shred_baseline_frame(shred_state, gt_grid, sensors, dataset, config)
    else:
        pred_shred = shred_baseline_seq2seq(shred_state, gt_grid, sensors, dataset, config)

    # [7] Evaluate
    print("\n[7] Evaluation ...")
    metrics_lapis = compute_metrics(pred_lapis, gt_grid, active_mask)
    metrics_shred = compute_metrics(pred_shred, gt_grid, active_mask)
    gt_range = float(gt_grid.max() - gt_grid.min())
    nrmse_lapis = float(metrics_lapis['rmse_active']) / gt_range if gt_range > 0 else 0.0
    nrmse_shred = float(metrics_shred['rmse_active']) / gt_range if gt_range > 0 else 0.0

    print(f"\n  Full sequence (T={T_total}):")
    print(f"  GT range: [{gt_grid.min():.4f}, {gt_grid.max():.4f}], delta={gt_range:.4f}")
    print(f"  SHRED (full obs): RMSE={metrics_shred['rmse_active']:.4f} "
          f"SSIM={metrics_shred['ssim']:.4f} NRMSE={nrmse_shred:.4f}")
    print(f"  LAPIS ({config.INFERENCE_MODE}): RMSE={metrics_lapis['rmse_active']:.4f} "
          f"SSIM={metrics_lapis['ssim']:.4f} NRMSE={nrmse_lapis:.4f}")

    metrics_lapis_future, metrics_shred_future = None, None
    if not use_backward:
        gt_future = gt_grid[obs_len:]
        metrics_lapis_future = compute_metrics(pred_lapis[obs_len:], gt_future, active_mask)
        metrics_shred_future = compute_metrics(pred_shred[obs_len:], gt_future, active_mask)
        print(f"\n  Future only (t={obs_len}...{T_total-1}):")
        print(f"  SHRED: RMSE={metrics_shred_future['rmse_active']:.4f} "
              f"SSIM={metrics_shred_future['ssim']:.4f}")
        print(f"  LAPIS: RMSE={metrics_lapis_future['rmse_active']:.4f} "
              f"SSIM={metrics_lapis_future['ssim']:.4f}")

    # [8] Save results
    print("\n[8] Saving results ...")
    res = config.RESULTS_DIR

    save_results_grid(gt_grid, pred_lapis, pred_shred, obs_len,
                      metrics_lapis, metrics_shred, res / "lapis_ndsi_results.png",
                      sensor_locs=sensors, cmap="Blues_r", vmin=-1.0, vmax=1.0,
                      symmetric=False, origin="upper", field_label="NDSI",
                      title_prefix="LAPIS-NDSI")

    obs_side = "end" if use_backward else "start"
    save_timeseries(gt_grid, pred_lapis, pred_shred, sensors, obs_len,
                    res / "timeseries_comparison.png",
                    ylabel_fmt="NDSI ({r},{c})", title="LAPIS-NDSI: Sensor Time Series",
                    obs_side=obs_side)

    np.save(res / "pred_lapis.npy", pred_lapis.astype(np.float32))
    np.save(res / "pred_shred.npy", pred_shred.astype(np.float32))

    results_json = to_json_safe({
        "inference_mode": config.INFERENCE_MODE,
        "T_total": T_total, "obs_len": obs_len,
        "gt_obs_fraction": config.GT_OBS_FRACTION,
        "n_sensors": config.N_SENSORS, "sensor_strategy": config.SENSOR_STRATEGY,
        "shred_mode": config.SHRED_MODE,
        "scaf_params": {"tau": config.TAU, "rho": config.RHO,
                        "K_consec": config.K_CONSEC, "smooth_window": config.SMOOTH_WINDOW},
        "gt_endpoint": gt_ep_info, "sim_endpoints": sim_ep_info,
        "gt_range": gt_range,
        "lapis": {"rmse": metrics_lapis['rmse_active'], "ssim": metrics_lapis['ssim'],
                  "nrmse": nrmse_lapis},
        "shred": {"rmse": metrics_shred['rmse_active'], "ssim": metrics_shred['ssim'],
                  "nrmse": nrmse_shred},
        **({"metrics_future_shred": metrics_shred_future,
            "metrics_future_lapis": metrics_lapis_future}
           if metrics_lapis_future is not None else {}),
        "timestamp": datetime.now().isoformat(),
    })
    with open(res / "lapis_ndsi_metrics.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Metrics saved: {res / 'lapis_ndsi_metrics.json'}")

    # [9] Animations
    print("\n[9] Generating animations ...")
    mode_tag = config.INFERENCE_MODE.capitalize()
    save_ndsi_video(gt_grid, pred_lapis, f"LAPIS {mode_tag}", config,
                    res / "lapis", f"lapis_{config.INFERENCE_MODE}")
    save_ndsi_video(gt_grid, pred_shred, "SHRED (full obs)", config,
                    res / "shred", "shred_reconstruction")

    # [10] Figure 1 sensor overlays (optional)
    if config.GENERATE_FIG1:
        print("\n[10] Generating Fig 1 sensor plots ...")
        generate_fig1_sensor_plots(sim_grids, gt_grid, sensors, config, res)

    # Done
    print(f"\n  LAPIS-NDSI complete! Results: {res}")
    print(f"  Mode={config.INFERENCE_MODE}, SHRED={config.SHRED_MODE}, "
          f"sensors={config.N_SENSORS} ({config.SENSOR_STRATEGY})")
    print(f"  LAPIS RMSE={metrics_lapis['rmse_active']:.4f} SSIM={metrics_lapis['ssim']:.4f} "
          f"NRMSE={nrmse_lapis:.4f}")

    log_path = res / f"session_{datetime.now():%Y%m%d_%H%M%S}.txt"
    with open(log_path, "w") as f:
        f.write(f"LAPIS-NDSI {config.INFERENCE_MODE}\n")
        f.write(f"LAPIS RMSE={metrics_lapis['rmse_active']:.4f} NRMSE={nrmse_lapis:.4f}\n")
        f.write(f"SHRED RMSE={metrics_shred['rmse_active']:.4f} NRMSE={nrmse_shred:.4f}\n")


if __name__ == "__main__":
    main()
