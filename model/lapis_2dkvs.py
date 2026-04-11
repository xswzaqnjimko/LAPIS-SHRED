#!/usr/bin/env python3
"""
lapis_2dkvs.py — LAPIS forward/backward reconstruction for 2D von Karman Vortex Street

Periodic vortex shedding past a circular cylinder.  Supports both forward
prediction from an initial window and backward reconstruction from a terminal window.

Usage:
    python lapis_2dkvs.py
    python lapis_2dkvs.py --inference_mode forward
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import jax

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from shred_jax import (
    Seq2SeqSHRED, FrameSHRED, ForwardFromWindow, BackwardFromWindow,
    EnsembleSeq2SeqDataset, EnsembleFrameDataset,
    train_ensemble_shred, train_ensemble_frame_shred,
    train_forward_model, train_backward_model,
    extract_latent_trajectories_seq2seq, extract_latent_trajectories_frame,
    lapis_forward_inference_seq2seq, lapis_backward_inference_seq2seq,
    lapis_backward_inference_frame,
    shred_baseline_seq2seq, shred_baseline_frame,
    compute_metrics, place_sensors, TeeLogger, to_json_safe,
)
from visualizations.timeseries import save_timeseries
from visualizations.pde_plots import save_kvs_results


# Configuration

class Config:
    BASE_DIR: Path = SCRIPT_DIR.parent
    DATA_DIR: Path = None
    RESULTS_DIR: Path = None

    OBS_FRACTION   = 0.10
    N_SENSORS      = 5
    SENSOR_STRATEGY = "variance"
    SEED           = 42

    SHRED_MODE     = "seq2seq"
    SEQ2SEQ_HIDDEN = 80
    NUM_LAYERS     = 2
    DROPOUT_RATE   = 0.1
    LAGS           = 5
    DECODER_LAYERS = (256, 256)

    BACKWARD_HIDDEN = 80
    BACKWARD_LAYERS = 2
    FORWARD_HIDDEN  = 80
    FORWARD_LAYERS  = 2

    INFERENCE_MODE = "backward"

    BATCH_SIZE      = 16
    EPOCHS_SHRED    = 300
    LR_SHRED        = 1e-3
    EPOCHS_FORWARD  = 500
    LR_FORWARD      = 1e-3
    LAMBDA_RECON    = 1.0
    LAMBDA_ANCHOR   = 2.0
    LAMBDA_SHAPE    = 5.0
    WEIGHT_DECAY    = 1e-5

    ACTIVE_WEIGHT   = 1.0
    TERMINAL_PAD    = 10

    @classmethod
    def initialize(cls, base_dir=None):
        if base_dir is not None:
            cls.BASE_DIR = base_dir
        cls.DATA_DIR = cls.BASE_DIR / "data"
        return cls

    @classmethod
    def finalize_results_dir(cls):
        suffix = "results_forward" if cls.INFERENCE_MODE == "forward" else "results_backward"
        cls.RESULTS_DIR = cls.BASE_DIR / suffix
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Data loading

def load_data(config):
    data_dir = config.DATA_DIR
    sim_files = sorted(data_dir.glob("sim_*.npz"))
    sim_grids = []
    for fpath in sim_files:
        d = np.load(fpath)
        sim_grids.append(d["omega"].astype(np.float32))
        print(f"    {fpath.name}: omega={d['omega'].shape}")
    gt_data = np.load(data_dir / "gt.npz")
    gt_grid = gt_data["omega"].astype(np.float32)
    cyl_mask = gt_data["cyl_mask"].astype(np.float32) if "cyl_mask" in gt_data else None
    print(f"    gt.npz: omega={gt_grid.shape}")
    return sim_grids, gt_grid, cyl_mask


def place_sensors(sim_grids, Nx, Ny, n_sensors, strategy="variance", seed=42):
    rng_np = np.random.RandomState(seed)
    if strategy == "grid":
        n_side = int(np.ceil(np.sqrt(n_sensors)))
        rows = np.linspace(1, Nx - 2, n_side, dtype=int)
        cols = np.linspace(1, Ny - 2, n_side, dtype=int)
        rr, cc = np.meshgrid(rows, cols)
        locs = np.column_stack([rr.ravel(), cc.ravel()])[:n_sensors]
    elif strategy == "stratified":
        var_maps = [np.var(g, axis=0) for g in sim_grids]
        variance = np.mean(var_maps, axis=0)
        flat_var = variance.ravel()
        mask = np.zeros(Nx * Ny, dtype=bool)
        for i in range(Nx):
            for j in range(Ny):
                if 2 <= i < Nx - 2 and 2 <= j < Ny - 2:
                    mask[i * Ny + j] = True
        weights = np.where(mask, flat_var + 1e-6, 0.0)
        weights = weights / weights.sum()
        indices = rng_np.choice(Nx * Ny, size=n_sensors, replace=False, p=weights)
        rows = indices // Ny
        cols = indices % Ny
        locs = np.column_stack([rows, cols])
    elif strategy == "variance":
        var_maps = [np.var(g, axis=0) for g in sim_grids]
        variance = np.mean(var_maps, axis=0)
        variance[:1, :] = 0
        variance[-1:, :] = 0
        flat_var = variance.ravel() + 1e-8
        weights = flat_var / flat_var.sum()
        indices = rng_np.choice(Nx * Ny, size=n_sensors, replace=False, p=weights)
        rows = indices // Ny
        cols = indices % Ny
        locs = np.column_stack([rows, cols])
    else:
        rows = rng_np.randint(0, Nx, size=n_sensors)
        cols = rng_np.randint(0, Ny, size=n_sensors)
        locs = np.column_stack([rows, cols])
    return locs


# Main

def main():
    parser = argparse.ArgumentParser(description="LAPIS-2DKVS")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--shred_mode", type=str, default=None, choices=["frame", "seq2seq"])
    parser.add_argument("--inference_mode", type=str, default=None, choices=["forward", "backward"])
    parser.add_argument("--obs_fraction", type=float, default=None)
    args = parser.parse_args()

    if args.base_dir is not None:
        config = Config.initialize(Path(args.base_dir))
    else:
        config = Config.initialize()
    if args.shred_mode is not None:
        config.SHRED_MODE = args.shred_mode
    if args.inference_mode is not None:
        config.INFERENCE_MODE = args.inference_mode
    if args.obs_fraction is not None:
        config.OBS_FRACTION = args.obs_fraction
    config.finalize_results_dir()

    use_backward = config.INFERENCE_MODE == "backward"
    mode_label = "Backward" if use_backward else "Forward"
    print(f"\n  LAPIS-2DKVS: {mode_label} Reconstruction\n")

    # Session logging
    log_path = config.RESULTS_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.txt"
    sys.stdout = TeeLogger(log_path)

    rng = jax.random.PRNGKey(config.SEED)

    # [1] Load
    print("[1] Loading data ...")
    sim_grids, gt_grid, cyl_mask = load_data(config)
    T_total = gt_grid.shape[0]
    Nx, Ny = gt_grid.shape[1], gt_grid.shape[2]
    obs_len = max(config.LAGS + 1, int(T_total * config.OBS_FRACTION))
    print(f"  T={T_total}, spatial={Nx}x{Ny}, obs_len={obs_len}")

    # [2] Sensors
    print("\n[2] Sensors & dataset ...")
    sensors = place_sensors(sim_grids, Nx, Ny, config.N_SENSORS,
                            strategy=config.SENSOR_STRATEGY, seed=config.SEED)
    active_mask = np.ones((Nx, Ny), dtype=bool)
    if cyl_mask is not None:
        active_mask[cyl_mask > 0.5] = False
        print(f"  Cylinder mask: {int((~active_mask).sum())} cells excluded")

    use_frame = config.SHRED_MODE.lower() == "frame"

    if use_frame:
        dataset = EnsembleFrameDataset(sim_grids, sensors, lags=config.LAGS, fit=True)
        print(f"  {len(dataset)} frame samples")

        print("\n[3] Training Frame SHRED ...")
        shred_model = FrameSHRED(
            n_sensors=config.N_SENSORS, hidden_size=config.SEQ2SEQ_HIDDEN,
            num_layers=config.NUM_LAYERS, decoder_layers=config.DECODER_LAYERS,
            state_dim=Nx * Ny, dropout_rate=config.DROPOUT_RATE)
        rng, srng = jax.random.split(rng)
        shred_state = train_ensemble_frame_shred(shred_model, dataset, active_mask, config, srng)

        latent_trajectories, z_inits = extract_latent_trajectories_frame(
            shred_state, dataset, sim_grids, sensors, config)
        T_originals = [t.shape[0] for t in latent_trajectories]
        latent_dim = latent_trajectories[0].shape[1]
        obs_len_latent = max(2, obs_len - config.LAGS)
    else:
        dataset = EnsembleSeq2SeqDataset(sim_grids, sensors, initial_pad=config.TERMINAL_PAD, fit=True)
        print(f"  {len(dataset)} sequences")

        print("\n[3] Training Seq2Seq SHRED ...")
        latent_dim = config.SEQ2SEQ_HIDDEN * 2
        shred_model = Seq2SeqSHRED(
            n_sensors=config.N_SENSORS, hidden_size=config.SEQ2SEQ_HIDDEN,
            num_layers=config.NUM_LAYERS, state_dim=Nx * Ny,
            dropout_rate=config.DROPOUT_RATE)
        rng, srng = jax.random.split(rng)
        shred_state = train_ensemble_shred(shred_model, dataset, active_mask, config, srng)

        latent_trajectories, z_inits = extract_latent_trajectories_seq2seq(shred_state, dataset, config)
        T_originals = dataset.T_originals
        obs_len_latent = obs_len

    # [4] Temporal model
    if use_backward:
        print("\n[4] Training Backward Model ...")
        model = BackwardFromWindow(
            latent_dim=latent_dim, hidden_dim=config.BACKWARD_HIDDEN,
            num_layers=config.BACKWARD_LAYERS, dropout_rate=config.DROPOUT_RATE)
        rng, brng = jax.random.split(rng)
        temporal_state = train_backward_model(
            model, latent_trajectories, T_originals, obs_len_latent, config, brng)
    else:
        print("\n[4] Training Forward Model ...")
        model = ForwardFromWindow(
            latent_dim=latent_dim, hidden_dim=config.FORWARD_HIDDEN,
            num_layers=config.FORWARD_LAYERS, dropout_rate=config.DROPOUT_RATE)
        rng, frng = jax.random.split(rng)
        temporal_state = train_forward_model(
            model, latent_trajectories, T_originals, obs_len_latent, config, frng)

    # [5] Inference
    obs_label = "last" if use_backward else "first"
    print(f"\n[5] LAPIS {mode_label} Inference ({obs_label} {obs_len} frames) ...")
    if use_backward:
        if use_frame:
            pred_lapis = lapis_backward_inference_frame(
                shred_state, temporal_state, gt_grid, sensors, dataset, obs_len_latent, config)
        else:
            pred_lapis = lapis_backward_inference_seq2seq(
                shred_state, temporal_state, gt_grid, sensors, dataset, obs_len, config)
    else:
        pred_lapis = lapis_forward_inference_seq2seq(
            shred_state, temporal_state, gt_grid, sensors, dataset, obs_len, config)

    # [6] Baseline
    print("\n[6] SHRED baseline ...")
    if use_frame:
        pred_shred = shred_baseline_frame(shred_state, gt_grid, sensors, dataset, config)
    else:
        pred_shred = shred_baseline_seq2seq(shred_state, gt_grid, sensors, dataset, config)

    # Zero cylinder region
    if cyl_mask is not None:
        obstacle = cyl_mask > 0.5
        pred_lapis[:, obstacle] = 0.0
        pred_shred[:, obstacle] = 0.0

    # [7] Evaluate
    print("\n[7] Evaluation ...")
    metrics_lapis = compute_metrics(pred_lapis, gt_grid, active_mask)
    metrics_shred = compute_metrics(pred_shred, gt_grid, active_mask)
    gt_range = float(gt_grid.max() - gt_grid.min())
    nrmse_l = float(metrics_lapis['rmse_active']) / gt_range if gt_range > 0 else 0.0
    nrmse_s = float(metrics_shred['rmse_active']) / gt_range if gt_range > 0 else 0.0
    print(f"  SHRED: RMSE={metrics_shred['rmse_active']:.4f} SSIM={metrics_shred['ssim']:.4f} NRMSE={nrmse_s:.4f}")
    print(f"  LAPIS: RMSE={metrics_lapis['rmse_active']:.4f} SSIM={metrics_lapis['ssim']:.4f} NRMSE={nrmse_l:.4f}")

    # [8] Save
    print("\n[8] Saving ...")
    res = config.RESULTS_DIR
    save_kvs_results(gt_grid, pred_lapis, pred_shred,
                     res / "lapis_2dkvs_results.png", sensor_locs=sensors, trim_rows=2)
    save_timeseries(gt_grid, pred_lapis, pred_shred, sensors, obs_len,
                    res / "timeseries_comparison.png",
                    ylabel_fmt="w({r},{c})", title="LAPIS-2DKVS",
                    obs_side="end" if use_backward else "start")
    np.save(res / "pred_lapis.npy", pred_lapis.astype(np.float32))
    np.save(res / "pred_shred.npy", pred_shred.astype(np.float32))

    results_json = to_json_safe({
        "inference_mode": config.INFERENCE_MODE,
        "T_total": T_total, "obs_len": obs_len,
        "n_sensors": config.N_SENSORS, "shred_mode": config.SHRED_MODE, "gt_range": gt_range,
        "lapis": {"rmse": metrics_lapis['rmse_active'], "ssim": metrics_lapis['ssim'], "nrmse": nrmse_l},
        "shred": {"rmse": metrics_shred['rmse_active'], "ssim": metrics_shred['ssim'], "nrmse": nrmse_s},
        "timestamp": datetime.now().isoformat(),
    })
    with open(res / "lapis_2dkvs_metrics.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"  LAPIS-2DKVS complete! Results: {res}")
    if hasattr(sys.stdout, 'close_log'):
        sys.stdout.close_log()


if __name__ == "__main__":
    main()
