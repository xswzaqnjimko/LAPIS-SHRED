#!/usr/bin/env python3
"""
lapis_2dkf.py — LAPIS backward reconstruction for 2D Kolmogorov Flow

Two-channel velocity field (u,v); sensors measure vorticity.
Evaluates on both velocity speed |u| and vorticity omega.

Usage:
    python lapis_2dkf.py
    python lapis_2dkf.py --shred_mode seq2seq
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import jax
from scipy.fft import fftfreq

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from shred_jax import (
    Seq2SeqSHRED, FrameSHRED, BackwardFromWindow,
    EnsembleSeq2SeqDataset, EnsembleFrameDataset,
    train_ensemble_shred, train_ensemble_frame_shred, train_backward_model,
    extract_latent_trajectories_seq2seq, extract_latent_trajectories_frame,
    lapis_backward_inference_seq2seq, lapis_backward_inference_frame,
    shred_baseline_seq2seq, shred_baseline_frame,
    compute_metrics, place_sensors, to_json_safe,
)
from visualizations.timeseries import save_timeseries
from visualizations.pde_plots import save_velocity_and_vorticity


# Configuration

class Config:
    BASE_DIR: Path = SCRIPT_DIR.parent
    DATA_DIR: Path = None
    RESULTS_DIR: Path = None

    OBS_FRACTION   = 0.10
    N_SENSORS      = 8
    SENSOR_STRATEGY = "variance"
    SEED           = 42

    SHRED_MODE     = "frame"
    SEQ2SEQ_HIDDEN = 4
    NUM_LAYERS     = 2
    DROPOUT_RATE   = 0.1
    LAGS           = 2
    DECODER_LAYERS = (256, 256)

    BACKWARD_HIDDEN = 96
    BACKWARD_LAYERS = 2

    BATCH_SIZE      = 64
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
        cls.RESULTS_DIR = cls.BASE_DIR / "results"
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls


# KF-specific helpers

def compute_vorticity_from_uv(uv_field, Lx=2*np.pi, Ly=2*np.pi):
    """Spectral vorticity: omega = dv/dx - du/dy.  Input: (..., 2, Nx, Ny)."""
    u = uv_field[..., 0, :, :]
    v = uv_field[..., 1, :, :]
    Nx, Ny = uv_field.shape[-2], uv_field.shape[-1]
    kx = fftfreq(Nx, d=Lx / (2 * np.pi * Nx)) * 2 * np.pi
    ky = fftfreq(Ny, d=Ly / (2 * np.pi * Ny)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    omega_hat = 1j * KX * np.fft.fft2(v) - 1j * KY * np.fft.fft2(u)
    return np.real(np.fft.ifft2(omega_hat))


def vorticity_sensor_extract(grid, sensor_locs):
    """Extract sensors from vorticity of a UV velocity field (T, 2, Nx, Ny)."""
    vort = compute_vorticity_from_uv(grid)
    return np.stack([vort[:, r, c] for r, c in sensor_locs], axis=1)


# Data loading

def load_data(config):
    data_dir = config.DATA_DIR
    sim_files = sorted(data_dir.glob("sim_*.npz"))
    sim_grids, sim_vorts = [], []
    for fpath in sim_files:
        d = np.load(fpath)
        sim_grids.append(d["UV"].astype(np.float32))
        sim_vorts.append(d["omega"].astype(np.float32))
        print(f"    {fpath.name}: UV={d['UV'].shape}")
    gt_data = np.load(data_dir / "gt.npz")
    gt_grid = gt_data["UV"].astype(np.float32)
    gt_vort = gt_data["omega"].astype(np.float32)
    print(f"    gt.npz: UV={gt_grid.shape}")
    return sim_grids, sim_vorts, gt_grid, gt_vort


# Main

def main():
    parser = argparse.ArgumentParser(description="LAPIS-2DKF")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--shred_mode", type=str, default=None, choices=["frame", "seq2seq"])
    parser.add_argument("--obs_fraction", type=float, default=None)
    args = parser.parse_args()

    if args.base_dir is not None:
        config = Config.initialize(Path(args.base_dir))
    else:
        config = Config.initialize()
    if args.shred_mode is not None:
        config.SHRED_MODE = args.shred_mode
    if args.obs_fraction is not None:
        config.OBS_FRACTION = args.obs_fraction

    print(f"\n  LAPIS-2DKF: Backward Reconstruction\n")
    rng = jax.random.PRNGKey(config.SEED)

    # [1] Load
    print("[1] Loading data ...")
    sim_grids, sim_vorts, gt_grid, gt_vort = load_data(config)
    T_total = gt_grid.shape[0]
    n_ch, Nx, Ny = gt_grid.shape[1], gt_grid.shape[2], gt_grid.shape[3]
    state_dim = n_ch * Nx * Ny
    obs_len = max(config.LAGS + 1, int(T_total * config.OBS_FRACTION))
    print(f"  T={T_total}, {n_ch}x{Nx}x{Ny}, state_dim={state_dim}, obs_len={obs_len}")

    # [2] Sensors on vorticity variance
    print("\n[2] Sensors & dataset ...")
    sensors = place_sensors(sim_vorts, Nx, Ny, config.N_SENSORS,
                            strategy=config.SENSOR_STRATEGY, seed=config.SEED)
    active_mask = np.ones((n_ch * Nx, Ny), dtype=bool)
    use_frame = config.SHRED_MODE.lower() == "frame"

    if use_frame:
        dataset = EnsembleFrameDataset(
            sim_grids, sensors, lags=config.LAGS, fit=True,
            sensor_extract_fn=vorticity_sensor_extract)
        print(f"  {len(dataset)} frame samples, mode=frame")

        print("\n[3] Training Frame SHRED ...")
        shred_model = FrameSHRED(
            n_sensors=config.N_SENSORS, hidden_size=config.SEQ2SEQ_HIDDEN,
            num_layers=config.NUM_LAYERS, decoder_layers=config.DECODER_LAYERS,
            state_dim=state_dim, dropout_rate=config.DROPOUT_RATE)
        rng, srng = jax.random.split(rng)
        shred_state = train_ensemble_frame_shred(shred_model, dataset, active_mask, config, srng)

        latent_trajectories, z_inits = extract_latent_trajectories_frame(
            shred_state, dataset, sim_grids, sensors, config,
            sensor_extract_fn=vorticity_sensor_extract)
        T_originals = [t.shape[0] for t in latent_trajectories]
        latent_dim = latent_trajectories[0].shape[1]
        obs_len_latent = max(2, obs_len - config.LAGS)
    else:
        dataset = EnsembleSeq2SeqDataset(
            sim_grids, sensors, initial_pad=config.TERMINAL_PAD, fit=True,
            sensor_extract_fn=vorticity_sensor_extract)
        print(f"  {len(dataset)} sequences, mode=seq2seq")

        print("\n[3] Training Seq2Seq SHRED ...")
        latent_dim = config.SEQ2SEQ_HIDDEN * 2
        shred_model = Seq2SeqSHRED(
            n_sensors=config.N_SENSORS, hidden_size=config.SEQ2SEQ_HIDDEN,
            num_layers=config.NUM_LAYERS, state_dim=state_dim,
            dropout_rate=config.DROPOUT_RATE)
        rng, srng = jax.random.split(rng)
        shred_state = train_ensemble_shred(shred_model, dataset, active_mask, config, srng)

        latent_trajectories, z_inits = extract_latent_trajectories_seq2seq(shred_state, dataset, config)
        T_originals = dataset.T_originals
        obs_len_latent = obs_len

    # [4] Backward model
    print("\n[4] Training Backward Model ...")
    backward_model = BackwardFromWindow(
        latent_dim=latent_dim, hidden_dim=config.BACKWARD_HIDDEN,
        num_layers=config.BACKWARD_LAYERS, dropout_rate=config.DROPOUT_RATE)
    rng, brng = jax.random.split(rng)
    backward_state = train_backward_model(
        backward_model, latent_trajectories, T_originals, obs_len_latent, config, brng)

    # [5] Inference
    print(f"\n[5] LAPIS Backward Inference ...")
    if use_frame:
        pred_lapis = lapis_backward_inference_frame(
            shred_state, backward_state, gt_grid, sensors, dataset, obs_len_latent, config,
            sensor_extract_fn=vorticity_sensor_extract)
    else:
        pred_lapis = lapis_backward_inference_seq2seq(
            shred_state, backward_state, gt_grid, sensors, dataset, obs_len, config,
            sensor_extract_fn=vorticity_sensor_extract)

    # [6] Baseline
    print("\n[6] SHRED baseline ...")
    if use_frame:
        pred_shred = shred_baseline_frame(
            shred_state, gt_grid, sensors, dataset, config,
            sensor_extract_fn=vorticity_sensor_extract)
    else:
        pred_shred = shred_baseline_seq2seq(
            shred_state, gt_grid, sensors, dataset, config,
            sensor_extract_fn=vorticity_sensor_extract)

    # [7] Evaluate (velocity speed + vorticity)
    print("\n[7] Evaluation ...")
    active_spatial = np.ones((Nx, Ny), dtype=bool)

    def _speed(uv): return np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
    gt_speed, l_speed, s_speed = _speed(gt_grid), _speed(pred_lapis), _speed(pred_shred)
    m_l_vel = compute_metrics(l_speed, gt_speed, active_spatial)
    m_s_vel = compute_metrics(s_speed, gt_speed, active_spatial)
    vel_range = float(gt_speed.max() - gt_speed.min())

    gt_v = compute_vorticity_from_uv(gt_grid)
    l_v = compute_vorticity_from_uv(pred_lapis)
    s_v = compute_vorticity_from_uv(pred_shred)
    m_l_vort = compute_metrics(l_v, gt_v, active_spatial)
    m_s_vort = compute_metrics(s_v, gt_v, active_spatial)
    vort_range = float(gt_v.max() - gt_v.min())

    print(f"  |u|: LAPIS RMSE={m_l_vel['rmse_active']:.4f} NRMSE={m_l_vel['rmse_active']/vel_range:.4f}")
    print(f"  |u|: SHRED RMSE={m_s_vel['rmse_active']:.4f} NRMSE={m_s_vel['rmse_active']/vel_range:.4f}")
    print(f"   w:  LAPIS RMSE={m_l_vort['rmse_active']:.4f} NRMSE={m_l_vort['rmse_active']/vort_range:.4f}")
    print(f"   w:  SHRED RMSE={m_s_vort['rmse_active']:.4f} NRMSE={m_s_vort['rmse_active']/vort_range:.4f}")

    # [8] Save
    print("\n[8] Saving ...")
    res = config.RESULTS_DIR
    save_velocity_and_vorticity(gt_grid, pred_lapis, pred_shred, res,
                                sensor_locs=sensors,
                                compute_vorticity_fn=compute_vorticity_from_uv)
    save_timeseries(gt_v, l_v, s_v, sensors, obs_len,
                    res / "timeseries_vorticity.png",
                    ylabel_fmt="w({r},{c})", title="LAPIS-2DKF: Vorticity", obs_side="end")
    np.save(res / "pred_lapis.npy", pred_lapis.astype(np.float32))
    np.save(res / "pred_shred.npy", pred_shred.astype(np.float32))

    results_json = to_json_safe({
        "T_total": T_total, "obs_len": obs_len,
        "n_sensors": config.N_SENSORS, "shred_mode": config.SHRED_MODE,
        "velocity_speed": {
            "gt_range": vel_range,
            "lapis": {"rmse": m_l_vel['rmse_active'], "ssim": m_l_vel['ssim']},
            "shred": {"rmse": m_s_vel['rmse_active'], "ssim": m_s_vel['ssim']},
        },
        "vorticity": {
            "gt_range": vort_range,
            "lapis": {"rmse": m_l_vort['rmse_active'], "ssim": m_l_vort['ssim']},
            "shred": {"rmse": m_s_vort['rmse_active'], "ssim": m_s_vort['ssim']},
        },
        "timestamp": datetime.now().isoformat(),
    })
    with open(res / "lapis_2dkf_metrics.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  LAPIS-2DKF complete! Results: {res}")


if __name__ == "__main__":
    main()
