#!/usr/bin/env python3
"""
data_generation_2dks.py — 2D Kuramoto–Sivashinsky ensemble data generation for LAPIS

Generates:
  - K simulation trajectories from perturbed initial conditions
  - 1 ground truth trajectory from an IC within the convex hull of the sim ICs
  - Saves everything to ../data/

The 2D KS equation (undamped):
    u_t + (1/2)(u_x^2 + u_y^2) + ∇²u + ∇⁴u = 0
on a doubly-periodic domain [0, Lx) x [0, Ly).

Unlike dissipative LAPIS experiments (which has dissipative terminal states),
the 2DKS is chaotic/spatiotemporally complex.  LAPIS-2DKS therefore uses a
short observed time window at the end (not a single static frame) and
backward-reconstructs the preceding trajectory.

Directory layout:
    LAPIS_2DKS/
    ├── code/          ← this script + lapis_2dks.py + shred_jax/
    ├── data/          ← generated .npz files
    └── results/

Usage:
    cd LAPIS_2DKS/code
    python data_generation_2dks.py
"""

import numpy as np
from scipy.fft import fft2, ifft2
import os
from pathlib import Path

# 2DKS SOLVER  (same ETDRK4 as generate_data.py)

class KuramotoSivashinsky2D:
    """ETDRK4 solver for the undamped 2D Kuramoto–Sivashinsky equation."""

    def __init__(self, Lx, Ly, Nx, Ny, dt):
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.dt = dt

        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        self.y = np.linspace(0, Ly, Ny, endpoint=False)

        dx, dy = Lx / Nx, Ly / Ny

        kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')

        K2 = self.KX**2 + self.KY**2
        K4 = K2**2

        self.L = K2 - K4

        kx_max = np.max(np.abs(kx)) * 2 / 3
        ky_max = np.max(np.abs(ky)) * 2 / 3
        self.dealias = (np.abs(self.KX) < kx_max) & (np.abs(self.KY) < ky_max)

        self._setup_etdrk4()

    def _setup_etdrk4(self):
        h, L = self.dt, self.L
        self.E  = np.exp(h * L)
        self.E2 = np.exp(h * L / 2)

        M = 32
        theta = np.pi * (np.arange(1, M + 1) - 0.5) / M
        r = np.exp(1j * theta)
        LR = h * L[:, :, None] + r[None, None, :]

        self.Q  = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=2))
        self.f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=2))
        self.f2 = h * np.real(np.mean(( 2 + LR + np.exp(LR) * (-2 + LR))          / LR**3, axis=2))
        self.f3 = h * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=2))

    def _nonlinear(self, u_hat):
        ux = np.real(ifft2(1j * self.KX * u_hat))
        uy = np.real(ifft2(1j * self.KY * u_hat))
        N_hat = fft2(-0.5 * (ux**2 + uy**2))
        return N_hat * self.dealias

    def step(self, u_hat):
        Nu = self._nonlinear(u_hat)
        a  = self.E2 * u_hat + self.Q * Nu
        Na = self._nonlinear(a)
        b  = self.E2 * u_hat + self.Q * Na
        Nb = self._nonlinear(b)
        c  = self.E2 * a + self.Q * (2 * Nb - Nu)
        Nc = self._nonlinear(c)
        return self.E * u_hat + self.f1*Nu + 2*self.f2*(Na + Nb) + self.f3*Nc

    def simulate(self, u0, T, save_every=1):
        n_steps = int(T / self.dt)
        n_save  = n_steps // save_every + 1

        U = np.zeros((n_save, self.Nx, self.Ny))
        u_hat = fft2(u0)
        U[0] = u0
        save_idx = 1

        for s in range(1, n_steps + 1):
            u_hat = self.step(u_hat)
            if np.any(np.isnan(u_hat)) or np.max(np.abs(u_hat)) > 1e10:
                print(f"  WARNING: blowup at step {s}")
                return U[:save_idx]
            if s % save_every == 0 and save_idx < n_save:
                U[save_idx] = np.real(ifft2(u_hat))
                save_idx += 1

        print(f"  Simulation complete: {save_idx} snapshots, "
              f"u in [{U.min():.3f}, {U.max():.3f}]")
        return U


# INITIAL CONDITION GENERATION

def make_base_ic(x, y, Lx, Ly):
    """
    Base initial condition: a smooth superposition of low-wavenumber modes.
    Returns (Nx, Ny) array normalized to max amplitude 1.
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    u0 = (np.cos(2 * np.pi * X / Lx) * np.cos(2 * np.pi * Y / Ly)
          + 0.5 * np.sin(4 * np.pi * X / Lx) * np.sin(2 * np.pi * Y / Ly))
    u0 /= np.abs(u0).max()
    return u0


def perturb_ic(u0_base, x, y, Lx, Ly, rng, amplitude=0.15, n_modes=4):
    """
    Add a random perturbation to the base IC using random Fourier modes.

    Parameters -
    u0_base : (Nx, Ny)  base initial condition
    amplitude : float    perturbation amplitude relative to base max
    n_modes : int        number of random Fourier modes to superpose
    """
    Nx, Ny = len(x), len(y)
    X, Y = np.meshgrid(x, y, indexing='ij')

    perturbation = np.zeros((Nx, Ny))
    for _ in range(n_modes):
        # Random low wavenumbers (1–3 cycles across domain)
        kx_n = rng.integers(1, 4)
        ky_n = rng.integers(1, 4)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        coeff = rng.uniform(-1.0, 1.0)
        perturbation += coeff * np.sin(2 * np.pi * kx_n * X / Lx + phase_x) \
                              * np.sin(2 * np.pi * ky_n * Y / Ly + phase_y)

    # Normalize perturbation
    pmax = np.abs(perturbation).max()
    if pmax > 0:
        perturbation = perturbation / pmax * amplitude

    u0 = u0_base + perturbation
    # Re-normalize so max amplitude ~ 1
    u0 /= np.abs(u0).max()
    return u0


# MAIN

if __name__ == "__main__":
    #  Physics & grid parameters  
    Lx, Ly     = 16 * np.pi, 16 * np.pi    # domain size
    Nx, Ny     = 64, 64                     # spatial resolution
    dt         = 0.05                       # time step
    T          = 25.0                       # total simulation time
    save_every = 5                          # save every 5 steps → dt_save = 0.25

    #  Ensemble parameters  
    K_sim           = 8       # number of simulation (training) trajectories
    perturb_amp     = 0.15    # perturbation amplitude for sim ICs (e.g. ±15%)
    gt_perturb_amp  = 0.08    # smaller perturbation for GT (within sim hull, e.g. 0.08)
    seed            = 42

    #  Paths  
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIR   = SCRIPT_DIR.parent / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    #  Solver  
    solver = KuramotoSivashinsky2D(Lx, Ly, Nx, Ny, dt)
    x, y = solver.x, solver.y

    dt_save = dt * save_every
    print("=" * 60)
    print("  2DKS Ensemble Data Generation for LAPIS")
    print("=" * 60)
    print(f"  Domain: [{Lx/np.pi:.0f}π × {Ly/np.pi:.0f}π], grid {Nx}×{Ny}")
    print(f"  dt={dt}, T={T}, save_every={save_every} → dt_save={dt_save}")
    print(f"  K_sim={K_sim}, perturb_amp={perturb_amp}, gt_perturb_amp={gt_perturb_amp}")
    print()

    #  Base IC  
    u0_base = make_base_ic(x, y, Lx, Ly)

    rng = np.random.default_rng(seed)

    #  Simulation ensemble  
    sim_trajectories = []
    sim_ics = []
    for k in range(K_sim):
        print(f"  Simulation {k+1}/{K_sim} ...")
        u0_k = perturb_ic(u0_base, x, y, Lx, Ly, rng, amplitude=perturb_amp)
        U_k = solver.simulate(u0_k, T, save_every)
        sim_trajectories.append(U_k)
        sim_ics.append(u0_k)

    #  Ground truth IC: convex combination of sim ICs + small perturbation  
    # Take a random convex combination of a subset of sim ICs
    print(f"\n  Ground truth IC (convex hull + small perturbation) ...")
    n_mix = min(4, K_sim)
    mix_idx = rng.choice(K_sim, size=n_mix, replace=False)
    weights = rng.dirichlet(np.ones(n_mix))
    u0_gt_mix = sum(w * sim_ics[i] for w, i in zip(weights, mix_idx))
    # Add a small additional perturbation
    u0_gt = perturb_ic(u0_gt_mix, x, y, Lx, Ly, rng, amplitude=gt_perturb_amp, n_modes=3)

    print(f"    Mixed ICs: indices={mix_idx.tolist()}, weights={[f'{w:.3f}' for w in weights]}")
    U_gt = solver.simulate(u0_gt, T, save_every)

    #  Save  
    t_vec = np.arange(U_gt.shape[0]) * dt_save

    # Save simulation ensemble
    for k in range(K_sim):
        fpath = DATA_DIR / f"sim_{k:02d}.npz"
        np.savez(fpath, U=sim_trajectories[k], u0=sim_ics[k],
                 x=x, y=y, t=t_vec[:sim_trajectories[k].shape[0]],
                 Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dt_save=dt_save)
        print(f"    Saved sim {k}: {sim_trajectories[k].shape} → {fpath}")

    # Save ground truth
    gt_path = DATA_DIR / "gt.npz"
    np.savez(gt_path, U=U_gt, u0=u0_gt,
             x=x, y=y, t=t_vec[:U_gt.shape[0]],
             Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dt_save=dt_save,
             mix_indices=mix_idx, mix_weights=weights)
    print(f"    Saved GT: {U_gt.shape} → {gt_path}")

    # Save metadata
    meta = {
        "K_sim": K_sim,
        "Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny,
        "dt": dt, "T": T, "save_every": save_every, "dt_save": dt_save,
        "perturb_amp": perturb_amp, "gt_perturb_amp": gt_perturb_amp,
        "seed": seed,
        "n_snapshots_per_sim": [s.shape[0] for s in sim_trajectories],
        "n_snapshots_gt": U_gt.shape[0],
    }
    import json
    meta_path = DATA_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"    Saved metadata → {meta_path}")

    #  Quick preview figure  
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(K_sim + 1, 5, figsize=(15, 3 * (K_sim + 1)))
        if K_sim + 1 == 1:
            axes = axes[None, :]

        # Determine shared time indices from GT (all trajectories have same length)
        T_gt = U_gt.shape[0]
        col_idxs = np.linspace(0, T_gt - 1, 5, dtype=int)

        # Compute per-column vmin/vmax across all rows (sims + GT)
        col_vmin = np.zeros(5)
        col_vmax = np.zeros(5)
        for col, ti in enumerate(col_idxs):
            all_vals = [U_gt[ti]]
            for row in range(K_sim):
                U_k = sim_trajectories[row]
                if ti < U_k.shape[0]:
                    all_vals.append(U_k[ti])
            col_vmin[col] = min(v.min() for v in all_vals)
            col_vmax[col] = max(v.max() for v in all_vals)
            # Make symmetric around zero for RdBu_r
            vabs = max(abs(col_vmin[col]), abs(col_vmax[col]))
            col_vmin[col], col_vmax[col] = -vabs, vabs

        for row in range(K_sim):
            U_k = sim_trajectories[row]
            for col, ti in enumerate(col_idxs):
                ax = axes[row, col]
                if ti < U_k.shape[0]:
                    im = ax.imshow(U_k[ti], cmap="RdBu_r",
                                   vmin=col_vmin[col], vmax=col_vmax[col])
                ax.set_title(f"t={t_vec[ti]:.1f}", fontsize=7)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(f"Sim {row}", fontsize=8)

        # GT row
        for col, ti in enumerate(col_idxs):
            ax = axes[K_sim, col]
            im = ax.imshow(U_gt[ti], cmap="RdBu_r",
                           vmin=col_vmin[col], vmax=col_vmax[col])
            ax.set_title(f"t={t_vec[ti]:.1f}", fontsize=7)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel("GT", fontsize=8, color="red")

        plt.suptitle("2DKS Ensemble: Simulation ICs + Ground Truth", fontsize=12)
        plt.tight_layout()
        fig_path = DATA_DIR / "data_preview.png"
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"    Saved preview → {fig_path}")
    except Exception as e:
        print(f"    (preview figure skipped: {e})")

    print("\n  Done!")
