#!/usr/bin/env python3
"""
data_generation_2dkf.py — 2D Kolmogorov Flow ensemble data generation for LAPIS

Generates:
  - K simulation trajectories from perturbed initial conditions
  - 1 ground truth trajectory from an IC within the convex hull of the sim ICs
  - Saves everything to ../data/

The 2D Kolmogorov flow (incompressible Navier-Stokes with sinusoidal forcing):
    ω_t + u · ∇ω = (1/Re) ∇²ω + f_ω
    u = ∇⊥ψ,   ∇²ψ = -ω
on a doubly-periodic domain [0, 2π) × [0, 2π).

We solve in vorticity-streamfunction form using a pseudo-spectral method
with ETDRK4 time integration.

Reference dataset: Kolmogorov flow at Re=500 with sinusoidal forcing
    f = [0, sin(k0 * y)] + 0.1,  k0 = 4
as described in:
  - Kochkov et al. (2021), "ML-accelerated CFD"
  - "Spectral Shaping for Neural PDE Surrogates" (ICLR 2025 submission)

Directory layout:
    LAPIS-SHRED/
    ├── data/          ← this script + generated .npz files
    ├── model/         ← lapis_2dkf.py + shred_jax/
    └── results/

Usage:
    cd LAPIS-SHRED/data
    python data_generation_2dkf.py
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
import os
from pathlib import Path

# 2D KOLMOGOROV FLOW SOLVER (vorticity-streamfunction, pseudo-spectral ETDRK4)

class KolmogorovFlow2D:
    """
    Pseudo-spectral ETDRK4 solver for 2D Kolmogorov flow in vorticity form.

    Vorticity equation:
        ω_t = -(u · ∇)ω + (1/Re) ∇²ω + curl(f)

    where u = ∇⊥ψ = (-ψ_y, ψ_x), ∇²ψ = -ω,
    and forcing f = [0, sin(k0*y)] + 0.1 → curl(f) = k0 * cos(k0*y).
    """

    def __init__(self, Nx, Ny, Re, dt, k0=4, f_offset=0.1):
        self.Nx, self.Ny = Nx, Ny
        self.Re = Re
        self.dt = dt
        self.k0 = k0
        self.f_offset = f_offset

        self.Lx, self.Ly = 2 * np.pi, 2 * np.pi
        self.x = np.linspace(0, self.Lx, Nx, endpoint=False)
        self.y = np.linspace(0, self.Ly, Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        dx, dy = self.Lx / Nx, self.Ly / Ny

        # Wavenumbers
        kx = fftfreq(Nx, d=dx) * 2 * np.pi
        ky = fftfreq(Ny, d=dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')

        self.K2 = self.KX**2 + self.KY**2
        # Avoid division by zero at k=0
        self.K2_inv = np.zeros_like(self.K2)
        self.K2_inv[self.K2 > 0] = 1.0 / self.K2[self.K2 > 0]

        # Linear operator: L = -(1/Re) * k^2  (viscous diffusion of vorticity)
        self.L = -(1.0 / Re) * self.K2

        # Dealiasing mask (2/3 rule)
        kx_max = np.max(np.abs(kx)) * 2 / 3
        ky_max = np.max(np.abs(ky)) * 2 / 3
        self.dealias = (np.abs(self.KX) < kx_max) & (np.abs(self.KY) < ky_max)

        # Forcing: curl of f = [0, sin(k0*y)] is k0*cos(k0*y)
        # The offset 0.1 in f is constant, so its curl is zero.
        self.forcing = k0 * np.cos(k0 * self.Y)
        self.forcing_hat = fft2(self.forcing) * self.dealias

        self._setup_etdrk4()

    def _setup_etdrk4(self):
        """Compute ETDRK4 coefficients via contour integral method."""
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

    def _omega_to_velocity(self, omega_hat):
        """Recover velocity from vorticity: ∇²ψ = -ω, u = (-ψ_y, ψ_x)."""
        psi_hat = -omega_hat * self.K2_inv
        u_hat = -1j * self.KY * psi_hat   # u = -ψ_y
        v_hat =  1j * self.KX * psi_hat   # v =  ψ_x
        return u_hat, v_hat

    def _nonlinear(self, omega_hat):
        """Nonlinear term: N(ω) = -(u·∇)ω + forcing."""
        u_hat, v_hat = self._omega_to_velocity(omega_hat)

        u  = np.real(ifft2(u_hat * self.dealias))
        v  = np.real(ifft2(v_hat * self.dealias))
        ox = np.real(ifft2(1j * self.KX * omega_hat * self.dealias))
        oy = np.real(ifft2(1j * self.KY * omega_hat * self.dealias))

        # Advection in physical space, then transform back
        advection = u * ox + v * oy
        N_hat = -fft2(advection) * self.dealias + self.forcing_hat
        return N_hat

    def step(self, omega_hat):
        """Single ETDRK4 step."""
        Nu = self._nonlinear(omega_hat)
        a  = self.E2 * omega_hat + self.Q * Nu
        Na = self._nonlinear(a)
        b  = self.E2 * omega_hat + self.Q * Na
        Nb = self._nonlinear(b)
        c  = self.E2 * a + self.Q * (2 * Nb - Nu)
        Nc = self._nonlinear(c)
        return self.E * omega_hat + self.f1*Nu + 2*self.f2*(Na + Nb) + self.f3*Nc

    def simulate(self, omega0, T, save_every=1):
        """
        Run simulation.

        Parameters - 
        omega0 : (Nx, Ny) initial vorticity field
        T : float, total simulation time
        save_every : int, save every N steps

        Returns - 
        Omega : (n_save, Nx, Ny) saved vorticity snapshots
        U : (n_save, Nx, Ny) saved u-velocity snapshots
        V : (n_save, Nx, Ny) saved v-velocity snapshots
        """
        n_steps = int(T / self.dt)
        n_save  = n_steps // save_every + 1

        Omega = np.zeros((n_save, self.Nx, self.Ny))
        U_vel = np.zeros((n_save, self.Nx, self.Ny))
        V_vel = np.zeros((n_save, self.Nx, self.Ny))

        omega_hat = fft2(omega0)

        # Save initial
        Omega[0] = omega0
        u_hat, v_hat = self._omega_to_velocity(omega_hat)
        U_vel[0] = np.real(ifft2(u_hat))
        V_vel[0] = np.real(ifft2(v_hat))
        save_idx = 1

        for s in range(1, n_steps + 1):
            omega_hat = self.step(omega_hat)
            if np.any(np.isnan(omega_hat)) or np.max(np.abs(omega_hat)) > 1e12:
                print(f"  WARNING: blowup at step {s}")
                return Omega[:save_idx], U_vel[:save_idx], V_vel[:save_idx]
            if s % save_every == 0 and save_idx < n_save:
                Omega[save_idx] = np.real(ifft2(omega_hat))
                u_hat, v_hat = self._omega_to_velocity(omega_hat)
                U_vel[save_idx] = np.real(ifft2(u_hat))
                V_vel[save_idx] = np.real(ifft2(v_hat))
                save_idx += 1

        print(f"  Simulation complete: {save_idx} snapshots, "
              f"ω in [{Omega[:save_idx].min():.3f}, {Omega[:save_idx].max():.3f}]")
        return Omega[:save_idx], U_vel[:save_idx], V_vel[:save_idx]


# INITIAL CONDITION GENERATION

def make_base_ic(x, y, Lx, Ly):
    """
    Base initial vorticity: a smooth superposition of low-wavenumber modes,
    mimicking a spun-up Kolmogorov flow state.
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    omega0 = (2.0 * np.sin(4 * Y)
              + 0.5 * np.cos(2 * X) * np.sin(2 * Y)
              + 0.3 * np.sin(3 * X) * np.cos(3 * Y))
    return omega0


def perturb_ic(omega_base, x, y, Lx, Ly, rng, amplitude=0.15, n_modes=6):
    """
    Add a random perturbation to the base IC using random Fourier modes.
    """
    Nx, Ny = len(x), len(y)
    X, Y = np.meshgrid(x, y, indexing='ij')

    perturbation = np.zeros((Nx, Ny))
    for _ in range(n_modes):
        kx_n = rng.integers(1, 6)
        ky_n = rng.integers(1, 6)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        coeff = rng.uniform(-1.0, 1.0)
        perturbation += coeff * np.sin(kx_n * X + phase_x) \
                              * np.sin(ky_n * Y + phase_y)

    pmax = np.abs(perturbation).max()
    if pmax > 0:
        base_scale = np.abs(omega_base).max()
        perturbation = perturbation / pmax * amplitude * base_scale

    return omega_base + perturbation


# MAIN

if __name__ == "__main__":
    # Physics & grid parameters
    Nx, Ny     = 64, 64                     # spatial resolution, e.g. 64, 64
    Re         = 50                         # Reynolds number, e.g. 500
    k0         = 4                          # forcing wavenumber
    dt         = 0.01                       # time step (smaller than KS due to advection CFL)
    T_burnin   = 10.0                       # burn-in time to reach statistically stationary state
    T_sim      = 10.0                       # simulation time after burn-in
    save_every = 10                         # save every 10 steps → dt_save = 0.1

    # Ensemble parameters
    K_sim           = 15      # number of simulation (training) trajectories
    perturb_amp     = 0.05    # perturbation amplitude for sim ICs (±15%)
    gt_perturb_amp  = 0.02    # smaller perturbation for GT (within sim hull)
    seed            = 42

    # Paths
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIR   = SCRIPT_DIR.parent / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Solver
    solver = KolmogorovFlow2D(Nx, Ny, Re, dt, k0=k0)
    x, y = solver.x, solver.y
    Lx, Ly = solver.Lx, solver.Ly

    dt_save = dt * save_every
    print("=" * 60)
    print("  2DKF Ensemble Data Generation for LAPIS")
    print("=" * 60)
    print(f"  Domain: [0, 2π) × [0, 2π), grid {Nx}×{Ny}")
    print(f"  Re={Re}, k0={k0}, dt={dt}, T_burnin={T_burnin}, T_sim={T_sim}")
    print(f"  save_every={save_every} → dt_save={dt_save}")
    print(f"  K_sim={K_sim}, perturb_amp={perturb_amp}, gt_perturb_amp={gt_perturb_amp}")
    print()

    # Base IC & burn-in
    omega0_raw = make_base_ic(x, y, Lx, Ly)

    print("  Burn-in to reach statistically stationary state ...")
    # Use a large save_every during burn-in since we only need the final state
    burnin_steps = int(T_burnin / dt)
    burnin_save = burnin_steps  # save only start and end
    Omega_burn, _, _ = solver.simulate(omega0_raw, T_burnin, save_every=burnin_save)
    omega0_base = Omega_burn[-1]
    print(f"    Burn-in complete. Base vorticity in [{omega0_base.min():.3f}, {omega0_base.max():.3f}]")

    rng = np.random.default_rng(seed)

    # Simulation ensemble
    sim_trajectories_omega = []
    sim_trajectories_u = []
    sim_trajectories_v = []
    sim_ics = []
    for k in range(K_sim):
        print(f"\n  Simulation {k+1}/{K_sim} ...")
        omega0_k = perturb_ic(omega0_base, x, y, Lx, Ly, rng, amplitude=perturb_amp)
        Omega_k, U_k, V_k = solver.simulate(omega0_k, T_sim, save_every)
        sim_trajectories_omega.append(Omega_k)
        sim_trajectories_u.append(U_k)
        sim_trajectories_v.append(V_k)
        sim_ics.append(omega0_k)

    # Ground truth IC: convex combination of sim ICs + small perturbation
    print(f"\n  Ground truth IC (convex hull + small perturbation) ...")
    n_mix = min(4, K_sim)
    mix_idx = rng.choice(K_sim, size=n_mix, replace=False)
    weights = rng.dirichlet(np.ones(n_mix))
    omega0_gt_mix = sum(w * sim_ics[i] for w, i in zip(weights, mix_idx))
    omega0_gt = perturb_ic(omega0_gt_mix, x, y, Lx, Ly, rng,
                           amplitude=gt_perturb_amp, n_modes=4)

    print(f"    Mixed ICs: indices={mix_idx.tolist()}, weights={[f'{w:.3f}' for w in weights]}")
    Omega_gt, U_gt, V_gt = solver.simulate(omega0_gt, T_sim, save_every)

    # Save
    t_vec = np.arange(Omega_gt.shape[0]) * dt_save

    # We store velocity as a 2-channel field: U_field = (u, v) stacked as (T, 2, Nx, Ny)
    # Also store vorticity separately for visualization
    for k in range(K_sim):
        T_k = sim_trajectories_omega[k].shape[0]
        UV_k = np.stack([sim_trajectories_u[k], sim_trajectories_v[k]], axis=1)  # (T, 2, Nx, Ny)
        fpath = DATA_DIR / f"sim_{k:02d}.npz"
        np.savez(fpath,
                 UV=UV_k,
                 omega=sim_trajectories_omega[k],
                 omega0=sim_ics[k],
                 x=x, y=y, t=t_vec[:T_k],
                 Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dt_save=dt_save, Re=Re)
        print(f"    Saved sim {k}: UV={UV_k.shape}, omega={sim_trajectories_omega[k].shape} → {fpath}")

    # Save ground truth
    UV_gt = np.stack([U_gt, V_gt], axis=1)  # (T, 2, Nx, Ny)
    gt_path = DATA_DIR / "gt.npz"
    np.savez(gt_path,
             UV=UV_gt,
             omega=Omega_gt,
             omega0=omega0_gt,
             x=x, y=y, t=t_vec[:Omega_gt.shape[0]],
             Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dt_save=dt_save, Re=Re,
             mix_indices=mix_idx, mix_weights=weights)
    print(f"    Saved GT: UV={UV_gt.shape}, omega={Omega_gt.shape} → {gt_path}")

    # Save metadata
    meta = {
        "K_sim": K_sim,
        "Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny,
        "Re": Re, "k0": k0,
        "dt": dt, "T_burnin": T_burnin, "T_sim": T_sim,
        "save_every": save_every, "dt_save": dt_save,
        "perturb_amp": perturb_amp, "gt_perturb_amp": gt_perturb_amp,
        "seed": seed,
        "n_snapshots_per_sim": [s.shape[0] for s in sim_trajectories_omega],
        "n_snapshots_gt": Omega_gt.shape[0],
        "fields": "UV = (u, v) velocity; omega = vorticity",
    }
    import json
    meta_path = DATA_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"    Saved metadata → {meta_path}")

    # Quick preview figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(K_sim + 1, 5, figsize=(15, 3 * (K_sim + 1)))
        if K_sim + 1 == 1:
            axes = axes[None, :]

        # Use vorticity for visualization
        T_gt_len = Omega_gt.shape[0]
        col_idxs = np.linspace(0, T_gt_len - 1, 5, dtype=int)

        # Per-column symmetric color range
        col_vlim = np.zeros(5)
        for col, ti in enumerate(col_idxs):
            all_vals = [Omega_gt[ti]]
            for row in range(K_sim):
                Om_k = sim_trajectories_omega[row]
                if ti < Om_k.shape[0]:
                    all_vals.append(Om_k[ti])
            col_vlim[col] = max(np.abs(v).max() for v in all_vals)

        for row in range(K_sim):
            Om_k = sim_trajectories_omega[row]
            for col, ti in enumerate(col_idxs):
                ax = axes[row, col]
                if ti < Om_k.shape[0]:
                    ax.imshow(Om_k[ti].T, cmap="RdBu_r", origin="lower",
                              vmin=-col_vlim[col], vmax=col_vlim[col])
                ax.set_title(f"t={t_vec[ti]:.1f}", fontsize=7)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(f"Sim {row}", fontsize=8)

        # GT row
        for col, ti in enumerate(col_idxs):
            ax = axes[K_sim, col]
            ax.imshow(Omega_gt[ti].T, cmap="RdBu_r", origin="lower",
                      vmin=-col_vlim[col], vmax=col_vlim[col])
            ax.set_title(f"t={t_vec[ti]:.1f}", fontsize=7)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel("GT", fontsize=8, color="red")

        plt.suptitle(f"2DKF Ensemble (Re={Re}): Vorticity — Sims + Ground Truth", fontsize=12)
        plt.tight_layout()
        fig_path = DATA_DIR / "data_preview.png"
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"    Saved preview → {fig_path}")
    except Exception as e:
        print(f"    (preview figure skipped: {e})")

    print("\n  Done!")
