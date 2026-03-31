#!/usr/bin/env python3
"""
data_generation_2dkvs.py — 2D von Kármán Vortex Street data generation for LAPIS
==================================================================================

Generates ensemble data for LAPIS backward reconstruction on flow past a
circular cylinder (von Kármán vortex street).

Solver: D2Q9 Lattice Boltzmann Method (LBM), vectorised streaming.

Ensemble strategy: **Phase-shifted branching**
  After burn-in, we continue running and save the LBM state at regular
  intervals (spaced ~1/3 shedding period apart). Each sim starts from a
  different saved state, so all trajectories show the same vortex street
  but at different shedding phases. This produces meaningful diversity
  on a low-dimensional manifold — ideal for LAPIS.

Directory layout:
    LAPIS_2DKVS/
    ├── code/          ← this script + lapis_2dkvs.py + shred_jax/
    ├── data/          ← generated .npz files
    └── results/

Usage:
    cd LAPIS_2DKVS/code
    python data_generation_2dkvs.py
"""

import numpy as np
import os, time
from pathlib import Path
import json


# Utilities
def downsample_blockavg(frame2d: np.ndarray, ds_x: int, ds_y: int) -> np.ndarray:
    """Downsample a 2D field (Nx, Ny) by block-averaging."""
    if frame2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {frame2d.shape}")
    Nx, Ny = frame2d.shape
    out_Nx = Nx // ds_x
    out_Ny = Ny // ds_y
    f = frame2d[:out_Nx * ds_x, :out_Ny * ds_y]
    f = f.reshape(out_Nx, ds_x, out_Ny, ds_y).mean(axis=(1, 3))
    return f.astype(np.float32)


# D2Q9 LATTICE BOLTZMANN SOLVER  (vectorised streaming)

class LatticeBoltzmann2D:
    """
    D2Q9 LBM with pre-computed index arrays for vectorised streaming.

    Boundary conditions:
        - Left:   Zou-He velocity inlet (uniform U_inf)
        - Right:  zero-gradient outlet
        - Top/Bottom: periodic
        - Cylinder: halfway bounce-back (no-slip)
    """

    # D2Q9 lattice constants
    e = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
    w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    def __init__(self, Nx, Ny, U_inf, Re, cyl_cx, cyl_cy, cyl_r):
        self.Nx, self.Ny = Nx, Ny
        self.U_inf = U_inf
        self.Re = Re
        self.D = 2 * cyl_r

        self.nu = U_inf * self.D / Re
        self.tau = 3.0 * self.nu + 0.5
        self.omega_lbm = 1.0 / self.tau

        print(f"    LBM: nu={self.nu:.6f}, tau={self.tau:.4f}, omega={self.omega_lbm:.4f}")

        # Cylinder mask
        yy, xx = np.meshgrid(np.arange(Ny), np.arange(Nx))
        self.obstacle = ((xx - cyl_cx)**2 + (yy - cyl_cy)**2) <= cyl_r**2

        # Pre-compute equilibrium weight arrays (9,)
        self.w_arr = self.w[:, None, None]  # (9, 1, 1) for broadcasting
        self.ex = self.e[:, 0][:, None, None].astype(float)  # (9, 1, 1)
        self.ey = self.e[:, 1][:, None, None].astype(float)

        # Initialise to equilibrium at uniform flow
        self.rho = np.ones((Nx, Ny))
        self.ux  = np.full((Nx, Ny), U_inf)
        self.uy  = np.zeros((Nx, Ny))
        self.ux[self.obstacle] = 0.0
        self.uy[self.obstacle] = 0.0
        self.f = self._equilibrium(self.rho, self.ux, self.uy)

    def _equilibrium(self, rho, ux, uy):
        """Vectorised equilibrium computation — no Python loop."""
        eu = self.ex * ux[None, :, :] + self.ey * uy[None, :, :]
        usq = ux**2 + uy**2
        return self.w_arr * rho[None, :, :] * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*usq[None, :, :])

    def step(self):
        """Single LBM step. Streaming uses np.roll for y (periodic) and
        slicing for x (non-periodic: inlet/outlet BCs handle boundaries)."""
        #  Collision (BGK) 
        f_eq = self._equilibrium(self.rho, self.ux, self.uy)
        f_out = self.f - self.omega_lbm * (self.f - f_eq)

        #  Streaming 
        # y-direction is periodic (top/bottom), x-direction is NOT (inlet/outlet).
        # Strategy: roll along y, then shift along x via slicing.
        # For direction i with (e_ix, e_iy):
        #   1) roll along y by e_iy (periodic)
        #   2) shift along x by e_ix (non-periodic: fill boundary with neighbor)
        f_new = np.empty_like(f_out)
        Nx = self.Nx
        for i in range(9):
            ex_i, ey_i = self.e[i, 0], self.e[i, 1]
            # Roll y (periodic)
            tmp = np.roll(f_out[i], ey_i, axis=1) if ey_i != 0 else f_out[i]
            # Shift x (non-periodic): copy interior, extrapolate at boundaries
            if ex_i == 0:
                f_new[i] = tmp
            elif ex_i == 1:
                # Stream from x-1 to x: f_new[i, x, :] = tmp[x-1, :]
                f_new[i, 1:, :] = tmp[:-1, :]
                f_new[i, 0, :]  = tmp[0, :]     # extrapolate (overwritten by inlet BC)
            elif ex_i == -1:
                # Stream from x+1 to x: f_new[i, x, :] = tmp[x+1, :]
                f_new[i, :-1, :] = tmp[1:, :]
                f_new[i, -1, :]  = tmp[-1, :]   # extrapolate (overwritten by outlet BC)

        #  Bounce-back on obstacle 
        f_bounce = f_out[:, self.obstacle]            # (9, n_obs)
        for i in range(9):
            f_new[self.opp[i]][self.obstacle] = f_bounce[i]

        #  Outlet (x=Nx-1): zero-gradient (applied before inlet) 
        f_new[:, -1, :] = f_new[:, -2, :]

        #  Zou-He inlet (x=0) 
        rho_in = (f_new[0, 0, :] + f_new[2, 0, :] + f_new[4, 0, :]
                  + 2*(f_new[3, 0, :] + f_new[6, 0, :] + f_new[7, 0, :])) \
                 / (1 - self.U_inf)
        f_new[1, 0, :] = f_new[3, 0, :] + (2.0/3.0) * rho_in * self.U_inf
        f_new[5, 0, :] = f_new[7, 0, :] - 0.5*(f_new[2, 0, :] - f_new[4, 0, :]) \
                          + (1.0/6.0) * rho_in * self.U_inf
        f_new[8, 0, :] = f_new[6, 0, :] + 0.5*(f_new[2, 0, :] - f_new[4, 0, :]) \
                          + (1.0/6.0) * rho_in * self.U_inf

        #  Macroscopic 
        self.f = f_new
        self.rho = np.sum(f_new, axis=0)
        self.ux = np.sum(f_new * self.ex, axis=0) / self.rho
        self.uy = np.sum(f_new * self.ey, axis=0) / self.rho
        self.ux[self.obstacle] = 0.0
        self.uy[self.obstacle] = 0.0

        # Stability check (periodic to avoid overhead)
        self._step_count = getattr(self, '_step_count', 0) + 1
        if self._step_count % 500 == 0:
            if np.any(np.isnan(self.rho)) or np.max(np.abs(self.ux)) > 1.0:
                raise RuntimeError(f"LBM blowup: rho in [{np.nanmin(self.rho):.4f}, {np.nanmax(self.rho):.4f}], "
                                   f"|ux|_max={np.nanmax(np.abs(self.ux)):.4f}")

    def compute_vorticity(self):
        """ω = ∂v/∂x - ∂u/∂y via central differences."""
        dvdx = (np.roll(self.uy, -1, axis=0) - np.roll(self.uy, 1, axis=0)) / 2.0
        dudy = (np.roll(self.ux, -1, axis=1) - np.roll(self.ux, 1, axis=1)) / 2.0
        omega = dvdx - dudy
        omega[self.obstacle] = 0.0
        return omega

    def run_steps(self, n_steps, verbose_every=5000):
        """Run n_steps without saving (for burn-in / phase advancement)."""
        for s in range(1, n_steps + 1):
            self.step()
            if verbose_every and s % verbose_every == 0:
                omega = self.compute_vorticity()
                print(f"    Step {s}/{n_steps}: ω ∈ [{omega.min():.4f}, {omega.max():.4f}]")

    def simulate_and_save(self, n_steps, save_every=100):
        """Run and record snapshots."""
        n_save = n_steps // save_every + 1
        omega_out = np.zeros((n_save, self.Nx, self.Ny), dtype=np.float32)
        omega_out[0] = self.compute_vorticity()
        save_idx = 1

        for s in range(1, n_steps + 1):
            self.step()
            if s % save_every == 0 and save_idx < n_save:
                omega_out[save_idx] = self.compute_vorticity()
                save_idx += 1
            if s % 5000 == 0:
                omega = self.compute_vorticity()
                print(f"    Step {s}/{n_steps}: ω ∈ [{omega.min():.4f}, {omega.max():.4f}]")

        print(f"  Done: {save_idx} snapshots, "
              f"ω ∈ [{omega_out[:save_idx].min():.4f}, {omega_out[:save_idx].max():.4f}]")
        return omega_out[:save_idx]

    def get_state(self):
        return {'f': self.f.copy(), 'rho': self.rho.copy(),
                'ux': self.ux.copy(), 'uy': self.uy.copy()}

    def set_state(self, state):
        self.f = state['f'].copy(); self.rho = state['rho'].copy()
        self.ux = state['ux'].copy(); self.uy = state['uy'].copy()


# MAIN

if __name__ == "__main__":
    #  Grid & base geometry (lattice units)  
    Nx, Ny     = 400, 160
    cyl_r_lu   = 16
    cyl_cx     = Nx // 5
    cyl_cy     = Ny // 2

    #  Time  
    K_sim      = 8
    n_sim      = 14000
    save_every = 100

    # Per-sim burn-in (kept modest for speed; includes transients for diversity)
    # Recommended: ~1–2 shedding periods. We'll estimate per sim from St~0.165.
    burnin_periods = (0.8, 1.6)   # random in this range per sim

    #  Output resolution (downsample from LBM grid)  
    ds_x, ds_y = 4, 4
    out_Nx = Nx // ds_x
    out_Ny = Ny // ds_y

    #  Parameter sweep knobs (safe for BGK stability)  
    # Tip: If you still want more diversity, widen Re_range a bit upward (not too high),
    # or widen U_range slightly, but keep tau comfortably above 0.51.
    U_range  = (0.03, 0.05)
    Re_range = (60.0, 130.0)

    # Inlet modulation (optional). This changes "wavelength/frequency" via time-varying U.
    # Keep amplitude moderate; set amp_range=(0,0) to disable.
    inlet_mod_amp_range = (0.00, 0.15)   # fraction of U0
    inlet_mod_T_scale   = (0.7, 1.6)     # scale on estimated shedding period

    # Stability / diversity guardrails
    tau_min = 0.535          # safer than 0.51; BGK can still blow up around 0.52–0.53
    U_cap   = 0.055          # cap on instantaneous inlet speed U(t) to reduce blowups
    max_attempts_per_sim = 12

    # Random wake perturbation during early burn-in (keeps sims separated)
    kick_amp_range = (5e-4, 4e-3)
    kick_steps     = 400

    seed = 42
    rng = np.random.default_rng(seed)

    #  Derived / bookkeeping  
    DATA_DIR = Path(__file__).resolve().parent / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Wake mask for adding small perturbations (same as original approach)
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    wake_mask = (xx > cyl_cx + 2*cyl_r_lu) & (np.abs(yy - cyl_cy) < 2.2*cyl_r_lu)

    print("=" * 78)
    print("  2D von Kármán Vortex Street — LBM Per-Sim (U,Re) Sweep Ensemble + GT")
    print("=" * 78)
    print(f"  Grid: {Nx}×{Ny}  |  Output: {out_Nx}×{out_Ny}  |  ds=({ds_x},{ds_y})")
    print(f"  Cylinder: cx={cyl_cx}, cy={cyl_cy}, r={cyl_r_lu}")
    print(f"  K_sim={K_sim}, n_sim={n_sim}, save_every={save_every}")
    print(f"  U_range={U_range}, Re_range={Re_range}\n")

    def estimate_T_shed_steps(U):
        # Shedding period ≈ D / (St * U), St ≈ 0.165 around Re~100
        D = 2.0 * cyl_r_lu
        return int(D / (0.165 * float(U)))

    def run_one(tag, U0, Re, mod_amp_frac, mod_T_steps, burnin_steps):
        solver = LatticeBoltzmann2D(Nx, Ny, U0, Re, cyl_cx, cyl_cy, cyl_r_lu)

        # Guardrail: BGK tends to get unhappy as tau → 0.5
        if solver.tau <= tau_min:
            raise RuntimeError(f"{tag}: tau too small for stability (tau={solver.tau:.4f}). "
                               f"Try lowering U_range upper bound or lowering Re_range upper bound.")
        # Clamp inlet modulation amplitude to keep U(t) in a stable range
        # U(t) = U0 * (1 + mod_amp_frac * sin(...))
        # Enforce: U0 * (1 + mod_amp_frac) <= U_cap  =>  mod_amp_frac <= U_cap/U0 - 1
        amp_cap_by_U = max(0.0, (U_cap / float(U0)) - 1.0)
        # Additional cap that tightens when tau is small (more fragile flow)
        if solver.tau < 0.54:
            amp_cap_by_tau = 0.06
        elif solver.tau < 0.55:
            amp_cap_by_tau = 0.10
        else:
            amp_cap_by_tau = 0.15
        mod_amp_frac = float(min(mod_amp_frac, amp_cap_by_U, amp_cap_by_tau))
        solver.mod_amp_used = mod_amp_frac
        print(f"  [{tag}] U0={U0:.4f}, Re={Re:.1f}, nu={solver.nu:.6f}, tau={solver.tau:.4f}, "
              f"burnin={burnin_steps}, mod_amp={mod_amp_frac:.3f}, mod_T={mod_T_steps}")

        #  Burn-in with a small, per-sim random wake kick (transient diversity) 
        amp = rng.uniform(*kick_amp_range)
        phase = rng.uniform(0, 2*np.pi)
        for s in range(1, burnin_steps + 1):
            # Optional inlet modulation by temporarily scaling U_inf (kept simple)
            if mod_amp_frac > 0.0:
                u_in = U0 * (1.0 + mod_amp_frac * np.sin(2*np.pi * s / mod_T_steps + phase))
                solver.U_inf = float(u_in)
            else:
                solver.U_inf = float(U0)

            solver.step()

            if s <= kick_steps:
                decay = np.exp(-s / (0.4 * kick_steps))
                # Kick in uy with a vertical sinusoid in the wake
                solver.uy += amp * decay * wake_mask * np.sin(2*np.pi*yy/Ny + phase)
                solver.ux[solver.obstacle] = 0.0
                solver.uy[solver.obstacle] = 0.0
                solver.rho = np.sum(solver.f, axis=0)
                solver.f = solver._equilibrium(solver.rho, solver.ux, solver.uy)

        #  Record trajectory 
        omega_full = solver.simulate_and_save(n_steps=n_sim, save_every=save_every)  # (T, Nx, Ny)

        # Downsample by block-average
        omega_ds = np.stack([downsample_blockavg(frame, ds_x, ds_y) for frame in omega_full], axis=0)
        return omega_ds, solver

    # Sample parameters for sims
    sims = []
    sim_params = []
    for k in range(K_sim):
        last_err = None
        for attempt in range(1, max_attempts_per_sim + 1):
            U0 = rng.uniform(*U_range)
            Re = rng.uniform(*Re_range)

            # Estimate shedding period and pick a burn-in length (in periods)
            T_shed = estimate_T_shed_steps(U0)
            burnin_steps = int(rng.uniform(*burnin_periods) * T_shed)

            # Inlet modulation (requested); will be clamped inside run_one for stability
            mod_amp_frac = rng.uniform(*inlet_mod_amp_range)
            mod_T_steps  = int(rng.uniform(*inlet_mod_T_scale) * T_shed)
            mod_T_steps  = max(mod_T_steps, 200)

            try:
                omega_ds, solver = run_one(tag=f"sim_{k:02d}", U0=U0, Re=Re,
                                           mod_amp_frac=mod_amp_frac, mod_T_steps=mod_T_steps,
                                           burnin_steps=burnin_steps)
                sims.append(omega_ds)
                sim_params.append(dict(U0=float(U0), Re=float(Re), nu=float(solver.nu), tau=float(solver.tau),
                                       burnin_steps=int(burnin_steps),
                                       inlet_mod_amp_frac=float(getattr(solver, 'mod_amp_used', mod_amp_frac)),
                                       inlet_mod_T_steps=int(mod_T_steps)))
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                print(f"    ! {e}  (resample {attempt}/{max_attempts_per_sim})")

        if last_err is not None:
            raise RuntimeError(f"Failed to generate sim_{k:02d} after {max_attempts_per_sim} attempts. "
                               f"Last error: {last_err}") from last_err

    # Ground truth: draw a different (U,Re) with the same stability guardrails (resample if needed)
    last_err = None
    for attempt in range(1, max_attempts_per_sim + 1):
        U_gt = rng.uniform(*U_range)
        Re_gt = rng.uniform(*Re_range)
        T_gt = estimate_T_shed_steps(U_gt)
        burnin_gt = int(rng.uniform(*burnin_periods) * T_gt)

        # Ensure GT has some mismatch, but cap so U(t) stays reasonable
        mod_amp_gt = max(inlet_mod_amp_range[1], 0.12)  # at least some modulation
        mod_amp_gt = min(mod_amp_gt, 0.25)
        # Cap modulation so U0*(1+amp) <= U_cap
        amp_cap = max(0.0, (U_cap / max(U_gt, 1e-9)) - 1.0)
        mod_amp_gt = min(mod_amp_gt, 0.95 * amp_cap)

        mod_T_gt = int(rng.uniform(*inlet_mod_T_scale) * T_gt)
        mod_T_gt = max(mod_T_gt, 200)

        try:
            omega_gt_ds, solver_gt = run_one(tag="gt", U0=U_gt, Re=Re_gt,
                                             mod_amp_frac=mod_amp_gt, mod_T_steps=mod_T_gt,
                                             burnin_steps=burnin_gt)
            break
        except RuntimeError as e:
            last_err = e
            print(f"    ! gt: {e}  (resample {attempt}/{max_attempts_per_sim})")
    else:
        raise RuntimeError("Failed to generate a stable GT after resampling. "
                           "Try lowering U_range upper bound, lowering Re_range upper bound, "
                           "or increasing tau_min / decreasing U_cap.") from last_err

    # Time vector in LBM steps (uniform across all runs)
    n_snaps = omega_gt_ds.shape[0]
    t_vec = np.arange(n_snaps) * save_every

    # Cylinder mask (downsampled) for consumers
    cyl_mask_full = solver_gt.obstacle.astype(np.uint8)
    cyl_mask_ds = downsample_blockavg(cyl_mask_full, ds_x, ds_y) > 0.0
    cyl_mask_ds = cyl_mask_ds.astype(np.uint8)

    # Save sims
    x = np.linspace(0, 1, out_Nx)
    y = np.linspace(0, 1, out_Ny)
    for k, omega in enumerate(sims):
        fpath = DATA_DIR / f"sim_{k:02d}.npz"
        np.savez(fpath,
                 omega=omega.astype(np.float32),
                 x=x, y=y, t=t_vec.astype(np.float32),
                 dt_save=float(save_every),
                 U0=sim_params[k]["U0"], Re=sim_params[k]["Re"],
                 nu=sim_params[k]["nu"], tau=sim_params[k]["tau"],
                 cyl_mask=cyl_mask_ds,
                 Nx=out_Nx, Ny=out_Ny)

    # Save GT
    gt_path = DATA_DIR / "gt.npz"
    np.savez(gt_path,
             omega=omega_gt_ds.astype(np.float32),
             x=x, y=y, t=t_vec.astype(np.float32),
             dt_save=float(save_every),
             U0=float(U_gt), Re=float(Re_gt),
             nu=float(solver_gt.nu), tau=float(solver_gt.tau),
             cyl_mask=cyl_mask_ds,
             Nx=out_Nx, Ny=out_Ny)

    # Save metadata
    meta = dict(
        Nx=Nx, Ny=Ny,
        out_Nx=out_Nx, out_Ny=out_Ny, ds_x=ds_x, ds_y=ds_y,
        cyl_cx=int(cyl_cx), cyl_cy=int(cyl_cy), cyl_r_lu=int(cyl_r_lu),
        K_sim=int(K_sim), n_sim=int(n_sim), save_every=int(save_every),
        U_range=tuple(map(float, U_range)), Re_range=tuple(map(float, Re_range)),
        burnin_periods=tuple(map(float, burnin_periods)),
        inlet_mod_amp_range=tuple(map(float, inlet_mod_amp_range)),
        inlet_mod_T_scale=tuple(map(float, inlet_mod_T_scale)),
        kick_amp_range=tuple(map(float, kick_amp_range)), kick_steps=int(kick_steps),
        seed=int(seed),
        sim_params=sim_params,
        gt_params=dict(U0=float(U_gt), Re=float(Re_gt), nu=float(solver_gt.nu), tau=float(solver_gt.tau),
                       burnin_steps=int(burnin_gt),
                       inlet_mod_amp_frac=float(mod_amp_gt), inlet_mod_T_steps=int(mod_T_gt)),
    )
    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone. Wrote:")
    for k in range(K_sim):
        print(f"  - {DATA_DIR / f'sim_{k:02d}.npz'}")
    print(f"  - {gt_path}")
    print(f"  - {DATA_DIR / 'metadata.json'}")
