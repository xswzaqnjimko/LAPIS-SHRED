"""
Shared utilities for LAPIS experiments.

Provides:
  - place_sensors: variance-weighted or stratified sensor placement
  - TeeLogger: stdout tee to log file
  - to_json_safe: numpy-safe JSON serialisation
"""

import sys
import numpy as np


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


class TeeLogger:
    """Tee stdout to both console and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w")

    def write(self, message):
        self.terminal.write(message)
        if not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if not self.log_file.closed:
            self.log_file.flush()

    def close_log(self):
        self.log_file.close()


def to_json_safe(obj):
    """Recursively convert numpy scalars/arrays to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
