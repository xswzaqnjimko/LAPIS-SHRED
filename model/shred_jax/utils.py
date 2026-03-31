"""
Shared utilities for LAPIS experiments.

Provides:
  - place_sensors: variance-weighted or stratified sensor placement
  - TeeLogger: stdout tee to log file
  - to_json_safe: numpy-safe JSON serialisation
"""

import sys
import numpy as np


def place_sensors(sim_grids, H, W, n_sensors, strategy="stratified", seed=42):
    """Place sensors using variance-weighted or stratified sampling.

    Args:
        sim_grids: list of (T, H, W) arrays (or variance maps computed externally)
        H, W: spatial dimensions
        n_sensors: number of sensors to place
        strategy: "stratified" (variance-weighted with boundary exclusion),
                  "variance" (pure variance-weighted), or "random"
        seed: random seed
    Returns:
        (n_sensors, 2) array of (row, col) sensor locations
    """
    rng = np.random.RandomState(seed)

    if strategy in ("stratified", "variance"):
        var_maps = [np.var(g, axis=0) for g in sim_grids]
        variance = np.mean(var_maps, axis=0)
        flat_var = variance.ravel()

        if strategy == "stratified":
            mask = np.zeros(H * W, dtype=bool)
            for i in range(H):
                for j in range(W):
                    if 2 <= i < H - 2 and 2 <= j < W - 2:
                        mask[i * W + j] = True
            weights = np.where(mask, flat_var + 1e-6, 0.0)
        else:
            weights = flat_var + 1e-8

        weights = weights / weights.sum()
        indices = rng.choice(H * W, size=n_sensors, replace=False, p=weights)
        rows = indices // W
        cols = indices % W
        return np.column_stack([rows, cols])
    else:
        rows = rng.randint(2, H - 2, size=n_sensors)
        cols = rng.randint(2, W - 2, size=n_sensors)
        return np.stack([rows, cols], axis=1)


class TeeLogger:
    """Tee stdout to both console and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
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
