"""
Shared per-sensor time-series comparison plot.
"""

import numpy as np
from pathlib import Path


def save_timeseries(gt, pred_lapis, pred_shred, sensors, obs_len, out_path,
                    transform_fn=None, ylabel_fmt="({r},{c})", title="",
                    obs_side="start"):
    """
    Plot sensor time-series comparing GT, LAPIS, and SHRED.

    Args:
        gt, pred_lapis, pred_shred: (T, ...) arrays
        sensors: (n_sensors, 2) array of (row, col)
        obs_len: observation window length
        out_path: output file path
        transform_fn: optional (T, ...) -> (T, H, W) transform before extraction
        ylabel_fmt: format string with {r}, {c} for sensor label
        title: plot suptitle
        obs_side: "start" (forward) or "end" (backward) for obs boundary line
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if transform_fn is not None:
        gt = transform_fn(gt)
        pred_lapis = transform_fn(pred_lapis)
        pred_shred = transform_fn(pred_shred)

    n_show = min(6, len(sensors))
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    T = gt.shape[0]
    t_axis = np.arange(T)

    if obs_side == "end":
        obs_boundary = T - obs_len
    else:
        obs_boundary = obs_len

    for si in range(n_show):
        r, c = sensors[si]
        ax = axes[si]
        ax.plot(t_axis, gt[:, r, c], "k-", lw=1.5, alpha=0.8, label="GT")
        ax.plot(t_axis, pred_lapis[:, r, c], "r--", lw=1.2, alpha=0.8, label="LAPIS")
        ax.plot(t_axis, pred_shred[:, r, c], "b:", lw=1.2, alpha=0.8, label="SHRED")
        ax.axvline(obs_boundary, color="green", ls="--", alpha=0.5, label="obs boundary")
        ax.set_ylabel(ylabel_fmt.format(r=r, c=c), fontsize=8)
        if si == 0:
            ax.legend(fontsize=7, ncol=4)

    axes[-1].set_xlabel("Time step")
    plt.suptitle(title or "Per-Sensor Time Series Comparison", fontsize=11)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")
