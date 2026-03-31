"""
Shared results grid visualization: 4-row x 5-column comparison figure.

Rows: Ground Truth, LAPIS, SHRED, |error|
Columns: 5 evenly spaced time snapshots
"""

import numpy as np
from pathlib import Path


def save_results_grid(gt, pred_lapis, pred_shred, obs_len, metrics_lapis, metrics_shred,
                      out_path, sensor_locs=None, cmap="RdBu_r", err_cmap="hot",
                      vmin=None, vmax=None, symmetric=True, origin="lower",
                      field_label="", title_prefix="LAPIS", mask=None,
                      transform_fn=None):
    """
    Save a 4x5 comparison figure (GT / LAPIS / SHRED / |error|).

    Args:
        gt, pred_lapis, pred_shred: (T, H, W) spatiotemporal arrays
        obs_len: observation window length (for annotation)
        metrics_lapis, metrics_shred: dicts with 'rmse_active', 'ssim'
        out_path: output file path
        sensor_locs: (n_sensors, 2) for overlay on first GT panel
        cmap: colormap for main panels
        err_cmap: colormap for error panel
        vmin, vmax: color limits (auto-computed if None)
        symmetric: if True, use symmetric limits around zero
        origin: imshow origin ("lower" or "upper")
        field_label: colorbar label (e.g. "NDSI", "omega")
        title_prefix: suptitle prefix
        mask: optional mask to apply to all panels
        transform_fn: optional callable to transform data before plotting
            (e.g. compute vorticity from velocity)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable)")
        return

    if transform_fn is not None:
        gt = transform_fn(gt)
        pred_lapis = transform_fn(pred_lapis)
        pred_shred = transform_fn(pred_shred)

    T = gt.shape[0]
    idxs = np.linspace(0, T - 1, 5, dtype=int)

    # Auto-compute color limits
    if vmin is None or vmax is None:
        all_vals = np.concatenate([d[t].ravel() for t in idxs for d in [gt, pred_lapis, pred_shred]])
        if symmetric:
            vlim = np.percentile(np.abs(all_vals), 99.5)
            vmin, vmax = -vlim, vlim
        else:
            vmin = np.percentile(all_vals, 0.5)
            vmax = np.percentile(all_vals, 99.5)

    global_err_max = max(np.abs(pred_lapis[t] - gt[t]).max() for t in idxs)

    fig, axes = plt.subplots(4, 5, figsize=(22, 18))
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.04, top=0.93, wspace=0.05, hspace=0.12)

    ims_main, ims_err = [], []
    for col, t in enumerate(idxs):
        for row, data in enumerate([gt, pred_lapis, pred_shred]):
            frame = data[t]
            if mask is not None:
                frame = np.where(mask, frame, np.nan)
            im = axes[row, col].imshow(
                frame.T if origin == "lower" else frame,
                cmap=cmap, origin=origin, vmin=vmin, vmax=vmax, aspect='equal')
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            ims_main.append(im)

        if col == 0 and sensor_locs is not None:
            if origin == "lower":
                axes[0, col].scatter(sensor_locs[:, 1], sensor_locs[:, 0],
                                     marker="x", c="k", s=48, linewidths=1, alpha=0.8)
            else:
                axes[0, col].scatter(sensor_locs[:, 1], sensor_locs[:, 0],
                                     marker="x", c="k", s=24, linewidths=1, alpha=0.8, zorder=5)

        err = np.abs(pred_lapis[t] - gt[t])
        im_err = axes[3, col].imshow(
            err.T if origin == "lower" else err,
            cmap=err_cmap, origin=origin, vmin=0, vmax=global_err_max, aspect='equal')
        axes[3, col].set_xticks([]); axes[3, col].set_yticks([])
        ims_err.append(im_err)

    for col, t in enumerate(idxs):
        axes[0, col].set_title(f"t={t}", fontsize=28)
    for r, label in enumerate(["GT", "LAPIS", "SHRED", "|error|"]):
        axes[r, 0].set_ylabel(label, fontsize=28, rotation=90, labelpad=14)

    # Colorbars aligned to panel positions
    fig.canvas.draw()
    bbox_top = axes[0, -1].get_position()
    bbox_r2 = axes[2, -1].get_position()
    cbar_ax1 = fig.add_axes([0.90, bbox_r2.y0, 0.015, bbox_top.y1 - bbox_r2.y0])
    cb1 = fig.colorbar(ims_main[0], cax=cbar_ax1)
    if field_label:
        cb1.set_label(field_label, fontsize=26)
    cb1.ax.tick_params(labelsize=22)

    bbox_r3 = axes[3, -1].get_position()
    cbar_ax2 = fig.add_axes([0.90, bbox_r3.y0, 0.015, bbox_r3.y1 - bbox_r3.y0])
    cb2 = fig.colorbar(ims_err[0], cax=cbar_ax2)
    cb2.set_label("|error|", fontsize=26)
    cb2.ax.tick_params(labelsize=22)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")
