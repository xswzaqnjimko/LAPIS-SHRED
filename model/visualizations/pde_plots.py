"""
PDE experiment visualization: polished publication figures.

Provides:
  - save_symlog_results: 4-row (GT/LAPIS/SHRED/|error|) with SymLogNorm + LogNorm
  - save_kvs_results: 3-row (GT/LAPIS/SHRED) with boundary trimming and scaled colorbar
  - save_velocity_and_vorticity: dual-figure output for Kolmogorov flow (speed + vorticity)
"""

import numpy as np
from pathlib import Path


def _visual_ticks(norm, n=9):
    """Generate evenly spaced ticks in normalised space."""
    fracs = np.linspace(0, 1, n)
    return [norm.inverse(f) for f in fracs]


def _smart_fmt(x):
    if x == 0 or abs(x) >= 0.1:
        return f"{x:.1f}"
    return f"{x:.1e}"


def _smart_fmt_scaled(x, scale):
    v = x / scale
    if v == 0:
        return "0"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.1f}"


# 2D KS and similar scalar chaotic fields

def save_symlog_results(gt, pred_lapis, pred_shred, out_path,
                        sensor_locs=None, transpose=False, origin="upper",
                        cmap="RdBu_r", linthresh_frac=0.03, field_label="u"):
    """
    4-row figure (GT / LAPIS / SHRED / |error|) with SymLogNorm main panels
    and LogNorm error panels.  Suitable for KS and similar chaotic scalar fields.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm, LogNorm
    except ImportError:
        print("  (matplotlib unavailable)")
        return

    T = gt.shape[0]
    idxs = np.linspace(0, T - 1, 5, dtype=int)

    all_vals = np.concatenate([np.abs(d[t]).ravel() for t in idxs
                               for d in [gt, pred_lapis, pred_shred]])
    global_vmax = np.percentile(all_vals, 99.5)
    global_err_max = max(np.abs(pred_lapis[t] - gt[t]).max() for t in idxs)

    linthresh = global_vmax * linthresh_frac
    norm_main = SymLogNorm(linthresh=linthresh, linscale=1.0,
                           base=np.e, vmin=-global_vmax, vmax=global_vmax)
    err_vmin = max(global_err_max * 1e-4, 1e-10)
    norm_err = LogNorm(vmin=err_vmin, vmax=global_err_max)

    fig, axes = plt.subplots(4, 5, figsize=(22, 18))
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.04, top=0.93,
                        wspace=0.05, hspace=0.12)
    ims_main, ims_err = [], []

    def _show(ax, frame):
        f = frame.T if transpose else frame
        return ax.imshow(f, cmap=cmap, origin=origin, norm=norm_main, aspect='equal')

    for col, t in enumerate(idxs):
        for row, data in enumerate([gt, pred_lapis, pred_shred]):
            im = _show(axes[row, col], data[t])
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            ims_main.append(im)
        if col == 0 and sensor_locs is not None:
            sy, sx = (sensor_locs[:, 1], sensor_locs[:, 0]) if transpose else (sensor_locs[:, 1], sensor_locs[:, 0])
            axes[0, col].scatter(sx, sy, marker="x", c="k", s=48, linewidths=1, alpha=0.8)
        err = np.abs(pred_lapis[t] - gt[t])
        ef = err.T if transpose else err
        im_err = axes[3, col].imshow(ef, cmap="hot", origin=origin, norm=norm_err, aspect='equal')
        axes[3, col].set_xticks([]); axes[3, col].set_yticks([])
        ims_err.append(im_err)

    for col, t in enumerate(idxs):
        axes[0, col].set_title(f"t={t}", fontsize=28)
    for r, label in enumerate(["GT", "LAPIS", "SHRED", "|error|"]):
        axes[r, 0].set_ylabel(label, fontsize=28, rotation=90, labelpad=14)

    fig.canvas.draw()
    bbox_top = axes[0, -1].get_position()
    bbox_r2 = axes[2, -1].get_position()
    cbar_ax1 = fig.add_axes([0.90, bbox_r2.y0, 0.015, bbox_top.y1 - bbox_r2.y0])
    cb1 = fig.colorbar(ims_main[0], cax=cbar_ax1)
    ticks1 = _visual_ticks(norm_main, 9)
    cb1.set_ticks(ticks1)
    cb1.set_ticklabels([_smart_fmt(t) for t in ticks1])
    cb1.ax.tick_params(labelsize=22)
    cb1.ax.minorticks_off()

    bbox_r3 = axes[3, -1].get_position()
    cbar_ax2 = fig.add_axes([0.90, bbox_r3.y0, 0.015, bbox_r3.y1 - bbox_r3.y0])
    cb2 = fig.colorbar(ims_err[0], cax=cbar_ax2)
    ticks2 = _visual_ticks(norm_err, 5)
    cb2.set_ticks(ticks2)
    cb2.set_ticklabels([_smart_fmt(t) for t in ticks2])
    cb2.ax.tick_params(labelsize=22)
    cb2.ax.minorticks_off()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved {out_path}")


# 2D KVS (von Karman vortex street)

def save_kvs_results(gt, pred_lapis, pred_shred, out_path,
                     sensor_locs=None, trim_rows=2):
    """
    3-row figure (GT / LAPIS / SHRED) with boundary trimming, transposed
    display, and scaled colorbar with exponent notation.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        return

    T = gt.shape[0]
    idxs = np.linspace(0, T - 1, 5, dtype=int)

    global_vlim = 0
    for t in idxs:
        for data in [gt, pred_lapis, pred_shred]:
            global_vlim = max(global_vlim, np.abs(data[t]).max())

    norm_main = Normalize(vmin=-global_vlim, vmax=global_vlim)
    exponent = int(np.floor(np.log10(max(global_vlim, 1e-30))))
    scale = 10.0 ** exponent

    fig, axes = plt.subplots(3, 5, figsize=(30, 8))
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.05, top=0.92,
                        wspace=0.05, hspace=0.15)
    ims_main = []

    for col, t in enumerate(idxs):
        for row, data in enumerate([gt, pred_lapis, pred_shred]):
            frame = data[t]
            if trim_rows > 0:
                frame = frame[trim_rows:-trim_rows, :]
            im = axes[row, col].imshow(frame.T, cmap="RdBu_r", origin="lower",
                                       aspect='auto', norm=norm_main)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            ims_main.append(im)
        if col == 0 and sensor_locs is not None:
            axes[0, col].scatter(sensor_locs[:, 0], sensor_locs[:, 1],
                                 marker="x", c="k", s=48, linewidths=1, alpha=0.8)

    for col, t in enumerate(idxs):
        axes[0, col].set_title(f"t={t}", fontsize=28)
    for r, label in enumerate(["GT", "LAPIS", "SHRED"]):
        axes[r, 0].set_ylabel(label, fontsize=28, rotation=90, labelpad=14)

    fig.canvas.draw()
    bbox_top = axes[0, -1].get_position()
    bbox_bot = axes[2, -1].get_position()
    cbar_ax = fig.add_axes([0.90, bbox_bot.y0, 0.015, bbox_top.y1 - bbox_bot.y0])
    cb = fig.colorbar(ims_main[0], cax=cbar_ax)
    ticks = np.linspace(-global_vlim, global_vlim, 9)
    cb.set_ticks(ticks)
    cb.set_ticklabels([_smart_fmt_scaled(t, scale) for t in ticks])
    cb.ax.tick_params(labelsize=22)
    cb.ax.set_title(f"$\\times 10^{{{exponent}}}$", fontsize=22, pad=12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved {out_path}")


# 2D KF (Kolmogorov flow): dual vorticity + velocity figures

def save_velocity_and_vorticity(gt_uv, pred_lapis_uv, pred_shred_uv, out_dir,
                                sensor_locs=None, compute_vorticity_fn=None):
    """
    Produce two 4-row figures: one for vorticity (SymLogNorm), one for
    velocity speed (LogNorm).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm, LogNorm
    except ImportError:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = gt_uv.shape[0]
    idxs = np.linspace(0, T - 1, 5, dtype=int)

    # Vorticity
    if compute_vorticity_fn is None:
        raise ValueError("compute_vorticity_fn required for KF plots")

    gt_vort = compute_vorticity_fn(gt_uv)
    lapis_vort = compute_vorticity_fn(pred_lapis_uv)
    shred_vort = compute_vorticity_fn(pred_shred_uv)

    all_vort = np.concatenate([np.abs(v[t]).ravel() for t in idxs
                               for v in [gt_vort, lapis_vort, shred_vort]])
    vmax_vort = np.percentile(all_vort, 99.5)
    err_vort_max = max(np.abs(lapis_vort[t] - gt_vort[t]).max() for t in idxs)

    _make_symlog_figure(gt_vort, lapis_vort, shred_vort, idxs,
                        vmax_vort, err_vort_max, linthresh_frac=0.2,
                        sensor_locs=sensor_locs,
                        out_path=out_dir / "lapis_2dkf_vorticity.png")

    # Velocity speed
    def _speed(uv):
        return np.sqrt(uv[:, 0, :, :] ** 2 + uv[:, 1, :, :] ** 2)

    gt_speed = _speed(gt_uv)
    lapis_speed = _speed(pred_lapis_uv)
    shred_speed = _speed(pred_shred_uv)

    all_speed = np.concatenate([s[t].ravel() for t in idxs
                                for s in [gt_speed, lapis_speed, shred_speed]])
    vmax_vel = np.percentile(all_speed, 99.5)
    err_vel_max = max(np.sqrt(np.sum((pred_lapis_uv[t] - gt_uv[t]) ** 2, axis=0)).max()
                      for t in idxs)

    _make_log_figure(gt_speed, lapis_speed, shred_speed,
                     pred_lapis_uv, gt_uv, idxs,
                     vmax_vel, err_vel_max,
                     sensor_locs=sensor_locs,
                     out_path=out_dir / "lapis_2dkf_velocity.png")


def _make_symlog_figure(gt, pred_l, pred_s, idxs, vmax, err_max,
                        linthresh_frac=0.2, sensor_locs=None, out_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm, LogNorm

    linthresh = vmax * linthresh_frac
    norm_main = SymLogNorm(linthresh=linthresh, linscale=1.0,
                           base=np.e, vmin=-vmax, vmax=vmax)
    err_vmin = max(err_max * 1e-4, 1e-10)
    norm_err = LogNorm(vmin=err_vmin, vmax=err_max)

    fig, axes = plt.subplots(4, 5, figsize=(22, 18))
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.04, top=0.93,
                        wspace=0.05, hspace=0.12)
    ims_main, ims_err = [], []

    for col, t in enumerate(idxs):
        for row, data in enumerate([gt, pred_l, pred_s]):
            im = axes[row, col].imshow(data[t].T, cmap="RdBu_r", origin="lower",
                                       norm=norm_main, aspect='equal')
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            ims_main.append(im)
        if col == 0 and sensor_locs is not None:
            axes[0, col].scatter(sensor_locs[:, 1], sensor_locs[:, 0],
                                 marker="x", c="k", s=48, linewidths=1, alpha=0.8)
        err = np.abs(pred_l[t] - gt[t])
        im_err = axes[3, col].imshow(err.T, cmap="hot", origin="lower",
                                      norm=norm_err, aspect='equal')
        axes[3, col].set_xticks([]); axes[3, col].set_yticks([])
        ims_err.append(im_err)

    for col, t in enumerate(idxs):
        axes[0, col].set_title(f"t={t}", fontsize=28)
    for r, label in enumerate(["GT", "LAPIS", "SHRED", "|error|"]):
        axes[r, 0].set_ylabel(label, fontsize=28, rotation=90, labelpad=14)

    fig.canvas.draw()
    bbox_top = axes[0, -1].get_position()
    bbox_r2 = axes[2, -1].get_position()
    cbar_ax1 = fig.add_axes([0.90, bbox_r2.y0, 0.015, bbox_top.y1 - bbox_r2.y0])
    cb1 = fig.colorbar(ims_main[0], cax=cbar_ax1)
    cb1.set_ticks(_visual_ticks(norm_main, 9))
    cb1.set_ticklabels([_smart_fmt(t) for t in _visual_ticks(norm_main, 9)])
    cb1.ax.tick_params(labelsize=22); cb1.ax.minorticks_off()

    bbox_r3 = axes[3, -1].get_position()
    cbar_ax2 = fig.add_axes([0.90, bbox_r3.y0, 0.015, bbox_r3.y1 - bbox_r3.y0])
    cb2 = fig.colorbar(ims_err[0], cax=cbar_ax2)
    cb2.set_ticks(_visual_ticks(norm_err, 5))
    cb2.set_ticklabels([_smart_fmt(t) for t in _visual_ticks(norm_err, 5)])
    cb2.ax.tick_params(labelsize=22); cb2.ax.minorticks_off()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {out_path}")
    plt.close()


def _make_log_figure(gt_speed, lapis_speed, shred_speed,
                     pred_lapis_uv, gt_uv, idxs,
                     vmax, err_max, sensor_locs=None, out_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    vmin = max(vmax * 1e-4, 1e-10)
    norm_vel = LogNorm(vmin=vmin, vmax=vmax)
    err_vmin = max(err_max * 1e-4, 1e-10)
    norm_err = LogNorm(vmin=err_vmin, vmax=err_max)

    fig, axes = plt.subplots(4, 5, figsize=(22, 18))
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.04, top=0.93,
                        wspace=0.05, hspace=0.12)
    ims_main, ims_err = [], []

    for col, t in enumerate(idxs):
        for row, data in enumerate([gt_speed, lapis_speed, shred_speed]):
            im = axes[row, col].imshow(data[t].T, cmap="viridis", origin="lower",
                                       norm=norm_vel, aspect='equal')
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            ims_main.append(im)
        if col == 0 and sensor_locs is not None:
            axes[0, col].scatter(sensor_locs[:, 1], sensor_locs[:, 0],
                                 marker="x", c="k", s=48, linewidths=1, alpha=0.8)
        vel_err = np.sqrt(np.sum((pred_lapis_uv[t] - gt_uv[t]) ** 2, axis=0))
        im_err = axes[3, col].imshow(vel_err.T, cmap="hot", origin="lower",
                                      norm=norm_err, aspect='equal')
        axes[3, col].set_xticks([]); axes[3, col].set_yticks([])
        ims_err.append(im_err)

    for col, t in enumerate(idxs):
        axes[0, col].set_title(f"t={t}", fontsize=28)
    for r, label in enumerate(["GT", "LAPIS", "SHRED", "|error|"]):
        axes[r, 0].set_ylabel(label, fontsize=28, rotation=90, labelpad=14)

    fig.canvas.draw()
    bbox_top = axes[0, -1].get_position()
    bbox_r2 = axes[2, -1].get_position()
    cbar_ax1 = fig.add_axes([0.90, bbox_r2.y0, 0.015, bbox_top.y1 - bbox_r2.y0])
    cb1 = fig.colorbar(ims_main[0], cax=cbar_ax1)
    cb1.set_ticks(_visual_ticks(norm_vel, 9))
    cb1.set_ticklabels([_smart_fmt(t) for t in _visual_ticks(norm_vel, 9)])
    cb1.ax.tick_params(labelsize=22); cb1.ax.minorticks_off()

    bbox_r3 = axes[3, -1].get_position()
    cbar_ax2 = fig.add_axes([0.90, bbox_r3.y0, 0.015, bbox_r3.y1 - bbox_r3.y0])
    cb2 = fig.colorbar(ims_err[0], cax=cbar_ax2)
    cb2.set_ticks(_visual_ticks(norm_err, 5))
    cb2.set_ticklabels([_smart_fmt(t) for t in _visual_ticks(norm_err, 5)])
    cb2.ax.tick_params(labelsize=22); cb2.ax.minorticks_off()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {out_path}")
    plt.close()
