"""
NDSI-specific visualization functions.

Provides:
  - SCAF diagnostic plot
  - Cut data preview (per-year temporal snapshots)
  - Figure 1 sensor overlay GIFs and time-series
  - NDSI snow-cover animations (GT, reconstruction, error)
"""

import numpy as np
from pathlib import Path


# SCAF diagnostics

def plot_scaf_diagnostics(sim_grids_raw, gt_grid_raw, sim_ep_info, gt_ep_info,
                          config, out_path):
    """Plot SCAF curves for all years with cut-point annotations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable)")
        return

    def _compute_scaf(arr, tau):
        T = arr.shape[0]
        scaf = np.zeros(T)
        for t in range(T):
            frame = arr[t]
            valid = np.isfinite(frame)
            n_valid = valid.sum()
            scaf[t] = (frame[valid] > tau).sum() / n_valid if n_valid > 0 else 0.0
        return scaf

    def _moving_average(x, window):
        if window <= 1:
            return x.copy()
        return np.convolve(x, np.ones(window) / window, mode='same')

    n_sim = len(sim_grids_raw)
    fig, axes = plt.subplots(n_sim + 1, 1, figsize=(12, 2.5 * (n_sim + 1)), sharex=False)
    if n_sim + 1 == 1:
        axes = [axes]

    all_grids = list(sim_grids_raw) + [gt_grid_raw]
    all_info = list(sim_ep_info) + [gt_ep_info]
    labels = [str(yr) for yr in config.SIM_YEARS[:n_sim]] + [str(config.GT_YEAR)]

    for i, (arr, info, label) in enumerate(zip(all_grids, all_info, labels)):
        ax = axes[i]
        scaf = _compute_scaf(arr, config.TAU)
        scaf_s = _moving_average(scaf, config.SMOOTH_WINDOW)
        ax.plot(scaf, alpha=0.4, label="SCAF (raw)")
        ax.plot(scaf_s, "k-", lw=1.5, label="SCAF (smooth)")
        ax.axhline(config.RHO, color="red", ls="--", alpha=0.5, label=f"rho={config.RHO}")
        t_peak = info.get('t_peak', 0)
        t_end = info.get('t_end', len(scaf) - 1)
        ax.axvline(t_peak, color="blue", ls=":", alpha=0.6, label=f"t_peak={t_peak}")
        ax.axvline(t_end, color="green", ls=":", alpha=0.6, label=f"t_end={t_end}")
        ax.set_ylabel("SCAF")
        ax.set_title(f"{label}  (T_orig={arr.shape[0]}, T_cut={info.get('T_cut', '?')})",
                     fontsize=9)
        ax.legend(fontsize=6, ncol=3)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


# Cut data preview

def save_cut_data_preview(sim_grids, gt_grid, sim_ep_info, gt_ep_info, config, out_path):
    """Preview figure of SCAF-cut data: one row per year, 5 temporal snapshots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_cols = 5
    years = config.SIM_YEARS[:len(sim_grids)]
    all_rows = [(str(yr), sim_grids[i], "Sim") for i, yr in enumerate(years)]
    all_rows.append((str(config.GT_YEAR), gt_grid, "GT"))
    n_rows = len(all_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row_i, (label, arr, kind) in enumerate(all_rows):
        T = arr.shape[0]
        t_idxs = np.linspace(0, T - 1, n_cols, dtype=int)
        for col, t in enumerate(t_idxs):
            ax = axes[row_i, col]
            ax.imshow(arr[t], cmap="Blues_r", vmin=-1.0, vmax=1.0)
            if col == 0:
                ax.text(0.02, 0.98, f"{kind} {label}\nT={T}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            ax.set_title(f"t={t}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.suptitle("LAPIS-NDSI: Per-Year Snow Cover (SCAF-cut)", fontsize=13, y=1.02)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


# Figure 1: sensor overlay GIFs and time-series

def select_highlighted_sensors(sensor_locs, n_highlight=3, seed=123):
    """Select well-separated highlighted sensors across the column range."""
    cols = sensor_locs[:, 1].astype(float)
    col_min, col_max = cols.min(), cols.max()
    band_edges = np.linspace(col_min, col_max, n_highlight + 1)

    selected, used = [], set()
    for b in range(n_highlight):
        lo, hi = band_edges[b], band_edges[b + 1]
        band_center = (lo + hi) / 2.0
        candidates = [i for i in range(len(sensor_locs)) if i not in used and lo <= cols[i] <= hi]
        if not candidates:
            dists = np.abs(cols - band_center)
            for i in np.argsort(dists):
                if i not in used:
                    candidates = [int(i)]
                    break
        best = min(candidates, key=lambda i: abs(cols[i] - band_center))
        selected.append(best)
        used.add(best)
    return selected


def make_sensor_gif(ndsi_grid, sensor_locs, highlight_idx, title_prefix,
                    out_path, obs_window=None, fps=3):
    """Create a GIF of NDSI frames with sensor overlays."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        return

    T, H, W = ndsi_grid.shape
    highlight_set = set(highlight_idx)
    normal_idx = [i for i in range(len(sensor_locs)) if i not in highlight_set]

    _styles = [
        {'marker': '^', 'c': 'tomato', 'edgecolors': 'darkred', 's': 600, 'linewidths': 2.0},
        {'marker': 'D', 'c': 'wheat', 'edgecolors': 'goldenrod', 's': 480, 'linewidths': 2.0},
        {'marker': 'p', 'c': 'hotpink', 'edgecolors': 'deeppink', 's': 660, 'linewidths': 2.0},
    ]

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(t):
        ax.clear()
        ax.imshow(ndsi_grid[t], cmap="Blues_r", vmin=-1.0, vmax=1.0, aspect="equal")
        ax.scatter(sensor_locs[normal_idx, 1], sensor_locs[normal_idx, 0],
                   marker='x', c='black', s=36, linewidths=1.4, zorder=5)
        for j, si in enumerate(highlight_idx):
            style = _styles[j % len(_styles)]
            ax.scatter(sensor_locs[si, 1], sensor_locs[si, 0],
                       marker=style['marker'], c=style['c'],
                       edgecolors=style['edgecolors'], s=style['s'],
                       linewidths=style['linewidths'], zorder=6)
        obs_marker = ""
        if obs_window is not None and obs_window[0] <= t <= obs_window[1]:
            obs_marker = "  [obs]"
        ax.set_title(f"{title_prefix}  t={t}/{T-1}{obs_marker}", fontsize=11)
        ax.axis("off")

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 // fps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer="pillow", fps=fps, dpi=120)
    plt.close()
    print(f"  Saved GIF: {out_path}")


def plot_sensor_timeseries_sim(sim_grid, sensor_locs, highlight_idx, out_path,
                               sim_label="Sim"):
    """Per-highlighted-sensor time-series for a single simulation year."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _styles = [
        {'marker': '^', 'c': 'tomato', 'edgecolors': 'darkred'},
        {'marker': 'D', 'c': 'wheat', 'edgecolors': 'goldenrod'},
        {'marker': 'p', 'c': 'hotpink', 'edgecolors': 'deeppink'},
    ]

    n_show = len(highlight_idx)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 1.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    T = sim_grid.shape[0]
    for i, si in enumerate(highlight_idx):
        ax = axes[i]
        r, c = sensor_locs[si]
        ts = sim_grid[:, r, c]
        ax.plot(np.arange(T), ts, color='#696969', linewidth=6)
        ax.set_ylabel(f"Sensor {si}\n(r={r},c={c})", fontsize=8, labelpad=45)
        style = _styles[i % len(_styles)]
        ax.plot(-0.04, 0.5, transform=ax.transAxes, marker=style['marker'],
                color=style['c'], markeredgecolor=style['edgecolors'],
                markersize=20, markeredgewidth=2.0, clip_on=False, zorder=10)
        ymin = ts.min() - 0.1 * max(abs(ts.max() - ts.min()), 0.2)
        ymax = ts.max() + 0.1 * max(abs(ts.max() - ts.min()), 0.2)
        ax.set_ylim(ymin, ymax)
        ax.grid(False)
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False, length=0)
        ax.set_xlim(0, T)

    fig.suptitle(f"{sim_label} - Sensor Time Series", fontsize=11)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def plot_sensor_timeseries_gt(gt_grid, sensor_locs, highlight_idx, obs_len, out_path,
                              gt_label="GT", obs_side="end"):
    """Per-highlighted-sensor time-series for ground truth with obs annotation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return

    _styles = [
        {'marker': '^', 'c': 'tomato', 'edgecolors': 'darkred'},
        {'marker': 'D', 'c': 'wheat', 'edgecolors': 'goldenrod'},
        {'marker': 'p', 'c': 'hotpink', 'edgecolors': 'deeppink'},
    ]

    n_show = len(highlight_idx)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 1.8 * n_show + 0.5), sharex=True)
    if n_show == 1:
        axes = [axes]

    T = gt_grid.shape[0]
    obs_start = T - obs_len if obs_side == "end" else 0
    obs_end = T if obs_side == "end" else obs_len

    for i, si in enumerate(highlight_idx):
        ax = axes[i]
        r, c = sensor_locs[si]
        ts = gt_grid[:, r, c]
        ax.plot(np.arange(T), ts, color='C0', linewidth=6)
        ax.scatter(np.arange(obs_start, obs_end), ts[obs_start:obs_end],
                   color='red', s=240, zorder=5, edgecolors='darkred', linewidths=0.8)
        ax.set_ylabel(f"Sensor {si}\n(r={r},c={c})", fontsize=8, labelpad=45)
        style = _styles[i % len(_styles)]
        ax.plot(-0.04, 0.5, transform=ax.transAxes, marker=style['marker'],
                color=style['c'], markeredgecolor=style['edgecolors'],
                markersize=20, markeredgewidth=2.0, clip_on=False, zorder=10)
        ymin = ts.min() - 0.1 * max(abs(ts.max() - ts.min()), 0.2)
        ymax = ts.max() + 0.1 * max(abs(ts.max() - ts.min()), 0.2)
        ax.set_ylim(ymin, ymax)
        ax.grid(False)
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False, length=0)
        ax.set_xlim(0, T)

    legend_elements = [
        Line2D([0], [0], color='C0', linewidth=2.5, label="GT sensor series"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=8, label="Observed frames"),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.98), frameon=False)

    side_label = "terminal" if obs_side == "end" else "initial"
    fig.suptitle(f"{gt_label} - Sensor Time Series ({side_label} obs)", fontsize=11, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def generate_fig1_sensor_plots(sim_grids, gt_grid, sensor_locs, config, out_dir):
    """Generate all Figure 1 sensor overlay assets (GIFs + time-series PNGs)."""
    out_dir = Path(out_dir) / "Fig1sensor"
    out_dir.mkdir(parents=True, exist_ok=True)

    highlight_idx = select_highlighted_sensors(sensor_locs, n_highlight=3, seed=123)
    print(f"  Highlighted sensors: {highlight_idx}")

    T_gt = gt_grid.shape[0]
    obs_len = 1 if config.INFERENCE_MODE == "backward" else max(1, int(T_gt * config.GT_OBS_FRACTION))
    obs_side = "end" if config.INFERENCE_MODE == "backward" else "start"

    for i, yr in enumerate(config.SIM_YEARS[:len(sim_grids)]):
        sc = sim_grids[i]
        make_sensor_gif(sc, sensor_locs, highlight_idx,
                        title_prefix=f"Sim {yr} (NDSI)",
                        out_path=out_dir / f"sim_{yr}_sensors.gif")
        plot_sensor_timeseries_sim(sc, sensor_locs, highlight_idx,
                                  out_path=out_dir / f"timeseries_sim_{yr}.png",
                                  sim_label=f"Sim {yr}")

    obs_window = (T_gt - obs_len, T_gt - 1) if obs_side == "end" else (0, obs_len - 1)
    make_sensor_gif(gt_grid, sensor_locs, highlight_idx,
                    title_prefix=f"GT {config.GT_YEAR} (NDSI)",
                    out_path=out_dir / f"gt_{config.GT_YEAR}_sensors.gif",
                    obs_window=obs_window)

    plot_sensor_timeseries_gt(gt_grid, sensor_locs, highlight_idx, obs_len,
                             out_path=out_dir / f"timeseries_gt_{config.GT_YEAR}.png",
                             gt_label=f"GT {config.GT_YEAR}", obs_side=obs_side)


# NDSI snow animations

def _save_animation_formats(anim, base_dir, filename, fps, dpi, generate_videos=False):
    """Save animation as GIF (always) and optionally MP4."""
    base_dir = Path(base_dir)
    gif_dir = base_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    gif_path = gif_dir / f"{filename}.gif"
    try:
        anim.save(str(gif_path), writer='pillow', fps=fps, dpi=dpi // 2)
        print(f"    Saved: {gif_path}")
    except Exception as e:
        print(f"    Warning: could not save GIF: {e}")

    if generate_videos:
        import matplotlib.animation as mpl_anim
        video_dir = base_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = video_dir / f"{filename}.mp4"
        try:
            import shutil
            if shutil.which('ffmpeg') is not None:
                writer = mpl_anim.FFMpegWriter(fps=fps, bitrate=1800)
                anim.save(str(mp4_path), writer=writer, dpi=dpi)
                print(f"    Saved: {mp4_path}")
            else:
                print("    Note: ffmpeg not found, skipping MP4")
        except Exception as e:
            print(f"    Note: could not save MP4: {e}")


def save_gt_video(gt, config, base_dir, base_name="ground_truth"):
    """Single-panel GT animation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        return

    T = gt.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    def animate(t):
        ax.clear()
        ax.imshow(gt[t], cmap="Blues_r", vmin=-1.0, vmax=1.0)
        ax.set_title(f'Ground Truth (t={t}/{T-1})', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        return [ax]

    anim = animation.FuncAnimation(fig, animate, frames=range(T),
                                   interval=1000 // config.VIDEO_FPS, blit=False)
    _save_animation_formats(anim, base_dir, f"{base_name}_combined",
                            config.VIDEO_FPS, config.VIDEO_DPI, config.GENERATE_VIDEOS)
    plt.close(fig)


def save_ndsi_video(gt, pred, title_prefix, config, base_dir, base_name="reconstruction"):
    """3-panel (GT / Prediction / Error) animation for NDSI."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        return

    T = min(gt.shape[0], pred.shape[0])
    err_seq = pred[:T] - gt[:T]
    emax = max(float(np.abs(err_seq).max()), 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(title_prefix, fontsize=12, fontweight='bold')

    def animate(t):
        for ax in axes:
            ax.clear()
        axes[0].imshow(gt[t], cmap="Blues_r", vmin=-1.0, vmax=1.0)
        axes[0].set_title(f'Ground Truth (t={t})', fontsize=9)
        axes[1].imshow(pred[t], cmap="Blues_r", vmin=-1.0, vmax=1.0)
        axes[1].set_title(f'{title_prefix} Pred (t={t})', fontsize=9)
        rmse = float(np.sqrt(np.mean(err_seq[t] ** 2)))
        axes[2].imshow(np.abs(err_seq[t]), cmap="hot", vmin=0, vmax=emax)
        axes[2].set_title(f'|Error| (RMSE={rmse:.4f})', fontsize=9)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        return axes

    anim = animation.FuncAnimation(fig, animate, frames=range(T),
                                   interval=1000 // config.VIDEO_FPS, blit=False)
    _save_animation_formats(anim, base_dir, f"{base_name}_combined",
                            config.VIDEO_FPS, config.VIDEO_DPI, config.GENERATE_VIDEOS)
    plt.close(fig)

    # Individual panel GIFs
    for panel_name, data_seq, cmap, v0, v1 in [
        ("gt", gt[:T], "Blues_r", -1.0, 1.0),
        ("pred", pred[:T], "Blues_r", -1.0, 1.0),
    ]:
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4.5))

        def _anim(t, _d=data_seq, _ax=ax2, _cm=cmap, _v0=v0, _v1=v1):
            _ax.clear()
            _ax.imshow(_d[t], cmap=_cm, vmin=_v0, vmax=_v1)
            _ax.set_title(f'{panel_name.upper()} (t={t})', fontsize=9)
            _ax.set_xticks([]); _ax.set_yticks([])
            return [_ax]

        a2 = animation.FuncAnimation(fig2, _anim, frames=range(T),
                                     interval=1000 // config.VIDEO_FPS, blit=False)
        _save_animation_formats(a2, base_dir, f"{base_name}_{panel_name}",
                                config.VIDEO_FPS, config.VIDEO_DPI, config.GENERATE_VIDEOS)
        plt.close(fig2)

    # Error panel
    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 4.5))

    def _anim_err(t):
        ax3.clear()
        rmse = float(np.sqrt(np.mean(err_seq[t] ** 2)))
        ax3.imshow(np.abs(err_seq[t]), cmap="hot", vmin=0, vmax=emax)
        ax3.set_title(f'|Error| (RMSE={rmse:.4f}, t={t})', fontsize=9)
        ax3.set_xticks([]); ax3.set_yticks([])
        return [ax3]

    a3 = animation.FuncAnimation(fig3, _anim_err, frames=range(T),
                                 interval=1000 // config.VIDEO_FPS, blit=False)
    _save_animation_formats(a3, base_dir, f"{base_name}_error",
                            config.VIDEO_FPS, config.VIDEO_DPI, config.GENERATE_VIDEOS)
    plt.close(fig3)
