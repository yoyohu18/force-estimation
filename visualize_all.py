"""Render every channel of every recording in dataset/ as PNG charts.

One h5 file holds five groups recorded at different rates:

    lowstate    500 Hz   raw joints, IMU, foot sensors
    highstate   ~300 Hz  onboard state estimate
    interp       50 Hz   everything resampled onto a common clock
    interp2      50 Hz   the validity-filtered version of interp
    lidar        ~10 Hz  lidar odometry pose

Each group is split into up to three figures, so a panel never mixes units:

    <group>_joints.png   the 12-wide arrays, one row per leg, one column per array
    <group>_body.png     the 3-wide vectors and the quaternions
    <group>_feet.png     foot_sensors / foot_mask per leg, plus validity

Joint and foot panels are shaded grey during stance (foot_mask == 0), so the
per-leg signals can be read against the gait phase they belong to.

Usage:
    python visualize_all.py                       # every file, every group
    python visualize_all.py -f 2026-08-03_19-43-13.h5
    python visualize_all.py -g interp2 lowstate --dpi 140
    python visualize_all.py --overview-only
"""

import argparse
import glob
import os

import h5py
import matplotlib

matplotlib.use("Agg")  # output is PNGs, so never wait on a GUI toolkit

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

# Okabe-Ito, in fixed order: assigned by position, never cycled per figure.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

LEG_NAMES = ["FR", "FL", "RR", "RL"]
JOINT_NAMES = ["hip", "thigh", "calf"]
AXIS_NAMES = ["x", "y", "z"]
QUAT_NAMES = ["w", "x", "y", "z"]  # the dataset stores w first (see dataloader)

GROUPS = ["lowstate", "highstate", "interp", "interp2", "lidar"]
STANCE_SHADE = "0.86"
GRID = dict(alpha=0.25, lw=0.5)


# --------------------------------------------------------------------------
# small helpers


def _time_s(times_ms):
    """Milliseconds since the epoch -> seconds since the start of the episode."""
    t = np.asarray(times_ms, dtype=np.float64)
    return (t - t[0]) / 1000.0


def _stride(n, max_points):
    """Decimation step that keeps a long trace under max_points samples."""
    if not max_points or n <= max_points:
        return 1
    return int(np.ceil(n / max_points))


def _break_gaps(t, y, max_gap):
    """Insert NaN rows at time gaps so the line breaks instead of jumping.

    interp2 drops invalid stretches, so consecutive samples can be seconds
    apart; without this the plot draws a straight line across the hole and it
    reads as real data.
    """
    cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
    if not len(cuts):
        return t, y
    t = np.insert(t, cuts, np.nan)
    y = np.insert(y, cuts, np.nan, axis=0)
    return t, y


def _shade_stance(ax, t, stance):
    """Grey background wherever that foot is on the ground."""
    if stance is None:
        return
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(t, 0, 1, where=stance, transform=trans, color=STANCE_SHADE,
                    step="mid", linewidth=0, zorder=0)


def _plot_lines(ax, t, data, names, ylabel, legend=True, rasterized=False):
    for i, name in enumerate(names):
        ax.plot(t, data[:, i], color=PALETTE[i], lw=1.0, label=name,
                rasterized=rasterized, solid_joinstyle="round")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(**GRID)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend and len(names) > 1:
        ax.legend(loc="upper right", fontsize=6.5, ncol=len(names), framealpha=0.85,
                  handlelength=1.2, columnspacing=0.9, borderpad=0.3)


def _kind(name, arr):
    """Which figure a dataset belongs in, from its width and name."""
    if arr.ndim == 1:
        return "scalar"
    w = arr.shape[1]
    if w == 12:
        return "joints"
    if w == 4 and "orient" in name:
        return "quat"
    if w == 4:
        return "feet"
    if w == 3:
        return "vec3"
    return "other"


def _read_group(f, group, max_points):
    """Load one group as {name: array}, decimated, plus the time axis."""
    g = f[group]
    n = g["times_ms"].shape[0]
    s = _stride(n, max_points)
    data = {k: np.asarray(g[k][::s]) for k in g.keys() if k != "times_ms"}
    return data, _time_s(g["times_ms"][::s]), s, n


def _title(fig, path, group, what, n, stride, extra=""):
    tag = f"   (every {stride}th sample of {n})" if stride > 1 else f"   ({n} samples)"
    fig.suptitle(f"{os.path.basename(path)}   {group}   {what}{tag}{extra}",
                 fontsize=11)


# --------------------------------------------------------------------------
# per-group figures


def fig_joints(path, group, data, t, stance, n, stride):
    """12-wide arrays: rows are legs, columns are arrays, 3 joint lines each."""
    names = sorted(k for k, v in data.items() if _kind(k, v) == "joints")
    if not names:
        return None

    fig, axes = plt.subplots(4, len(names), figsize=(4.6 * len(names) + 1.2, 9),
                             sharex=True, squeeze=False)
    for col, name in enumerate(names):
        arr = data[name].reshape(-1, 4, 3)
        for leg in range(4):
            ax = axes[leg, col]
            _shade_stance(ax, t, stance[:, leg] if stance is not None else None)
            tb, yb = _break_gaps(t, arr[:, leg, :], 1.0)
            _plot_lines(ax, tb, yb, JOINT_NAMES,
                        f"{LEG_NAMES[leg]}" if col == 0 else "",
                        legend=(leg == 0), rasterized=len(t) > 20000)
            if leg == 0:
                ax.set_title(name, fontsize=9)
        axes[-1, col].set_xlabel("time [s]")
    axes[0, 0].set_xlim(t[0], t[-1])
    _title(fig, path, group, "joint arrays (12) — grey = stance", n, stride)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_body(path, group, data, t, n, stride):
    """3-wide vectors and quaternions, one panel each, stacked on a shared clock."""
    vecs = sorted(k for k, v in data.items() if _kind(k, v) == "vec3")
    quats = sorted(k for k, v in data.items() if _kind(k, v) == "quat")
    names = vecs + quats
    if not names:
        return None

    ncol = 2 if len(names) > 5 else 1
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.0 * ncol, 1.7 * nrow + 1.0),
                             sharex=True, squeeze=False)
    flat = axes.T.reshape(-1)  # fill column by column
    for ax, name in zip(flat, names):
        labels = QUAT_NAMES if name in quats else AXIS_NAMES
        tb, yb = _break_gaps(t, data[name], 1.0)
        _plot_lines(ax, tb, yb, labels, name, rasterized=len(t) > 20000)
    for ax in flat[len(names):]:
        ax.set_visible(False)
    for col in range(ncol):
        last = min(nrow, len(names) - col * nrow) - 1
        if last >= 0:
            axes[last, col].set_xlabel("time [s]")
    flat[0].set_xlim(t[0], t[-1])
    _title(fig, path, group, "body vectors & orientation", n, stride)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_feet(path, group, data, t, stance, n, stride):
    """Per-leg contact channels: force trace over the stance shading, plus validity."""
    feet = sorted(k for k, v in data.items() if _kind(k, v) == "feet")
    scalars = sorted(k for k, v in data.items() if _kind(k, v) == "scalar")
    if not feet:
        return None

    nrow = 4 + len(scalars)
    fig, axes = plt.subplots(nrow, 1, figsize=(13, 1.5 * nrow + 1.0), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    for leg in range(4):
        ax = axes[leg]
        _shade_stance(ax, t, stance[:, leg] if stance is not None else None)
        for i, name in enumerate(feet):
            tb, yb = _break_gaps(t, data[name][:, [leg]], 1.0)
            ax.plot(tb, yb[:, 0], color=PALETTE[i], lw=1.0, label=name,
                    rasterized=len(t) > 20000)
        ax.set_ylabel(LEG_NAMES[leg], fontsize=9)
        ax.grid(**GRID)
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if leg == 0:
            handles = [Line2D([], [], color=PALETTE[i], lw=1.5) for i in range(len(feet))]
            handles.append(plt.Rectangle((0, 0), 1, 1, color=STANCE_SHADE))
            ax.legend(handles, feet + ["stance (mask == 0)"], loc="upper right",
                      fontsize=7, ncol=len(feet) + 1, framealpha=0.85)
    for row, name in enumerate(scalars, start=4):
        ax = axes[row]
        ax.step(t, np.asarray(data[name], dtype=float), where="mid",
                color=PALETTE[3], lw=1.0, rasterized=len(t) > 20000)
        ax.set_ylabel(name, fontsize=8)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(**GRID)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time [s]")
    axes[0].set_xlim(t[0], t[-1])
    _title(fig, path, group, "foot channels", n, stride)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_file(path, groups, outdir, max_points, dpi):
    """Write every figure for one episode; returns the paths written."""
    written = []
    with h5py.File(path, "r") as f:
        for group in groups:
            if group not in f:
                continue
            data, t, stride, n = _read_group(f, group, max_points)
            mask = data.get("foot_mask")
            stance = (np.asarray(mask) == 0) if mask is not None else None

            figs = {
                "joints": fig_joints(path, group, data, t, stance, n, stride),
                "body": fig_body(path, group, data, t, n, stride),
                "feet": fig_feet(path, group, data, t, stance, n, stride),
            }
            for what, fig in figs.items():
                if fig is None:
                    continue
                out = os.path.join(outdir, f"{group}_{what}.png")
                fig.savefig(out, dpi=dpi)
                plt.close(fig)
                written.append(out)
    return written


# --------------------------------------------------------------------------
# dataset-level overview


def fig_overview(files, group="interp"):
    """One page summarising all episodes: where they went and what they contain."""
    eps = []
    for path in files:
        with h5py.File(path, "r") as f:
            g = f[group]
            eps.append({
                "name": os.path.basename(path).replace(".h5", "").split("_")[-1],
                "t": _time_s(g["times_ms"]),
                "pos": np.asarray(g["positions"]),
                "vel": np.asarray(g["lin_vels_in_b"]),
                "mask": np.asarray(g["foot_mask"]),
                "sens": np.asarray(g["foot_sensors"]),
                "tau": np.asarray(g["joint_torques"]).reshape(-1, 4, 3),
                "n_low": f["lowstate"]["times_ms"].shape[0],
            })

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    ax = axes[0, 0]
    for i, e in enumerate(eps):
        ax.plot(e["pos"][:, 0], e["pos"][:, 1], lw=1.0, color=PALETTE[i % len(PALETTE)],
                alpha=0.85)
        ax.plot(e["pos"][0, 0], e["pos"][0, 1], "o", ms=4,
                color=PALETTE[i % len(PALETTE)])
    ax.set_title("walked paths (dot = start)", fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(**GRID)

    ax = axes[0, 1]
    dur = [e["t"][-1] for e in eps]
    y = np.arange(len(eps))
    ax.barh(y, dur, color=PALETTE[0], height=0.62)
    for i, d in enumerate(dur):
        ax.text(d + 2, i, f"{d:.0f}s", va="center", fontsize=7.5, color="0.35")
    ax.set_yticks(y, [e["name"] for e in eps], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("duration [s]"); ax.set_title(
        f"episode length  (total {sum(dur) / 60:.1f} min, "
        f"{sum(e['n_low'] for e in eps):,} lowstate frames)", fontsize=10)
    ax.grid(axis="x", **GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[0, 2]
    stance = np.array([[(e["mask"][:, l] == 0).mean() for l in range(4)] for e in eps])
    for l in range(4):
        ax.plot(y, stance[:, l], "o-", ms=4, lw=1.2, color=PALETTE[l], label=LEG_NAMES[l])
    ax.set_xticks(y, [e["name"] for e in eps], rotation=90, fontsize=7)
    ax.set_ylabel("fraction of frames in stance"); ax.set_ylim(0, 1)
    ax.set_title("duty factor per leg", fontsize=10)
    ax.legend(fontsize=7.5, ncol=4); ax.grid(**GRID)

    ax = axes[1, 0]
    sens = np.concatenate([e["sens"] for e in eps])
    msk = np.concatenate([e["mask"] for e in eps])
    bins = np.linspace(0, max(np.percentile(sens, 99.5), 1e-3), 60)
    ax.hist(sens[msk == 0].ravel(), bins=bins, color=PALETTE[0], alpha=0.75,
            label="stance (mask 0)")
    ax.hist(sens[msk == 1].ravel(), bins=bins, color=PALETTE[1], alpha=0.75,
            label="swing (mask 1)")
    ax.set_yscale("log"); ax.set_xlabel("foot_sensors"); ax.set_ylabel("frames")
    ax.set_title("contact-force distribution, all episodes", fontsize=10)
    ax.legend(fontsize=8); ax.grid(**GRID)

    ax = axes[1, 1]
    speed = [np.linalg.norm(e["vel"], axis=1) for e in eps]
    bp = ax.boxplot(speed, vert=True, widths=0.6, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set(facecolor=PALETTE[2], alpha=0.55, edgecolor="0.35")
    for part in ("medians", "whiskers", "caps"):
        for a in bp[part]:
            a.set(color="0.35")
    ax.set_xticks(y + 1, [e["name"] for e in eps], rotation=90, fontsize=7)
    ax.set_ylabel("|lin_vels_in_b| [m/s]")
    ax.set_title("body speed per episode", fontsize=10)
    ax.grid(axis="y", **GRID)

    ax = axes[1, 2]
    tau = np.concatenate([e["tau"] for e in eps])
    pos = np.arange(12).reshape(4, 3)
    for j in range(3):
        vals = [tau[:, l, j] for l in range(4)]
        ax.boxplot(vals, positions=pos[:, j], widths=0.7, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=PALETTE[j], alpha=0.55, edgecolor="0.35"),
                   medianprops=dict(color="0.25"),
                   whiskerprops=dict(color="0.35"), capprops=dict(color="0.35"))
    ax.set_xticks(pos.ravel(), [f"{LEG_NAMES[l]}\n{JOINT_NAMES[j]}"
                                for l in range(4) for j in range(3)], fontsize=6.5)
    ax.set_ylabel("joint_torques [Nm]")
    ax.set_title("torque spread per joint, all episodes", fontsize=10)
    ax.grid(axis="y", **GRID)

    for ax in axes.ravel():
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"dataset overview — {len(eps)} episodes, group '{group}'", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--data-dir", default=os.path.join(here, "dataset"))
    p.add_argument("-o", "--out", default=None, help="output dir (default <data>/figures)")
    p.add_argument("-f", "--file", nargs="*", help="only these episodes")
    p.add_argument("-g", "--group", nargs="*", default=GROUPS, choices=GROUPS)
    p.add_argument("--max-points", type=int, default=30000,
                   help="decimate traces longer than this (0 disables)")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--overview-only", action="store_true")
    p.add_argument("--no-overview", action="store_true")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if not files:
        raise SystemExit(f"no .h5 files in {args.data_dir}")
    if args.file:
        want = {os.path.basename(x) for x in args.file}
        files = [f for f in files if os.path.basename(f) in want]
        if not files:
            raise SystemExit(f"none of {sorted(want)} found in {args.data_dir}")

    root = args.out or os.path.join(args.data_dir, "figures")
    os.makedirs(root, exist_ok=True)
    total = 0

    if not args.overview_only:
        for path in files:
            stem = os.path.basename(path).replace(".h5", "")
            outdir = os.path.join(root, stem)
            os.makedirs(outdir, exist_ok=True)
            written = plot_file(path, args.group, outdir, args.max_points, args.dpi)
            total += len(written)
            print(f"{stem}: {len(written)} figures -> {outdir}")

    if not args.no_overview:
        fig = fig_overview(files)
        out = os.path.join(root, "dataset_overview.png")
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        total += 1
        print(f"overview -> {out}")

    print(f"\n{total} PNG(s) written under {root}")


if __name__ == "__main__":
    main()
