"""Plot the 33 input features of an episode, laid out the way they group.

    joint_positions             12   4 legs x 3 joints
    estimated_joint_velocities  12   4 legs x 3 joints
    lin_vels_in_b                3
    angular_vel                  3
    gravity in body frame        3

Joint rows are shaded grey where that foot is in stance, so you can read the
per-leg signals against the gait phase they belong to.

Usage:
    python visualize_features.py                        # first episode, 20 s from the start
    python visualize_features.py -f 2026-08-03_19-36-12.h5 -s 40 -d 15
    python visualize_features.py --all                  # every episode, saved as PNG
"""

import argparse
import glob
import importlib.util
import os

import matplotlib


def _select_backend():
    """Choose a GUI backend that can actually be imported; return True if none.

    matplotlib.use() only records the name — the backend module is imported
    lazily when the first figure is created, so a missing toolkit surfaces as a
    traceback out of plt.subplots() rather than out of use(). Probing the
    toolkit up front keeps the fallback to Agg (save PNGs) working.
    """
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return True
    for module, backend in (("tkinter", "TkAgg"), ("PyQt6", "QtAgg"),
                            ("PySide6", "QtAgg"), ("PyQt5", "QtAgg")):
        if importlib.util.find_spec(module) is not None:
            matplotlib.use(backend)
            return False
    return True


HEADLESS = _select_backend()
if HEADLESS:
    matplotlib.use("Agg")

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import blended_transform_factory

from dataloader import load_episode

LEG_NAMES = ["FR", "FL", "RR", "RL"]
JOINT_NAMES = ["hip", "thigh", "calf"]
JOINT_COLORS = ["tab:blue", "tab:orange", "tab:green"]
AXIS_NAMES = ["x", "y", "z"]
AXIS_COLORS = ["tab:red", "tab:blue", "tab:green"]
FS = 50.0  # sample rate of the interp groups, Hz

# Feature layout inside the 33-dim vector, mirroring dataloader.load_episode.
SLICE_QPOS = slice(0, 12)
SLICE_QVEL = slice(12, 24)
SLICE_LINVEL = slice(24, 27)
SLICE_ANGVEL = slice(27, 30)
SLICE_GRAVITY = slice(30, 33)


def _shade_stance(ax, t, stance):
    """Grey background wherever the foot is on the ground."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(t, 0, 1, where=stance, transform=trans, color="0.85",
                    step="mid", linewidth=0, zorder=0)


def _plot_group(ax, t, data, names, colors, ylabel):
    for i, (name, color) in enumerate(zip(names, colors)):
        ax.plot(t, data[:, i], color=color, lw=1.0, label=name)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(loc="upper right", fontsize=7, ncol=len(names), framealpha=0.8)
    ax.grid(alpha=0.25, lw=0.5)


def plot_episode(path, start_s=0.0, dur_s=20.0, group="interp2"):
    """Build one figure showing every input channel over a time slice."""
    feats, targets, times = load_episode(path, group)
    with h5py.File(path, "r") as f:
        stance_all = f[group]["foot_mask"][...] == 0  # mask 0 == foot on the ground

    i0 = int(start_s * FS)
    i1 = min(len(feats), i0 + int(dur_s * FS)) if dur_s else len(feats)
    if i0 >= len(feats):
        raise ValueError(
            f"start {start_s}s is past the end of {os.path.basename(path)} "
            f"({len(feats) / FS:.1f}s long)"
        )
    feats, stance_all = feats[i0:i1], stance_all[i0:i1]
    t = (times[i0:i1] - times[i0]) / 1000.0  # seconds from the window start

    qpos = feats[:, SLICE_QPOS].reshape(-1, 4, 3)
    qvel = feats[:, SLICE_QVEL].reshape(-1, 4, 3)

    fig, axes = plt.subplots(
        7, 2, figsize=(15, 13), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.1, 1.1, 1.1]},
    )

    # rows 0-3: per-leg joint angles (left column) and velocities (right column)
    for leg in range(4):
        for col, (arr, unit) in enumerate(((qpos, "rad"), (qvel, "rad/s"))):
            ax = axes[leg, col]
            _shade_stance(ax, t, stance_all[:, leg])
            _plot_group(ax, t, arr[:, leg, :], JOINT_NAMES, JOINT_COLORS,
                        f"{LEG_NAMES[leg]}\n[{unit}]")
            if leg == 0:
                ax.set_title(["joint_positions (12)",
                              "estimated_joint_velocities (12)"][col], fontsize=10)

    # rows 4-6: body-level channels, each spanning the full width
    body = [
        (feats[:, SLICE_LINVEL], "lin_vels_in_b\n[m/s]"),
        (feats[:, SLICE_ANGVEL], "angular_vel\n[rad/s]"),
        (feats[:, SLICE_GRAVITY], "gravity_b\n[m/s^2]"),
    ]
    for row, (data, ylabel) in enumerate(body, start=4):
        gs = axes[row, 0].get_gridspec()
        for ax in axes[row, :]:
            ax.remove()
        ax = fig.add_subplot(gs[row, :])
        axes[row, 0] = ax
        _plot_group(ax, t, data, AXIS_NAMES, AXIS_COLORS, ylabel)
        ax.set_xlim(t[0], t[-1])

    axes[6, 0].set_xlabel("time [s]")
    fig.suptitle(
        f"{os.path.basename(path)}   {group}   "
        f"t = {start_s:.1f}-{start_s + (t[-1] - t[0]):.1f}s   "
        f"({i1 - i0} frames)   grey = stance",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(here, "dataset"))
    p.add_argument("-f", "--file", help="episode filename (default: the first one)")
    p.add_argument("-s", "--start", type=float, default=0.0, help="start time [s]")
    p.add_argument("-d", "--dur", type=float, default=20.0,
                   help="window length [s], 0 for the whole episode")
    p.add_argument("-g", "--group", default="interp2", choices=["interp", "interp2"])
    p.add_argument("--all", action="store_true", help="plot every episode")
    p.add_argument("--out", default=None, help="directory for PNGs (implies saving)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if not files:
        raise SystemExit(f"no .h5 files in {args.data_dir}")
    if args.all:
        selected = files
    elif args.file:
        selected = [f for f in files if os.path.basename(f) == os.path.basename(args.file)]
        if not selected:
            raise SystemExit(f"{args.file} not found in {args.data_dir}")
    else:
        selected = files[:1]

    if HEADLESS or args.out:
        outdir = args.out or os.path.join(args.data_dir, "figures")
        os.makedirs(outdir, exist_ok=True)
        for path in selected:
            name = os.path.basename(path)
            fig = plot_episode(path, args.start, args.dur, args.group)
            # save and close one at a time: whole-episode figures are large and
            # holding all of them open runs the process out of memory
            fig.savefig(os.path.join(outdir, name.replace(".h5", "_features.png")), dpi=100)
            plt.close(fig)
            print(f"  {name}")
        print(f"saved {len(selected)} figure(s) to {outdir}")
        if HEADLESS:
            print("set DISPLAY (e.g. `export DISPLAY=:1`) to get interactive windows")
    else:
        for path in selected:
            plot_episode(path, args.start, args.dur, args.group)
        plt.show()


if __name__ == "__main__":
    main()
