import h5py
import numpy as np
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

import matplotlib.pyplot as plt


LEG_NAMES = ["FR", "FL", "RR", "RL"] 


def visualize_foot_mask(data, title=""):
    foot_mask = np.asarray(data['foot_mask'])          # (T, 4)
    tau       = np.asarray(data['joint_torques'])      # (T, 12)

    T = tau.shape[0]
    tau = tau.reshape(T, 4, 3)                          # (T, 4, 3) -> per-leg xyz
    tau_norm = np.linalg.norm(tau, axis=-1)            # (T, 4)     per-leg torque norm

    # normalize each leg's norm to [0,1] just for overlay readability
    tn = tau_norm / (tau_norm.max(axis=0, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(tn[:, i], color="tab:green", lw=1.2, label="||tau|| (norm)")
        ax.plot(foot_mask[:, i], color="tab:gray", lw=1.0, ls="--", label="foot_mask")
        ax.set_ylabel(LEG_NAMES[i])
        ax.set_ylim(-0.05, 1.15)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestep")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    mainpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    datalist = sorted(f for f in os.listdir(mainpath) if f.endswith(".h5"))

    figs = []
    for item in datalist:
        datapath = os.path.join(mainpath, item)
        print(item)
        with h5py.File(datapath, "r") as f:
            data_h5 = f['interp']
            data = {key: np.asarray(data_h5[key][...]) for key in data_h5.keys()}
            
            # Write your own dataloader first (use LLM for this, you do not need to understand the data loader)
            # Use 
            # joint_positions
            # estimated_joint_velocities
            # lin_vels_in_b
            # angular_vel
            # use orientation to estimate the gravity in body frame
        figs.append((item, visualize_foot_mask(data, title=item)))

    if HEADLESS:
        outdir = os.path.join(mainpath, "figures")
        os.makedirs(outdir, exist_ok=True)
        for item, fig in figs:
            fig.savefig(os.path.join(outdir, item.replace(".h5", ".png")), dpi=100)
        print(f"[no display] saved {len(figs)} figures to {outdir}")
        print("set DISPLAY (e.g. `export DISPLAY=:1`) to get interactive windows instead")
    else:
        # Show every figure at once, then block until all windows are closed
        plt.show()