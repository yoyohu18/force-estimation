"""Diagnostics that a single MSE number hides.

MSE collapses 38040 errors into one scalar, and several very different
failures land on the same scalar: a prediction that is half a stride late, one
that flattens every impact peak, and one that is simply biased on one foot can
all score alike. This script takes the trained checkpoint apart along the axes
that matter for a contact estimator:

    per foot            is one leg much worse than the others
    stance vs swing     swing is nearly constant and easy; stance is the job
    peak tracking       MSE tolerates flattened peaks, contact does not
    RMSE / MAE ratio    1.25 for Gaussian error; higher means heavy tails
    sum of forces       the four channels must add to body weight (Step 2)
    negative outputs    physically impossible, so any is a defect

Usage:
    python evaluate.py --ckpt checkpoints/mlp.pt
"""

import argparse
import os

import h5py
import numpy as np
import torch

from dataset import TRAIN_SPLIT, VAL_FILES, VAL_SPLIT, ForceWindowDataset
from model import MLP
from modes import QUAD_STAND, QUAD_WALK, REDUCED, label_modes

FEET = ["FR", "FL", "RR", "RL"]
GAUSSIAN_RATIO = (np.pi / 2) ** 0.5  # RMSE/MAE for zero-mean Gaussian error


@torch.no_grad()
def predict(model, ds, device, batch=4096):
    model.eval()
    out = []
    for i in range(0, len(ds), batch):
        x = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))])
        out.append(model(x.to(device)).cpu())
    return torch.cat(out).numpy()


def targets_and_mask(ds, files):
    """Ground-truth force and foot_mask aligned with ds.index."""
    masks = []
    for path in files:
        with h5py.File(path, "r") as f:
            masks.append(f["interp2"]["foot_mask"][...])
    ep, end = ds.index[:, 0], ds.index[:, 1]
    y = np.stack([ds.episodes[e][1][t] for e, t in zip(ep, end)])
    m = np.stack([masks[e][t] for e, t in zip(ep, end)])
    return y, m


def report(name, pred, y, mask):
    err = pred - y
    mse = float((err ** 2).mean())
    mae = float(np.abs(err).mean())
    rmse = mse ** 0.5
    var = float(y.var())

    print(f"\n=== {name} ===   {len(y)} windows")
    print(f"  MSE {mse:.5f}   RMSE {rmse:.4f}   MAE {mae:.4f}   "
          f"R2 {1 - mse / var:+.4f}")
    print(f"  RMSE/MAE {rmse / mae:.2f}  (Gaussian would be {GAUSSIAN_RATIO:.2f}"
          f"{' -> heavy tails, a few large errors dominate'if rmse / mae > 1.35 else ''})")

    print("  per foot:")
    for k, foot in enumerate(FEET):
        e = err[:, k]
        print(f"    {foot}  RMSE {float((e ** 2).mean()) ** 0.5:.4f}  "
              f"bias {float(e.mean()):+.4f}  "
              f"R2 {1 - float((e ** 2).mean()) / float(y[:, k].var()):+.4f}")

    stance = mask == 0  # Step 2: mask == 1 is swing
    for label, sel in (("stance", stance), ("swing", ~stance)):
        e = err[sel]
        print(f"  {label:<7} {sel.mean() * 100:5.1f}% of samples   "
              f"RMSE {float((e ** 2).mean()) ** 0.5:.4f}   "
              f"bias {float(e.mean()):+.4f}")

    # peaks: MSE happily tolerates a model that shaves every impact
    hi = y >= np.quantile(y, 0.95)
    print(f"  top 5% of true force (>= {np.quantile(y, 0.95):.3f}):"
          f"  mean true {y[hi].mean():.3f}  mean pred {pred[hi].mean():.3f}"
          f"   under-shoot {100 * (1 - pred[hi].mean() / y[hi].mean()):.1f}%")

    # Step 2: the four sensors sum to body weight regardless of which feet carry it
    st, sp = y.sum(1), pred.sum(1)
    print(f"  sum of 4 forces:  true {st.mean():.3f}+-{st.std():.3f}   "
          f"pred {sp.mean():.3f}+-{sp.std():.3f}   "
          f"corr {np.corrcoef(st, sp)[0, 1]:+.3f}")
    print(f"  negative predictions: {100 * (pred < 0).mean():.1f}%   "
          f"most negative {pred.min():+.3f}")
    return {"mse": mse, "pred": pred, "y": y, "mask": mask}


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 8))
    axes = np.atleast_2d(axes).reshape(2, n)

    for j, (name, r) in enumerate(results.items()):
        pred, y, mask = r["pred"], r["y"], r["mask"]

        # a slice of the FR channel, with stance shaded
        ax = axes[0, j]
        s = slice(1500, 1900)
        t = np.arange(s.stop - s.start) / 50.0
        ax.plot(t, y[s, 0], lw=1.4, label="true")
        ax.plot(t, pred[s, 0], lw=1.2, label="pred")
        ax.fill_between(t, -0.05, 1.05, where=mask[s, 0] == 0, color="0.85",
                        zorder=0, label="stance")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{name}: FR force, 8 s")
        ax.set_xlabel("time [s]")
        ax.legend(loc="upper right", fontsize=8)

        # predicted vs true; slope < 1 means the range is being compressed
        ax = axes[1, j]
        idx = np.random.default_rng(0).choice(y.size, min(6000, y.size), replace=False)
        ax.scatter(y.ravel()[idx], pred.ravel()[idx], s=2, alpha=0.2)
        lim = [-0.1, 1.1]
        ax.plot(lim, lim, "k--", lw=1, label="ideal")
        slope = np.polyfit(y.ravel(), pred.ravel(), 1)[0]
        ax.plot(lim, np.polyval(np.polyfit(y.ravel(), pred.ravel(), 1), lim),
                "r-", lw=1, label=f"fit, slope {slope:.2f}")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.set_title(f"{name}: predicted vs true")
        ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/mlp.pt")
    p.add_argument("--history", type=int, default=20)
    p.add_argument("--hidden", type=int, default=64)
    args = p.parse_args()

    import glob

    here = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(here, args.ckpt), weights_only=False)

    files = sorted(glob.glob(os.path.join(here, "dataset", "*.h5")))
    train_files = [f for f in files if os.path.basename(f) not in set(VAL_FILES)]
    quad_files = [f for f in files if os.path.basename(f) in set(VAL_FILES)]

    val_sets = {
        "quad": (ForceWindowDataset(quad_files, args.history,
                                    modes=(QUAD_WALK, QUAD_STAND), stats=ck["stats"]),
                 quad_files),
        "reduced": (ForceWindowDataset(train_files, args.history, modes=(REDUCED,),
                                       split=VAL_SPLIT, stats=ck["stats"]),
                    train_files),
    }

    model = MLP(history=args.history, hidden=args.hidden).to(device)
    model.load_state_dict(ck["model"])
    print(f"checkpoint: epoch {ck['epoch']}   monitored val MSE {ck['val_mse']:.5f}")

    results = {}
    for name, (ds, src) in val_sets.items():
        y, mask = targets_and_mask(ds, src)
        results[name] = report(name, predict(model, ds, device), y, mask)

    plot(results, os.path.join(here, "dataset", "figures", "eval_mlp.png"))
