"""Non-learned reference points, measured on exactly the split train.py uses.

Without these, "val MSE 0.008" is a number with no meaning. Three anchors:

    predict the training mean   the floor any model must clear to be worth running
    k-nearest neighbour         what pure memorisation of the training set buys
    8-bit quantisation noise    the ceiling nothing can pass, since foot_sensors
                                is recorded on a 0.005 grid (see Step 2)

Run this before reading any training result.
"""

import os

import numpy as np
import torch

from train import build_loaders

QUANT_STEP = 0.005  # measured from lowstate/foot_sensors: 205 distinct values


def _tensors(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.reshape(len(x), -1))
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def knn_mse(xt, yt, xq, yq, k=30, n_train=20000, n_query=3000, seed=0, device="cuda"):
    """k-NN regression: average the targets of the k closest training windows."""
    g = torch.Generator().manual_seed(seed)
    ti = torch.randperm(len(xt), generator=g)[:n_train]
    qi = torch.randperm(len(xq), generator=g)[:n_query]
    xt, yt = xt[ti].to(device), yt[ti].to(device)
    xq, yq = xq[qi].to(device), yq[qi].to(device)

    tn = (xt ** 2).sum(1)
    out = []
    for i in range(0, len(xq), 256):
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b; the ||a||^2 term is constant
        # per query row, so it cannot change which neighbour is nearest.
        d = tn[None, :] - 2.0 * xq[i:i + 256] @ xt.T
        out.append(yt[d.topk(k, dim=1, largest=False).indices].mean(1))
    return float(((torch.cat(out) - yq) ** 2).mean())


def report(name, mse, var):
    print(f"  {name:<34} MSE {mse:.5f}   RMSE {mse ** 0.5:.4f}   "
          f"R2 {1 - mse / var:+.4f}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for history in (1, 5, 20, 40):
        train_loader, val_loaders, _ = build_loaders(
            os.path.join(here, "dataset"), history, 1024)
        xt, yt = _tensors(train_loader)
        print(f"\nhistory={history}   train {len(xt)} windows, {xt.shape[1]} dims")

        for split, loader in val_loaders.items():
            xq, yq = _tensors(loader)
            var = float(yq.var(unbiased=False))
            print(f" val/{split}   (variance of targets = {var:.5f})")
            report("predict the training mean", float(((yt.mean(0) - yq) ** 2).mean()), var)
            report(f"{30}-NN", knn_mse(xt, yt, xq, yq, device=device), var)
            report("8-bit quantisation floor", QUANT_STEP ** 2 / 12, var)
