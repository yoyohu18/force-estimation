"""Training loop: forward, loss, backward, step — plus validation and early stop.

The whole of learning is these five lines, repeated:

    pred = model(x)          # forward, and build the graph on the way
    loss = loss_fn(pred, y)  # one scalar saying how wrong we are
    optimizer.zero_grad()    # .backward() accumulates, so clear first
    loss.backward()          # walk the graph backwards, fill every .grad
    optimizer.step()         # theta <- theta - lr * grad (Adam's version of it)

Everything else in this file is bookkeeping: which numbers to report, when to
stop, and which weights to keep.

Reference points this has to beat, measured on the same split:
    predict the training mean   MSE 0.01625
    k-NN, history=20            MSE 0.01053
    8-bit quantisation floor    MSE 0.0000021
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TRAIN_SPLIT, VAL_FILES, VAL_SPLIT, ForceWindowDataset
from model import MLP, count_parameters, negative_fraction
from modes import QUAD_STAND, QUAD_WALK, REDUCED


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()  # tells dropout/batchnorm to behave as in training
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # .item() pulls the scalar off the GPU and drops the graph with it;
        # accumulating `loss` itself would keep every batch's graph alive.
        total += loss.item() * len(x)
        n += len(x)
    return total / n


@torch.no_grad()  # no graph, no .grad — validation never updates anything
def evaluate(model, loader, device):
    model.eval()
    se, ae, n, neg = 0.0, 0.0, 0, 0.0
    ys = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        se += ((pred - y) ** 2).sum().item()
        ae += (pred - y).abs().sum().item()
        neg += (pred < 0).sum().item()
        n += y.numel()
        ys.append(y)
    y_all = torch.cat(ys)
    mse = se / n
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": ae / n,
        # R^2 against the variance of *this* split: 0 means "no better than
        # always predicting this split's mean", 1 means perfect.
        "r2": 1.0 - mse / y_all.var(unbiased=False).item(),
        "neg_frac": neg / n,
    }


def fit(model, train_loader, val_loaders, device, epochs=200, lr=1e-3,
        patience=20, ckpt="checkpoints/mlp.pt", stats=None, log_every=10,
        monitor="quad"):
    """Train until the monitored validation loss stops improving, keep the best weights.

    `monitor` matters more than it looks. Averaging four-legged and reduced-
    contact error into one number lets whichever split is worse decide when to
    stop: monitoring "all" here stopped at epoch 1 while four-legged error was
    still falling, because the reduced split was diverging at the same time.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        metrics = {name: evaluate(model, dl, device) for name, dl in val_loaders.items()}
        val_loss = metrics[monitor]["mse"]
        history.append({"epoch": epoch, "train": train_loss,
                        **{f"{k}_mse": v["mse"] for k, v in metrics.items()}})

        if val_loss < best:
            best, best_epoch = val_loss, epoch
            # .cpu() clones off the GPU so later steps cannot mutate the copy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % log_every == 0 or epoch == 1:
            parts = "  ".join(f"{k} {v['mse']:.5f}" for k, v in metrics.items())
            print(f"  epoch {epoch:>4}  train {train_loss:.5f}   val: {parts}"
                  f"{'   *' if epoch == best_epoch else ''}")

        if epoch - best_epoch >= patience:
            print(f"  early stop: no improvement for {patience} epochs")
            break

    model.load_state_dict(best_state)  # roll back to the best epoch, not the last
    print(f"\nbest epoch {best_epoch}  val MSE {best:.5f}   "
          f"({time.perf_counter() - t0:.1f}s)")

    if ckpt:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save({"model": best_state, "stats": stats,
                    "epoch": best_epoch, "val_mse": best,
                    "history": history, "config": vars(model)}, ckpt)
        print(f"saved -> {ckpt}")
    return history


def build_loaders(data_dir, history, batch_size):
    """Two holdout strategies, one per contact regime.

    Four-legged data is held out by whole recording — the strongest guarantee
    available, since two recordings are minutes apart. Reduced-contact data
    cannot be: each topology appears in exactly one recording, so each of those
    blocks is cut in time instead, 72% to train and the tail to validation with
    a 3 s gap between.
    """
    import glob

    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    val_names = set(VAL_FILES)
    train_files = [f for f in files if os.path.basename(f) not in val_names]
    quad_val_files = [f for f in files if os.path.basename(f) in val_names]

    train_ds = ForceWindowDataset(train_files, history, split=TRAIN_SPLIT)
    stats = train_ds.stats

    val_sets = {
        "quad": ForceWindowDataset(quad_val_files, history,
                                   modes=(QUAD_WALK, QUAD_STAND), stats=stats),
        "reduced": ForceWindowDataset(train_files, history, modes=(REDUCED,),
                                      split=VAL_SPLIT, stats=stats),
    }
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loaders = {k: DataLoader(v, batch_size=1024, shuffle=False) for k, v in val_sets.items()}
    return train_loader, val_loaders, stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=int, default=20)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default="checkpoints/mlp.pt")
    p.add_argument("--monitor", default="quad", choices=["quad", "reduced"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    here = os.path.dirname(os.path.abspath(__file__))

    train_loader, val_loaders, stats = build_loaders(
        os.path.join(here, "dataset"), args.history, args.batch_size)
    print(f"device {device}   train {len(train_loader.dataset)} windows   "
          + "   ".join(f"val/{k} {len(v.dataset)}" for k, v in val_loaders.items()))

    model = MLP(history=args.history, hidden=args.hidden).to(device)
    total, _ = count_parameters(model)
    print(f"MLP hidden={args.hidden}  {total} parameters\n")

    fit(model, train_loader, val_loaders, device, epochs=args.epochs, lr=args.lr,
        patience=args.patience, ckpt=os.path.join(here, args.ckpt), stats=stats,
        monitor=args.monitor)

    print("\nfinal metrics on the best checkpoint:")
    for name, dl in val_loaders.items():
        m = evaluate(model, dl, device)
        print(f"  val/{name:<8} MSE {m['mse']:.5f}  RMSE {m['rmse']:.4f}  "
              f"MAE {m['mae']:.4f}  R2 {m['r2']:+.4f}  neg {100 * m['neg_frac']:.1f}%")
