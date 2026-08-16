"""Sliding-window Dataset: a sample is `history` consecutive frames.

A sample is

    x : (history, 33)  standardised features, the window ending at frame t
    y : (4,)           foot_sensors at frame t

Windows are never allowed to span a boundary the robot did not actually cross.
Three kinds exist and all three are enforced here:

    file boundary   two recordings are minutes apart; concatenating them and
                    then windowing invents a sample where the robot teleports
    mode boundary   half four-legged walking, half handstand, labelled as one
                    thing; Step 4 decided these get dropped
    time gap        a jump in times_ms inside a group

So episodes are kept separate and the index holds (episode, end frame) pairs
instead of a flat offset into one big array.
"""

import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from features import apply_stats, build_features, fit_stats
from modes import FS, REDUCED, label_modes

MAX_GAP_MS = 25.0  # interp2 is a 20 ms grid; anything larger is a hole


def _blocks(values, max_gap=None, times=None):
    """Start/stop indices of runs where `values` is constant and time is contiguous."""
    cuts = list(np.flatnonzero(np.diff(values)) + 1)
    if times is not None:
        cuts += list(np.flatnonzero(np.diff(times) > max_gap) + 1)
    cuts = np.unique(cuts)
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, len(values)]
    return zip(starts, stops)


class ForceWindowDataset(Dataset):
    """`history` frames of context per sample, drawn from within one segment.

    `modes=None` keeps every frame. `stats` must come from the training split;
    see the note in features.fit_stats.
    """

    def __init__(self, files, history=20, modes=None, group="interp2", stats=None,
                 split=None, split_modes=(REDUCED,), gap_s=3.0):
        """`split` cuts each block whose mode is in `split_modes` in time.

        ("head", 0.72) keeps the first 72% of the block, ("tail", 0.72) keeps
        what follows after a `gap_s` pause. Four-legged data is held out by
        whole recording instead, which is a stronger guarantee; the reduced
        modes only occur in one recording each, so there is nothing to hold
        out at file level and a timed cut is the best available.

        The gap matters: neighbouring frames are 4x closer to each other than
        two random frames, so a cut with no gap puts near-duplicates on both
        sides. 3 s is about six stride periods.
        """
        if history < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        if split is not None and split[0] not in ("head", "tail"):
            raise ValueError(f"split must be ('head'|'tail', frac), got {split!r}")
        self.history = history
        self.episodes = []          # one (x, y) pair per file
        index = []                  # (episode id, index of the last frame in the window)
        fit_frames = []             # only frames some window actually reaches

        for path in files:
            with h5py.File(path, "r") as f:
                g = f[group]
                x = build_features(g)
                y = g["foot_sensors"][...].astype(np.float32)
                times = g["times_ms"][...]
                mode = label_modes(g["foot_mask"][...])

            ep = len(self.episodes)
            self.episodes.append((x, y))
            used = np.zeros(len(x), dtype=bool)
            for start, stop in _blocks(mode, MAX_GAP_MS, times):
                if modes is not None and mode[start] not in modes:
                    continue
                if split is not None and mode[start] in split_modes:
                    which, frac = split
                    cut = start + int((stop - start) * frac)
                    start, stop = ((start, cut) if which == "head"
                                   else (cut + int(gap_s * FS), stop))
                if stop - start < history:
                    continue  # the block is shorter than one window
                used[start:stop] = True
                # `end` is the labelled frame; the window is [end-history+1, end]
                index += [(ep, end) for end in range(start + history - 1, stop)]
            fit_frames.append(x[used])

        if not index:
            raise ValueError(
                f"no window of {history} frames fits any segment "
                f"(modes={modes}, {len(files)} file(s))"
            )
        self.index = np.asarray(index, dtype=np.int64)
        self.stats = fit_stats(np.concatenate(fit_frames)) if stats is None else stats
        self.episodes = [(apply_stats(x, self.stats), y) for x, y in self.episodes]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ep, end = self.index[i]
        x, y = self.episodes[ep]
        window = x[end - self.history + 1 : end + 1]  # (history, 33)
        return torch.from_numpy(window.copy()), torch.from_numpy(y[end].copy())


VAL_FILES = (
    "2026-08-03_19-26-22.h5",   # trot, four-legged
    "2026-08-03_19-44-40.h5",   # pronk, four-legged — a gait the train split barely has
)

# Reduced-contact data cannot be held out by file: each of the three contact
# topologies (front pair, rear pair, diagonal) occurs in exactly one recording.
# Holding one out entirely turns its validation number into an extrapolation
# test — measured that way the model scored R2 = -5.8 on rear-pair walking it
# had never seen. So those blocks get a timed cut instead.
TRAIN_SPLIT = ("head", 0.72)
VAL_SPLIT = ("tail", 0.72)


def make_loaders(data_dir=None, history=20, batch_size=256, modes=None,
                 val_files=VAL_FILES, num_workers=0):
    """Train/val DataLoaders, split by whole recording.

    Splitting by frame instead would put frame t in train and t+1 in val; they
    are 20 ms apart and near-identical, which inflates the validation score by
    an order of magnitude (measured: 11x with a nearest-neighbour model).

    num_workers stays 0 on purpose: every episode is already in RAM, and h5py
    handles do not survive the fork that worker processes do.
    """
    from torch.utils.data import DataLoader

    data_dir = data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    val = [f for f in files if os.path.basename(f) in set(val_files)]
    train = [f for f in files if os.path.basename(f) not in set(val_files)]
    if not val or not train:
        raise ValueError(f"bad split: {len(train)} train / {len(val)} val in {data_dir}")

    train_ds = ForceWindowDataset(train, history, modes)
    val_ds = ForceWindowDataset(val, history, modes, stats=train_ds.stats)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        train_ds.stats,
    )


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "dataset", "*.h5")))

    ds = ForceWindowDataset(files, history=20)
    print(f"all files, history=20:  {len(ds)} windows "
          f"({60902 - len(ds)} frames lost to segment edges)")

    # no window may straddle two episodes or reach past the start of one
    eps, ends = ds.index[:, 0], ds.index[:, 1]
    lens = np.array([len(x) for x, _ in ds.episodes])
    print(f"  starts inside its episode: {bool((ends - ds.history + 1 >= 0).all())}")
    print(f"  ends inside its episode:   {bool((ends < lens[eps]).all())}")

    train_loader, val_loader, stats = make_loaders(history=20)
    print(f"\nsplit: train {len(train_loader.dataset)} windows "
          f"/ val {len(val_loader.dataset)} windows")
    x, y = next(iter(train_loader))
    print(f"one batch:  x {tuple(x.shape)} {x.dtype}   y {tuple(y.shape)} {y.dtype}")
    print(f"  x[:, :, :30] mean {x[:, :, :30].mean():+.4f} std {x[:, :, :30].std():.4f}")
    print(f"  x[:, :, 30:] is the unit gravity vector, left unscaled: "
          f"|g| = {x[:, :, 30:].norm(dim=-1).mean():.4f}")
