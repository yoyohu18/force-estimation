"""The simplest possible Dataset for this project: one frame in, one frame out.

A torch Dataset answers exactly two questions:

    len(ds)   -> how many samples are there
    ds[i]     -> what does sample i look like

Nothing else. Batching, shuffling and parallel loading are the DataLoader's
job (Step 7), which is why the two are separate classes: this file decides
*what a sample is*, and never has to care *how samples are grouped*.

Here a sample is one 50 Hz frame:

    x : (33,)  standardised features   -> features.py
    y : (4,)   foot_sensors, ~[0, 1]

Step 6 replaces the single frame with a window of history; only __getitem__
changes.
"""

import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from features import apply_stats, build_features, fit_stats
from modes import QUAD_STAND, QUAD_WALK, label_modes


class ForceDataset(Dataset):
    """Frames from one or more recordings.

    `modes=None` keeps every frame — four-legged walking, standing, and the
    two- and three-legged stretches alike. Pass a tuple of mode codes to keep
    only some, e.g. `modes=(REDUCED,)` to build a reduced-contact test set.

    `stats` must be fitted on the training split and then passed to every other
    split. Refitting it per split both leaks the validation distribution into
    the preprocessing and, worse, silently rescales out-of-distribution inputs
    back into the range the model was trained on.
    """

    def __init__(self, files, modes=None, group="interp2", stats=None):
        xs, ys = [], []
        for path in files:
            with h5py.File(path, "r") as f:
                g = f[group]
                x = build_features(g)
                y = g["foot_sensors"][...].astype(np.float32)
                mode = label_modes(g["foot_mask"][...])
            keep = np.ones(len(mode), bool) if modes is None else np.isin(mode, modes)
            if keep.any():
                xs.append(x[keep])
                ys.append(y[keep])
        if not xs:
            raise ValueError(f"no frames matched modes={modes} in {len(files)} file(s)")

        self.x = np.concatenate(xs)
        self.y = np.concatenate(ys)
        self.stats = fit_stats(self.x) if stats is None else stats
        self.x = apply_stats(self.x, self.stats)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        # .copy() because torch.from_numpy shares memory with the numpy array;
        # without it, anything that writes to the tensor would edit the dataset.
        return torch.from_numpy(self.x[i].copy()), torch.from_numpy(self.y[i].copy())


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "dataset", "*.h5")))

    ds = ForceDataset(files)
    print(f"len(ds) = {len(ds)}")

    x, y = ds[0]
    print(f"\nds[0] is a {type((x, y)).__name__} of two tensors:")
    print(f"  x  shape {tuple(x.shape)}  dtype {x.dtype}  device {x.device}")
    print(f"  y  shape {tuple(y.shape)}  dtype {y.dtype}")
    print(f"\n  x = {x[:6].numpy().round(3)} ...   (first 6 of 33)")
    print(f"  y = {y.numpy().round(3)}   (FR FL RR RL)")

    # a Dataset is not a tensor and has no shape of its own — it is a lookup
    print(f"\nds[10] and ds[11] are 20 ms apart:")
    print(f"  |x10 - x11| = {torch.norm(ds[10][0] - ds[11][0]):.3f}")
    print(f"  |x10 - x900| = {torch.norm(ds[10][0] - ds[900][0]):.3f}")
