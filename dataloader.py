"""Minimal dataloader for the Go2 force-estimation recordings.

Inputs (33 dims per timestep, in this order):
    joint_positions              12
    estimated_joint_velocities   12
    lin_vels_in_b                 3
    angular_vel                   3
    gravity in body frame         3   (derived from `orientations`)

Everything here is proprioceptive. World pose is deliberately excluded: the
raw quaternion's roll/pitch is already carried by the body-frame gravity
vector and its yaw is just heading, while world position is a property of the
recorded trajectory rather than of the contact, so feeding it in invites
memorising routes that will not transfer to a new site.

Targets:
    foot_sensors  (4,)  normalized vertical force per foot, ~[0, 1]
    foot_mask     (4,)  binary contact label

Note on foot_mask polarity: mask == 1 goes with *low* foot_sensors readings
(0.054 vs 0.238 on average), and the four sensors sum to ~0.87 when all four
masks are 0. So mask == 0 is the stance phase and mask == 1 is swing.
`target="contact"` returns 1 - foot_mask so that 1 means "foot on the ground".
"""

import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FEATURE_DIM = 33  # joints + body twist + gravity
N_FEET = 4
GRAVITY = 9.81


def gravity_in_body(quat):
    """(T, 4) quaternion in (w, x, y, z) order -> (T, 3) gravity in body frame.

    This is -g times the third row of the body->world rotation matrix. The
    dataset stores w first; feeding an (x, y, z, w) array here silently returns
    the wrong direction, so the order matters.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return -GRAVITY * np.stack(
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1
    )


def load_episode(path, group="interp2"):
    """Read one h5 file into (features, targets, times_ms).

    `interp2` is the validity-filtered version of `interp`; both are 50 Hz.
    """
    with h5py.File(path, "r") as f:
        g = f[group]
        feats = np.concatenate(
            [
                g["joint_positions"][...],
                g["estimated_joint_velocities"][...],
                g["lin_vels_in_b"][...],
                g["angular_vel"][...],
                gravity_in_body(g["orientations"][...]),
            ],
            axis=1,
        ).astype(np.float32)
        targets = {
            "force": g["foot_sensors"][...].astype(np.float32),
            "contact": (1.0 - g["foot_mask"][...]).astype(np.float32),
        }
        times = g["times_ms"][...]
    return feats, targets, times


def _contiguous_segments(times, max_gap_ms=25.0):
    """Split indices into runs with no time gap, so windows never span a hole."""
    breaks = np.flatnonzero(np.diff(times) > max_gap_ms) + 1
    return np.split(np.arange(len(times)), breaks)


class ForceDataset(Dataset):
    """Sliding-window dataset over one or more episodes.

    Each item is (x, y) where x has shape (history, FEATURE_DIM) — the window
    ending at the labelled timestep — and y has shape (4,). With history=1 the
    leading axis is still present, so an MLP just needs a flatten.
    """

    def __init__(self, files, history=1, group="interp2", target="force", stats=None):
        if target not in ("force", "contact"):
            raise ValueError(f"target must be 'force' or 'contact', got {target!r}")
        if history < 1:
            raise ValueError(f"history must be >= 1, got {history}")

        self.history = history
        self.target = target
        self.n_features = FEATURE_DIM
        self.episodes = []
        self.index = []  # (episode_id, end_timestep)

        for ep_id, path in enumerate(files):
            feats, targets, times = load_episode(path, group)
            self.episodes.append((feats, targets[target]))
            for seg in _contiguous_segments(times):
                # only windows that fit entirely inside this segment
                for end in seg[history - 1:]:
                    self.index.append((ep_id, int(end)))

        self.index = np.asarray(self.index, dtype=np.int64)
        self.stats = stats if stats is not None else self.fit_stats()

    def fit_stats(self):
        """Per-feature mean/std over every frame. Fit on train, pass to val."""
        allf = np.concatenate([f for f, _ in self.episodes], axis=0)
        mean = allf.mean(0)
        std = allf.std(0)
        std[std < 1e-6] = 1.0  # constant channels would otherwise blow up
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ep_id, end = self.index[i]
        feats, targ = self.episodes[ep_id]
        window = feats[end - self.history + 1 : end + 1]
        window = (window - self.stats["mean"]) / self.stats["std"]
        return torch.from_numpy(window.copy()), torch.from_numpy(targ[end].copy())


def make_loaders(
    data_dir=None,
    history=1,
    target="force",
    batch_size=256,
    n_val=2,
    group="interp2",
    num_workers=0,
    val_files=None,
    seed=0,
):
    """Split episodes by file (never by frame) and build train/val loaders.

    Splitting by frame would put near-identical neighbouring timesteps on both
    sides and inflate the validation score.

    The split is a seeded shuffle rather than "last n files": the recordings are
    named by timestamp, and the last one (2026-08-03_19-51-05) is 66% standing
    still, so a tail split hands you a validation set dominated by a static
    episode. Pass `val_files` to pin the split yourself.
    """
    # the .h5 files live in a dataset/ subdirectory next to this script
    data_dir = data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if len(files) <= n_val:
        raise ValueError(f"need more than {n_val} files, found {len(files)} in {data_dir}")

    if val_files is not None:
        val_set = {os.path.basename(v) for v in val_files}
        val_files = [f for f in files if os.path.basename(f) in val_set]
        train_files = [f for f in files if os.path.basename(f) not in val_set]
        if not val_files:
            raise ValueError(f"none of {sorted(val_set)} found in {data_dir}")
    else:
        order = np.random.default_rng(seed).permutation(len(files))
        val_idx = set(order[:n_val].tolist())
        train_files = [f for i, f in enumerate(files) if i not in val_idx]
        val_files = [f for i, f in enumerate(files) if i in val_idx]
    print(f"[split] train {len(train_files)} files | "
          f"val {[os.path.basename(v) for v in val_files]}")

    train = ForceDataset(train_files, history, group, target)
    val = ForceDataset(val_files, history, group, target, stats=train.stats)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        train.stats,
    )


if __name__ == "__main__":
    train_loader, val_loader, stats = make_loaders(history=10, target="force")

    print(f"train windows: {len(train_loader.dataset)}")
    print(f"val   windows: {len(val_loader.dataset)}")

    x, y = next(iter(train_loader))
    print(f"x {tuple(x.shape)} {x.dtype}   y {tuple(y.shape)} {y.dtype}")
    print(f"x mean {x.mean():+.4f} std {x.std():.4f}  (should be near 0 / 1)")
    print(f"y range [{y.min():.3f}, {y.max():.3f}]")
    print(f"feature dim: {train_loader.dataset.n_features}")
