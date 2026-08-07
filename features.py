"""Build the model input from one recording, plus a self-test of its invariances.

Layout (33 dims), decided in Step 4:

     0:12   joint_positions              rad
    12:24   estimated_joint_velocities   rad/s   (centred difference, NOT causal)
    24:27   lin_vels_in_b                m/s     (from lidar odometry, not proprioceptive)
    27:30   angular_vel                  rad/s
    30:33   unit gravity in body frame   -       (dimensionless, already in [-1, 1])

Excluded on purpose:
    orientations   raw quaternion carries yaw and is double-covered; the unit
                   gravity vector keeps the roll/pitch part and drops both
    positions      world coordinates; a model that reads them memorises routes
    joint_torques  couples to contact force directly through J_c^T f_c, so it
                   answers half the question — held out for a later ablation
    foot_sensors   the target
    foot_mask      a function of the target

Only columns 0:30 get standardised. Columns 30:33 are a unit vector: it is
already O(1), and dividing each component by its own std would break
||g_hat|| == 1, which is the one exact geometric constraint in the input.
"""

import numpy as np

N_JOINTS = 12
FEATURE_DIM = 33
GRAVITY = 9.81

# column blocks, so nothing downstream has to hard-code offsets
JP = slice(0, 12)
JV = slice(12, 24)
LV = slice(24, 27)
AV = slice(27, 30)
GB = slice(30, 33)
STD_COLS = slice(0, 30)  # everything except the unit gravity vector

FEATURE_NAMES = (
    [f"jp{i}" for i in range(12)]
    + [f"jv{i}" for i in range(12)]
    + ["vb_x", "vb_y", "vb_z"]
    + ["w_x", "w_y", "w_z"]
    + ["g_x", "g_y", "g_z"]
)


def unit_gravity_body(quat):
    """(T, 4) quaternion in (w, x, y, z) order -> (T, 3) unit gravity in body frame.

    This is minus the third row of the body->world rotation, i.e. where "down"
    points as seen from the robot. Every term is quadratic in the components,
    so q and -q give the same answer, and a yaw rotation leaves it untouched.
    The dataset stores w first; an (x, y, z, w) array silently gives a wrong
    direction here rather than an error.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return -np.stack(
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1
    )


def build_features(group):
    """h5 group (interp or interp2) -> (T, 33) float32 feature matrix."""
    return np.concatenate(
        [
            group["joint_positions"][...],
            group["estimated_joint_velocities"][...],
            group["lin_vels_in_b"][...],
            group["angular_vel"][...],
            unit_gravity_body(group["orientations"][...]),
        ],
        axis=1,
    ).astype(np.float32)


def fit_stats(feats):
    """Per-column mean/std over the training frames only.

    Columns 30:33 are pinned to mean 0 / std 1 so applying the transform is a
    no-op there — the unit gravity vector passes through untouched.
    """
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std < 1e-6] = 1.0  # a constant column would otherwise divide by ~0
    mean[GB] = 0.0
    std[GB] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_stats(feats, stats):
    return (feats - stats["mean"]) / stats["std"]


# --------------------------------------------------------------------------
# self-test: the three invariances the feature design claims to have
# --------------------------------------------------------------------------

def _quat_mul(a, b):
    """Hamilton product, (w, x, y, z) order, broadcasting over leading axes."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def check_invariances(quat, rng=None):
    """Verify unit norm, double-cover invariance and yaw invariance. Returns errors."""
    rng = rng or np.random.default_rng(0)
    g = unit_gravity_body(quat)

    # 1. the vector is a unit vector
    norm_err = np.abs(np.linalg.norm(g, axis=1) - 1.0).max()

    # 2. q and -q describe the same rotation, so they must give the same vector
    flip_err = np.abs(unit_gravity_body(-quat) - g).max()

    # 3. turning the robot on the spot must not change where "down" is
    psi = rng.uniform(-np.pi, np.pi, size=len(quat))
    q_yaw = np.stack(
        [np.cos(psi / 2), np.zeros_like(psi), np.zeros_like(psi), np.sin(psi / 2)],
        axis=-1,
    )
    yaw_err = np.abs(unit_gravity_body(_quat_mul(q_yaw, quat)) - g).max()

    return {"unit_norm": norm_err, "double_cover": flip_err, "yaw": yaw_err}


if __name__ == "__main__":
    import glob
    import os

    import h5py

    from modes import MODE_NAMES, label_modes

    here = os.path.dirname(os.path.abspath(__file__))
    path = sorted(glob.glob(os.path.join(here, "dataset", "*.h5")))[0]

    with h5py.File(path, "r") as f:
        g = f["interp2"]
        feats = build_features(g)
        quat = g["orientations"][...]
        mode = label_modes(g["foot_mask"][...])

    print(f"{os.path.basename(path)}   features {feats.shape} {feats.dtype}")
    print("modes:", {MODE_NAMES[k]: int((mode == k).sum()) for k in MODE_NAMES})

    print("\ninvariance self-test (all should be ~1e-7 or smaller):")
    for name, err in check_invariances(quat).items():
        print(f"  {name:<14} max error {err:.2e}")

    stats = fit_stats(feats)
    z = apply_stats(feats, stats)
    print("\nafter standardisation:")
    print(f"  cols  0:30  mean {z[:, STD_COLS].mean():+.2e}  std {z[:, STD_COLS].std():.4f}")
    print(f"  cols 30:33  mean {z[:, GB].mean():+.4f}  std {z[:, GB].std():.4f}"
          f"   |g_hat| = {np.linalg.norm(z[:, GB], axis=1).mean():.6f}")
