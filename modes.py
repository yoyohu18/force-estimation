"""Per-frame locomotion-mode labels.

The recordings are not homogeneous. Three of the twelve hold long stretches
where the robot is not walking on four legs at all:

    19-36-12   148 s walking on the front pair (RR/RL held up)
    19-41-12    45 s walking on the rear pair  (FR/FL held up)
    19-46-51    60 s balancing on the FR+RL diagonal

Body attitude does not separate these: the diagonal balance keeps the trunk
level (7.8 deg tilt, indistinguishable from normal walking). The contact
pattern does, so that is what this module reads.

The labels are for *organising the data* — which frames go to train, which are
held out as an out-of-distribution test. They are derived from foot_mask, so
they must never be fed to the model as an input feature: foot_mask is a
function of the target.

Usage:
    from modes import label_modes, MODE_NAMES
    mode = label_modes(foot_mask)          # (T,) int8
"""

import numpy as np

QUAD_WALK = 0   # all four legs load-bearing, at least one stepping
QUAD_STAND = 1  # all four planted, nothing stepping
REDUCED = 2     # at least one leg held off the ground -> a different contact mode

MODE_NAMES = {QUAD_WALK: "quad_walk", QUAD_STAND: "quad_stand", REDUCED: "reduced"}

FS = 50.0        # interp/interp2 sample rate [Hz]
WIN_S = 1.0      # decision window; ~2 stride periods at 2 Hz
LIFT_THR = 0.05  # stance fraction below this -> the leg is being carried
PLANT_THR = 0.95 # stance fraction above this -> the leg never leaves the ground
MIN_BLOCK_S = 2.0  # a pattern must hold this long to count as its own mode

# Why MIN_BLOCK_S exists: during the 60 s diagonal balance in 19-46-51 the
# raised FL foot taps the ground for ~120 ms every few seconds. Six frames of
# contact push its stance fraction over LIFT_THR, and the 1 s decision window
# then smears that tap into a 52-frame "FL is down" block. Those blocks came
# out exactly WIN_S long, which is the tell: they are an artefact of the
# smoothing, not a mode the robot was ever in. Runs shorter than MIN_BLOCK_S
# are absorbed back into whichever neighbour they interrupted.


def _rolling_mean(x, win):
    """Centred moving average along axis 0. Windows shrink at the edges."""
    csum = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    n = len(x)
    lo = np.clip(np.arange(n) - win // 2, 0, n)
    hi = np.clip(np.arange(n) + win // 2 + 1, 0, n)
    return (csum[hi] - csum[lo]) / (hi - lo)[:, None]


def stance_fraction(foot_mask, fs=FS, win_s=WIN_S):
    """(T, 4) foot_mask -> (T, 4) fraction of a ~1 s window each leg is loaded.

    Recall the polarity: foot_mask == 1 is swing, == 0 is stance. A window is
    needed because a single frame cannot tell "mid-swing" from "held up and
    not used at all" — over a stride the first returns to the ground, the
    second does not.
    """
    contact = 1.0 - np.asarray(foot_mask, dtype=np.float64)
    return _rolling_mean(contact, int(round(win_s * fs)))


def lifted_legs(foot_mask, fs=FS, win_s=WIN_S):
    """(T, 4) bool: which legs are being carried rather than used."""
    return stance_fraction(foot_mask, fs, win_s) <= LIFT_THR


def _despeckle(codes, min_len):
    """Absorb runs shorter than min_len into their longer neighbour."""
    out = np.asarray(codes).copy()
    while True:
        cuts = np.flatnonzero(np.diff(out)) + 1
        starts, ends = np.r_[0, cuts], np.r_[cuts, len(out)]
        lens = ends - starts
        short = np.flatnonzero(lens < min_len)
        if len(short) == 0 or len(lens) == 1:
            return out
        k = short[np.argmin(lens[short])]  # shortest run first, so ties resolve stably
        prev_len = lens[k - 1] if k > 0 else -1
        next_len = lens[k + 1] if k + 1 < len(lens) else -1
        src = k - 1 if prev_len >= next_len else k + 1
        out[starts[k]:ends[k]] = out[starts[src]]


def contact_pattern(foot_mask, fs=FS, win_s=WIN_S, min_block_s=MIN_BLOCK_S):
    """(T,) int: a 4-bit code for *which* legs are lifted, FR/FL/RR/RL = bits 0..3.

    REDUCED lumps three physically different situations together — walking on
    the front pair, on the rear pair, and balancing on a diagonal. They are
    separate contact topologies, so keep them separable.
    """
    code = (lifted_legs(foot_mask, fs, win_s) * np.array([1, 2, 4, 8])).sum(axis=1)
    return _despeckle(code, int(round(min_block_s * fs)))


def pattern_name(code):
    """Human-readable name for a contact_pattern code."""
    legs = ["FR", "FL", "RR", "RL"]
    up = [legs[i] for i in range(4) if code >> i & 1]
    return "none_lifted" if not up else "lifted_" + "+".join(up)


def label_modes(foot_mask, fs=FS, win_s=WIN_S, min_block_s=MIN_BLOCK_S):
    """(T, 4) foot_mask -> (T,) coarse mode label."""
    planted = stance_fraction(foot_mask, fs, win_s) >= PLANT_THR
    pattern = contact_pattern(foot_mask, fs, win_s, min_block_s)

    # Despeckle walk-vs-stand on its own first. Doing it against REDUCED as
    # well lets a genuine four-legged interlude fragment on every walk/stand
    # boundary and then get absorbed run-by-run into the reduced blocks
    # around it — 8 s of ordinary walking disappeared that way.
    mode = np.where(planted.all(axis=1), QUAD_STAND, QUAD_WALK)
    mode = _despeckle(mode, int(round(min_block_s * fs)))
    mode[pattern != 0] = REDUCED  # the pattern is already despeckled; it wins
    return mode.astype(np.int8)


def summarize(mode):
    """Frame counts per mode, as a dict keyed by name."""
    return {name: int((mode == k).sum()) for k, name in MODE_NAMES.items()}


if __name__ == "__main__":
    import glob
    import os

    import h5py

    here = os.path.dirname(os.path.abspath(__file__))
    total = {name: 0 for name in MODE_NAMES.values()}
    by_pattern = {}

    print(f"{'file':<22}{'quad_walk':>11}{'quad_stand':>12}{'reduced':>10}   reduced is")
    for path in sorted(glob.glob(os.path.join(here, "dataset", "*.h5"))):
        with h5py.File(path, "r") as f:
            fm = f["interp2"]["foot_mask"][...]
        mode = label_modes(fm)
        counts = summarize(mode)
        for k, v in counts.items():
            total[k] += v

        pat = np.where(mode == REDUCED, contact_pattern(fm), 0)
        for code in np.unique(pat[pat != 0]):
            by_pattern[int(code)] = by_pattern.get(int(code), 0) + int((pat == code).sum())
        which = ", ".join(pattern_name(int(c)) for c in np.unique(pat[pat != 0])) or "-"
        print(f"{os.path.basename(path)[:19]:<22}"
              f"{counts['quad_walk']:>11}{counts['quad_stand']:>12}"
              f"{counts['reduced']:>10}   {which}")

    print(f"\n{'TOTAL':<22}{total['quad_walk']:>11}"
          f"{total['quad_stand']:>12}{total['reduced']:>10}")

    print("\nreduced frames by contact pattern:")
    for code, n in sorted(by_pattern.items(), key=lambda kv: -kv[1]):
        print(f"  {pattern_name(code):<22}{n:>8}")
