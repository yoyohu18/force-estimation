"""Infer the gait of each recording from the foot-contact timing alone.

The method is the standard one from locomotion studies:

  1. contact = (foot_mask == 0), cleaned of runs shorter than MIN_RUN_S so a
     single-sample sensor glitch does not read as a footfall.
  2. Split the episode into segments where the *set of stepping legs* is
     constant — some recordings switch from four-legged walking to walking on
     two legs partway through, and averaging across that switch is meaningless.
  3. Per segment: stride period T = median interval between touchdowns of the
     reference leg, duty factor = fraction of the stride each foot is loaded,
     and the relative phase of every other leg, phi = (td_leg - td_ref) / T
     taken as a circular mean over all strides.
  4. Classify by nearest phase template. The templates are what separates the
     symmetric gaits: they differ *only* in which legs share a phase.

         trot   diagonal pairs together   FL .5  RR .5  RL 0
         pace   lateral pairs together    FL .5  RR  0  RL .5
         bound  front pair, then rear     FL  0  RR .5  RL .5
         pronk  all four together         FL  0  RR  0  RL 0
         walk   one foot at a time        FL .5  RR .25 RL .75  (duty > .5)

     Distance is circular, so 0.02 and 0.98 count as the same phase.

Usage:
    python gait_analysis.py                    # table + gait diagram figure
    python gait_analysis.py -f 2026-08-03_19-36-12.h5
"""

import argparse
import glob
import os

import h5py
import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
LEG_NAMES = ["FR", "FL", "RR", "RL"]
GROUP = "lowstate"      # 500 Hz: the only rate that resolves touchdown timing
MIN_RUN_S = 0.04        # shorter contact/flight runs are sensor chatter
WIN_S = 4.0             # sliding window for the stepping-leg state
MIN_SEG_S = 5.0         # ignore segments too short to hold a stride
MIN_STRIDES = 4         # below this the phase estimate is not worth reporting

# phase of FL, RR, RL relative to FR
TEMPLATES = {
    "trot":  (0.5, 0.5, 0.0),
    "pace":  (0.5, 0.0, 0.5),
    "bound": (0.0, 0.5, 0.5),
    "pronk": (0.0, 0.0, 0.0),
    "walk":  (0.5, 0.25, 0.75),
}


def _runs(x):
    """[(start, stop, value)] for each constant run in a boolean array."""
    edges = np.flatnonzero(np.diff(x.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [len(x)]))
    return [(a, b, bool(x[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def _despeckle(contact, min_len):
    """Flip runs shorter than min_len samples to their neighbours' value."""
    out = contact.copy()
    for a, b, v in _runs(contact):
        if b - a < min_len:
            out[a:b] = not v
    return out


def _circ_mean(phases):
    """Mean of values on the unit circle; plain averaging breaks across 0/1."""
    z = np.exp(2j * np.pi * np.asarray(phases))
    return float(np.angle(z.mean()) / (2 * np.pi) % 1.0), float(abs(z.mean()))


def _circ_dist(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def segment(contact, fs):
    """Split into stretches with a constant stepping/planted/airborne pattern."""
    win, hop = int(WIN_S * fs), int(WIN_S * fs / 4)
    states, centers = [], []
    for i in range(0, len(contact) - win + 1, hop):
        frac = contact[i:i + win].mean(axis=0)
        states.append("".join("C" if f > 0.95 else "-" if f < 0.05 else "S"
                              for f in frac))
        centers.append(i + win // 2)
    if not states:
        return []

    # cut at the midpoint between the windows straddling a state change, so
    # neighbouring segments meet exactly instead of overlapping by a window
    bounds, start = [], 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            bounds.append((start, i))
            start = i

    segs = []
    for k, (s, e) in enumerate(bounds):
        a = 0 if k == 0 else (centers[bounds[k - 1][1] - 1] + centers[s]) // 2
        b = len(contact) if k == len(bounds) - 1 else (
            centers[e - 1] + centers[bounds[k + 1][0]]) // 2
        if (b - a) / fs >= MIN_SEG_S:
            segs.append((a, b, states[s]))
    return segs


def describe(contact, fs, state):
    """Duty factors, stride frequency, phases and a gait label for one segment."""
    stepping = [i for i, c in enumerate(state) if c == "S"]
    duty = contact.mean(axis=0)

    if not stepping:
        label = "standing (all four planted)" if state == "CCCC" else f"static [{state}]"
        return dict(label=label, duty=duty, freq=np.nan, phases={}, n_strides=0)

    # touchdown = rising edge of contact, per stepping leg
    td = {}
    for leg in stepping:
        edges = np.flatnonzero(np.diff(contact[:, leg].astype(np.int8)) == 1) + 1
        td[leg] = edges / fs
    ref = max(stepping, key=lambda l: len(td[l]))
    if len(td[ref]) < MIN_STRIDES:
        return dict(label="too few strides to classify", duty=duty, freq=np.nan,
                    phases={}, n_strides=len(td[ref]))

    T = float(np.median(np.diff(td[ref])))
    phases, conc = {}, {}
    for leg in stepping:
        if leg == ref or not len(td[leg]):
            phases[leg], conc[leg] = (0.0, 1.0) if leg == ref else (np.nan, 0.0)
            continue
        # each reference touchdown paired with the next touchdown of this leg
        idx = np.searchsorted(td[leg], td[ref])
        idx = idx[idx < len(td[leg])]
        if not len(idx):
            phases[leg], conc[leg] = np.nan, 0.0
            continue
        lag = td[leg][idx] - td[ref][:len(idx)]
        phases[leg], conc[leg] = _circ_mean(lag / T)

    # re-reference to FR whenever FR steps, so the printed phases can be read
    # straight against the templates instead of against whichever leg had the
    # most touchdowns
    if 0 in phases and np.isfinite(phases[0]):
        shift = phases[0]
        phases = {l: (p - shift) % 1.0 for l, p in phases.items()}

    label = classify(phases, duty, stepping, ref, conc)
    return dict(label=label, duty=duty, freq=1.0 / T, phases=phases,
                n_strides=len(td[ref]) - 1, conc=conc, ref=ref, stepping=stepping)


def classify(phases, duty, stepping, ref, conc):
    """Nearest phase template, or a two-leg description when only two step."""
    # circular concentration R of the per-stride phases: 1 is a metronome, 0 is
    # noise. Below 0.4 the mean phase is meaningless, so no template is fitted.
    r_min = min(conc.get(l, 0.0) for l in stepping)
    if r_min < 0.4:
        return f"irregular (phase not locked, R={r_min:.2f})"

    if len(stepping) == 2:
        a, b = stepping
        d = _circ_dist(phases[a] - phases[b], 0.0)
        pair = f"{LEG_NAMES[a]}+{LEG_NAMES[b]}"
        kind = "in-phase hopping" if d < 0.15 else (
            "alternating" if _circ_dist(phases[a] - phases[b], 0.5) < 0.15 else "irregular")
        return f"two-leg {kind} on {pair}"

    # shift every phase so FR is the zero reference, then match the templates
    rel = {}
    for leg in range(4):
        if leg in phases and not np.isnan(phases[leg]):
            rel[leg] = (phases[leg] - phases.get(0, 0.0)) % 1.0
    if 0 not in rel:  # FR is not stepping -> three-leg gait, no template applies
        return f"three-leg gait ({'+'.join(LEG_NAMES[l] for l in stepping)} stepping)"

    best, bestd = None, 9e9
    for name, tpl in TEMPLATES.items():
        d = np.mean([_circ_dist(rel[leg], tpl[leg - 1])
                     for leg in (1, 2, 3) if leg in rel])
        if d < bestd:
            best, bestd = name, d
    mean_duty = float(np.mean([duty[l] for l in stepping]))
    quality = "" if bestd < 0.08 else f" (loose, d={bestd:.2f})" if bestd < 0.15 else "?"
    flight = "walking " if mean_duty > 0.5 and best == "trot" else ""
    tag = f"{flight}{best}{quality}"
    if len(stepping) < 4:
        tag += f", only {'+'.join(LEG_NAMES[l] for l in stepping)} stepping"
    return tag


def analyse(path):
    with h5py.File(path, "r") as f:
        g = f[GROUP]
        t = (np.asarray(g["times_ms"], dtype=float) - g["times_ms"][0]) / 1000.0
        contact = np.asarray(g["foot_mask"]) == 0
    fs = 1.0 / float(np.median(np.diff(t)))
    contact = np.column_stack([_despeckle(contact[:, l], int(MIN_RUN_S * fs))
                               for l in range(4)])

    out = []
    for a, b, state in segment(contact, fs):
        info = describe(contact[a:b], fs, state)
        info.update(t0=t[a], t1=t[b - 1], state=state, span=(a, b))
        out.append(info)
    return out, contact, t, fs


def fig_gait(rows, out_path):
    """Footfall diagram: one row per segment, a representative 6 s of contacts."""
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.55 * n + 1.0), squeeze=False)
    for ax, r in zip(axes[:, 0], rows):
        c, t, fs = r["contact"], r["t"], r["fs"]
        a, b = r["span"]
        mid = (a + b) // 2
        w = int(6 * fs)
        i0, i1 = max(a, mid - w // 2), min(b, mid + w // 2)
        tt = t[i0:i1] - t[i0]
        for leg in range(4):
            for s, e, v in _runs(c[i0:i1, leg]):
                if v:
                    ax.barh(3 - leg, tt[e - 1] - tt[s], left=tt[s], height=0.62,
                            color=PALETTE[leg], edgecolor="none")
        ax.set_yticks(range(4), LEG_NAMES[::-1], fontsize=8)
        ax.set_xlim(0, tt[-1] if len(tt) else 6)
        ax.set_ylim(-0.6, 3.6)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        freq = f"{r['freq']:.2f} Hz" if np.isfinite(r["freq"]) else "-"
        duty = " ".join(f"{LEG_NAMES[l]} {r['duty'][l]:.2f}" for l in range(4))
        ax.set_title(f"{r['file']}   t={r['t0']:.0f}-{r['t1']:.0f}s   "
                     f"{r['label']}   |   stride {freq}   duty: {duty}",
                     fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("time [s] (6 s from the middle of the segment)")
    fig.suptitle("inferred gaits — coloured bar = foot loaded (foot_mask == 0)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--data-dir", default=os.path.join(here, "dataset"))
    p.add_argument("-f", "--file", nargs="*")
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if args.file:
        want = {os.path.basename(x) for x in args.file}
        files = [f for f in files if os.path.basename(f) in want]
    if not files:
        raise SystemExit(f"no matching .h5 in {args.data_dir}")

    rows = []
    for path in files:
        name = os.path.basename(path).replace(".h5", "")
        segs, contact, t, fs = analyse(path)
        print(f"\n{name}")
        for s in segs:
            freq = f"{s['freq']:5.2f} Hz" if np.isfinite(s["freq"]) else "    -   "
            ph = "  ".join(f"{LEG_NAMES[l]} {s['phases'][l]:.2f}"
                           for l in sorted(s["phases"])) if s["phases"] else ""
            print(f"  {s['t0']:6.1f}-{s['t1']:6.1f}s [{s['state']}]  {freq}  "
                  f"duty {np.round(s['duty'], 2)}  {s['label']}")
            if ph:
                print(f"          phase: {ph}   ({s['n_strides']} strides)")
            s.update(file=name, contact=contact, t=t, fs=fs)
            rows.append(s)

    out = args.out or os.path.join(args.data_dir, "figures", "gait_summary.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    moving = [r for r in rows if "S" in r["state"]]  # standing rows are flat bars
    pages = [moving[i:i + 12] for i in range(0, len(moving), 12)]
    for k, page in enumerate(pages, 1):
        path = out if len(pages) == 1 else out.replace(".png", f"_{k}.png")
        fig_gait(page, path)
        print(f"  page {k}: {len(page)} segments -> {path}")
    print(f"\n{len(rows)} segments ({len(moving)} with stepping)")


if __name__ == "__main__":
    main()
