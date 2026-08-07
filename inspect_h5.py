"""Print the structure of an h5 recording: every group, dataset, shape and range.

Usage:
    python inspect_h5.py                    # first *.h5 next to this script
    python inspect_h5.py path/to/file.h5
"""

import glob
import os
import sys

import h5py
import numpy as np


def describe(path, preview_rows=3):
    print(f"=== {os.path.basename(path)} ===\n")
    with h5py.File(path, "r") as f:
        if f.attrs:
            print("file attrs:", dict(f.attrs), "\n")

        def visit(name, obj):
            indent = "  " * name.count("/")
            if isinstance(obj, h5py.Group):
                print(f"{indent}{name.split('/')[-1]}/   (group, {len(obj)} members)")
                if obj.attrs:
                    print(f"{indent}   attrs: {dict(obj.attrs)}")
            else:
                line = f"{indent}{name.split('/')[-1]:<28} {str(obj.shape):<14} {obj.dtype}"
                if obj.size and np.issubdtype(obj.dtype, np.number):
                    a = obj[...]
                    line += f"  min {np.nanmin(a):+.4g}  max {np.nanmax(a):+.4g}  mean {np.nanmean(a):+.4g}"
                print(line)

        f.visititems(visit)

        # a couple of raw rows so you can eyeball actual values
        first_group = next((k for k in f if isinstance(f[k], h5py.Group)), None)
        if first_group:
            print(f"\n--- first {preview_rows} rows of /{first_group} ---")
            for key, d in f[first_group].items():
                if isinstance(d, h5py.Dataset) and d.ndim >= 1:
                    print(f"{key}:\n{np.asarray(d[:preview_rows])}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        paths = sorted(glob.glob(os.path.join(here, "*.h5")))[:1]
    for p in paths:
        describe(p)
