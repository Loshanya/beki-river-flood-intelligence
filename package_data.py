
from pathlib import Path
import re
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT / "ML"
OUTPUT = Path(__file__).resolve().parent / "flood_grids.npz"

discharges = []
depth_grids = []
velocity_grids = []

for folder in sorted(DATA_ROOT.glob("BEKI_FinalQ_*_GVF - Copy")):
    match = re.search(r"BEKI_FinalQ_(\d+)_GVF", folder.name)
    if not match:
        continue

    q = int(match.group(1))

    # Find average-grid files, excluding original/backup versions.
    depth_files = [
        p for p in folder.glob("h_avg*.xlsx")
        if "_org" not in p.stem.lower()
    ]
    velocity_files = [
        p for p in folder.glob("rv_avg*.xlsx")
        if "_org" not in p.stem.lower()
    ]

    if not depth_files or not velocity_files:
        print(f"Skipping Q={q}: average grid file missing")
        continue

    h = pd.read_excel(depth_files[0], header=None).to_numpy(
        dtype=np.float32
    )
    v = pd.read_excel(velocity_files[0], header=None).to_numpy(
        dtype=np.float32
    )

    if h.shape != (286, 600) or v.shape != (286, 600):
        print(f"Skipping Q={q}: unexpected shapes {h.shape}, {v.shape}")
        continue

    if not np.isfinite(h).all() or not np.isfinite(v).all():
        print(f"Skipping Q={q}: non-finite values")
        continue

    discharges.append(q)
    depth_grids.append(h)
    velocity_grids.append(v)
    print(f"Included Q={q}")

if not discharges:
    raise RuntimeError("No valid scenario grids were found.")

order = np.argsort(discharges)
np.savez_compressed(
    OUTPUT,
    discharges=np.asarray(discharges, dtype=np.int32)[order],
    depth=np.stack(depth_grids)[order],
    velocity=np.stack(velocity_grids)[order],
)

print(f"\nSaved: {OUTPUT}")
print(f"Scenarios: {len(discharges)}")
print(f"Package size: {OUTPUT.stat().st_size / (1024**2):.2f} MB")
