
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Beki River | Flood Intelligence",
    page_icon="🌊",
    layout="wide",
)

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "flood_grids.npz"

COLORS = [
    "#D9E2EC", "#36A2EB", "#63C7B2",
    "#F4D35E", "#F28E2B", "#D64545"
]
LABELS = [
    "Dry / unclassified", "Very low", "Low",
    "Moderate", "High", "Very high"
]

# ---------------- STYLING ----------------
st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"] {
    background: #F3F6FA;
    color: #243447;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1500px; }
.hero {
    padding: 26px 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, #102A43, #176B87);
    color: #FFFFFF;
    margin-bottom: 22px;
}
.hero h1 { color: #FFFFFF !important; margin-bottom: 6px; }
.hero p { color: #D9EAF2 !important; margin-bottom: 0; }
h1, h2, h3, h4, p, label, li, .stMarkdown,
[data-testid="stCaptionContainer"] { color: #243447; }
[data-testid="stMetricLabel"] { color: #526579 !important; }
[data-testid="stMetricValue"] { color: #102A43 !important; }
[data-testid="stSidebar"] { background: #E5EDF5; }
[data-testid="stSidebar"] * { color: #243447; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #102A43 !important;
    -webkit-text-fill-color: #102A43 !important;
    border: 2px solid #176B87 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background-color: #DCEEF5 !important;
    color: #102A43 !important;
}
.stDownloadButton button {
    background: #176B87;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
}
.stDownloadButton button:hover {
    background: #102A43;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)


# ---------------- COMPACT DATA ----------------
@st.cache_data(show_spinner="Loading packaged hydraulic grids...")
def load_packaged_data(path_string):
    with np.load(path_string, allow_pickle=False) as data:
        discharges = data["discharges"].copy()
        depth = data["depth"].copy()
        velocity = data["velocity"].copy()

    if not (len(discharges) == len(depth) == len(velocity)):
        raise ValueError("Packaged data arrays have inconsistent lengths.")

    if len(discharges) == 0:
        raise ValueError("No hydraulic scenarios were found.")

    if depth.shape != velocity.shape:
        raise ValueError("Depth and velocity grid shapes do not match.")

    return discharges, depth, velocity


def calculate_hazard(h, v):
    fhi = h * (v + 0.5)
    wet = np.isfinite(h) & np.isfinite(v) & (h > 0.1)

    classes = np.zeros(h.shape, dtype=np.uint8)
    classes[wet & (fhi <= 0.5)] = 1
    classes[wet & (fhi > 0.5) & (fhi <= 1.0)] = 2
    classes[wet & (fhi > 1.0) & (fhi <= 2.0)] = 3
    classes[wet & (fhi > 2.0) & (fhi <= 3.0)] = 4
    classes[wet & (fhi > 3.0)] = 5

    return fhi, classes, wet


def predict_at_discharge(q, discharges, depth, velocity):
    lower = np.flatnonzero(discharges <= q)
    upper = np.flatnonzero(discharges >= q)

    if len(lower) == 0 or len(upper) == 0:
        raise ValueError("Discharge is outside the available data range.")

    i1 = lower[-1]
    i2 = upper[0]
    q1 = int(discharges[i1])
    q2 = int(discharges[i2])

    if q1 == q2:
        return depth[i1], velocity[i1], "Available hydraulic scenario", (q1, q2)

    # Linear interpolation between the surrounding available scenarios
    weight = (q - q1) / (q2 - q1)
    h = depth[i1] + weight * (depth[i2] - depth[i1])
    v = velocity[i1] + weight * (velocity[i2] - velocity[i1])

    return h, v, "Linear interpolation", (q1, q2)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
  <h1>🌊 BEKI RIVER FLOOD INTELLIGENCE</h1>
  <p>Spatial flood hazard assessment · Hydraulic scenario analysis ·
  Discharge-based estimation</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
if not DATA_FILE.exists():
    st.error("Packaged data file flood_grids.npz is missing.")
    st.info("Place flood_grids.npz in the same folder as this app.py.")
    st.stop()

try:
    discharges, depths, velocities = load_packaged_data(str(DATA_FILE))
except Exception as exc:
    st.error(f"Unable to load the packaged hydraulic data: {exc}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Simulation controls")
st.sidebar.caption("Choose a discharge scenario")

q_min = int(discharges.min())
q_max = int(discharges.max())

q = st.sidebar.number_input(
    "Discharge (cumecs)",
    min_value=q_min,
    max_value=q_max,
    value=min(300, q_max),
    step=50,
)

st.sidebar.info(
    "Intermediate values are estimated by linear interpolation "
    "between the nearest available scenarios."
)
st.sidebar.markdown("---")
st.sidebar.caption("Study area: Beki River, Assam")
st.sidebar.caption("Model status: Prototype / under validation")
st.sidebar.caption(f"Available scenarios: {len(discharges)} of 50")

# ---------------- COMPUTE ----------------
try:
    with st.spinner("Computing hazard estimate..."):
        h, v, method, bracket = predict_at_discharge(
            float(q), discharges, depths, velocities
        )
        fhi, classes, wet = calculate_hazard(h, v)
except Exception as exc:
    st.error(f"Unable to calculate this scenario: {exc}")
    st.stop()

# ---------------- METRICS ----------------
counts = np.bincount(classes.ravel(), minlength=6)
high_count = int(counts[4] + counts[5])
wet_count = int(wet.sum())
total = classes.size

st.caption(
    f"Method: {method} · Reference scenarios: "
    f"{bracket[0]} and {bracket[1]} cumecs"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected discharge", f"{q:,.0f} m³/s")
c2.metric("Wet cells", f"{wet_count:,}")
c3.metric("High + very high cells", f"{high_count:,}")
c4.metric("Grid cells", f"{total:,}")

# ---------------- MAP AND DISTRIBUTION ----------------
left, right = st.columns([1.8, 1])

with left:
    st.subheader("Flood hazard map")
    st.caption(
        "Grid-based visualization. Geographic registration has not yet "
        "been verified; axes are grid indices."
    )

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.imshow(classes, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_title(f"Estimated hazard classes at {q:,.0f} cumecs")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.subheader("Hazard distribution")
    distribution = pd.DataFrame({
        "Class": LABELS,
        "Cells": counts,
    })
    st.dataframe(
        distribution, hide_index=True, use_container_width=True
    )

    st.subheader("Hazard legend")
    for i, label in enumerate(LABELS):
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'margin:7px 0;"><span style="display:inline-block;width:16px;'
            f'height:16px;border-radius:4px;background:{COLORS[i]};">'
            f'</span>{label}</div>',
            unsafe_allow_html=True,
        )

# ---------------- HYDRAULIC INDICATORS ----------------
st.markdown("---")
st.subheader("Hydraulic indicators")

m1, m2, m3 = st.columns(3)
m1.metric("Mean depth", f"{np.nanmean(h):.3f}")
m2.metric("Mean velocity", f"{np.nanmean(v):.3f}")
m3.metric("Maximum FHI", f"{np.nanmax(fhi):.3f}")

# ---------------- EXPORT ----------------
st.markdown("---")
st.subheader("Export estimated grid")

export_col1, export_col2 = st.columns(2)

with export_col1:
    st.download_button(
        "Download hazard classes (CSV)",
        data=pd.DataFrame(classes).to_csv(index=False, header=False),
        file_name=f"hazard_classes_Q{q:g}.csv",
        mime="text/csv",
    )

with export_col2:
    st.download_button(
        "Download FHI grid (CSV)",
        data=pd.DataFrame(fhi).to_csv(index=False, header=False),
        file_name=f"fhi_Q{q:g}.csv",
        mime="text/csv",
    )

# ---------------- DISCLAIMER ----------------
st.markdown("---")
st.caption(
    "Research prototype. Estimates use supplied hydraulic scenarios, "
    "linear interpolation, and the current provisional FHI formula. "
    "The 8600-cumec scenario is absent from the packaged data; estimates "
    "at that discharge use the surrounding available scenarios. Hazard "
    "thresholds, masks, spatial registration, and model accuracy require "
    "further validation."
)
