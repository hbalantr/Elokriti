"""
Create a publication-quality AL progress figure:
(a) normalized dominated hypervolume vs cumulative MD-evaluated candidates
(b) Pareto set size vs cumulative MD-evaluated candidates

This script is only for the progress figure (fig_al_progress),
not the objective-space tradeoff figure.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Files and settings
# =========================================================
EVALUATED_FILE = "evaluated_clean_no_expansion.csv"
OUTDIR = "figures_al_progress"
os.makedirs(OUTDIR, exist_ok=True)

OBJECTIVES = ["elastic_modulus", "volumetric_shrinkage", "ffv"]
DIRECTIONS = ["max", "min", "min"]

MC_SAMPLES = 300000
SEED = 42

COL_HV = "#1f77b4"
COL_PARETO_SIZE = "#d62728"


# =========================================================
# Plot style: bigger, bolder, publication-quality
# =========================================================
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 16,
    "axes.linewidth": 2.2,
    "axes.labelsize": 22,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "xtick.major.size": 8.0,
    "ytick.major.size": 8.0,
    "xtick.major.width": 1.9,
    "ytick.major.width": 1.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =========================================================
# Helper functions
# =========================================================
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce"
    )


def parse_iteration_value(value):
    s = str(value).lower().strip()

    if s in {"initial", "init", "seed", "d0", "0"}:
        return 0

    s = s.replace("_", "-").replace(" ", "")

    if s.startswith("iter-"):
        return int(s.split("iter-")[-1])
    if s.startswith("iter"):
        return int(s.split("iter")[-1])
    if s.startswith("i") and s[1:].isdigit():
        return int(s[1:])
    if s.startswith("d") and s[1:].isdigit():
        return int(s[1:])
    if s.isdigit():
        return int(s)

    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not parse iteration label: {value}")


def stage_name(k):
    return "initial" if int(k) == 0 else f"iter-{int(k)}"


def short_stage_name(k):
    return "initial" if int(k) == 0 else f"i{int(k)}"


def objective_signs(directions):
    signs = []
    for d in directions:
        d = str(d).lower().strip()
        if d == "max":
            signs.append(1.0)
        elif d == "min":
            signs.append(-1.0)
        else:
            raise ValueError(f"Invalid objective direction: {d}")
    return np.array(signs, dtype=float)


def pareto_mask_maximize(Y):
    """
    Return True for non-dominated points.
    Y must already be in maximize-space.
    """
    n = Y.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        dominates = np.all(Y[i] >= Y, axis=1) & np.any(Y[i] > Y, axis=1)
        is_pareto[dominates] = False
        is_pareto[i] = True

    return is_pareto


def normalize_objectives(Y, lower, upper):
    span = np.maximum(upper - lower, 1e-12)
    return np.clip((Y - lower) / span, 0.0, 1.0)


def estimate_hypervolume_fraction(front_normalized, mc_samples):
    """
    Monte Carlo dominated hypervolume fraction in [0,1]^M.
    Reference point is [0,0,0] after normalization.
    """
    if front_normalized.size == 0:
        return 0.0

    dominated = np.any(
        np.all(front_normalized[:, None, :] >= mc_samples[None, :, :], axis=2),
        axis=0,
    )
    return float(np.mean(dominated))


def compute_hv_and_pareto(df_subset, lower, upper, mc_samples):
    signs = objective_signs(DIRECTIONS)
    Y_raw = df_subset[OBJECTIVES].to_numpy(dtype=float)
    Y_max = Y_raw * signs

    p_mask = pareto_mask_maximize(Y_max)
    pareto_Y = Y_max[p_mask]

    pareto_Y_norm = normalize_objectives(pareto_Y, lower, upper)
    hv = estimate_hypervolume_fraction(pareto_Y_norm, mc_samples)

    return hv, p_mask


def format_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.23, linewidth=1.0)

    ax.tick_params(
        axis="both",
        which="major",
        top=False,
        right=False,
        direction="out",
        width=1.9,
        length=8,
        pad=6,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.2)
        spine.set_color("black")

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def panel_label(ax, label):
    ax.text(
        -0.16, 1.08, label,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


# =========================================================
# Load evaluated data
# =========================================================
if not os.path.exists(EVALUATED_FILE):
    raise FileNotFoundError(f"Cannot find {EVALUATED_FILE}")

df = pd.read_csv(EVALUATED_FILE)
df.columns = df.columns.str.strip()

required = ["reaction_id", "iteration", *OBJECTIVES]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

if "status" in df.columns:
    df["status"] = df["status"].astype(str).str.lower().str.strip()
    df = df[df["status"].isin(["completed", "complete", "valid", "done"])].copy()

for col in OBJECTIVES:
    df[col] = clean_numeric(df[col])

df = df.dropna(subset=OBJECTIVES).copy()
df["iteration_index"] = df["iteration"].apply(parse_iteration_value)
df = df.sort_values(["iteration_index", "reaction_id"]).reset_index(drop=True)

print("\nEvaluated counts by iteration:")
print(df.groupby("iteration_index").size())


# =========================================================
# Compute hypervolume and Pareto summary
# =========================================================
signs = objective_signs(DIRECTIONS)
Y_all_max = df[OBJECTIVES].to_numpy(dtype=float) * signs

lower = Y_all_max.min(axis=0)
upper = Y_all_max.max(axis=0)

rng = np.random.default_rng(SEED)
mc_samples = rng.random((MC_SAMPLES, len(OBJECTIVES)))

summary_rows = []
previous_hv = None
max_iter = int(df["iteration_index"].max())

for k in range(max_iter + 1):
    cumulative = df[df["iteration_index"] <= k].copy()
    if cumulative.empty:
        continue

    hv, p_mask = compute_hv_and_pareto(
        cumulative,
        lower=lower,
        upper=upper,
        mc_samples=mc_samples,
    )

    delta_hv = np.nan if previous_hv is None else hv - previous_hv
    percent_gain = np.nan if previous_hv is None else 100.0 * delta_hv / max(previous_hv, 1e-12)

    summary_rows.append({
        "stage": stage_name(k),
        "iteration_index": k,
        "n_cumulative_candidates": len(cumulative),
        "hypervolume_fraction": hv,
        "delta_hv_from_previous": delta_hv,
        "percent_gain_from_previous": percent_gain,
        "pareto_size": int(p_mask.sum()),
    })

    previous_hv = hv

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(OUTDIR, "mobo_hypervolume_summary.csv"), index=False)

print("\n===== Hypervolume summary =====")
print(summary.to_string(index=False))


# =========================================================
# Figure: AL progress
# =========================================================
x = summary["n_cumulative_candidates"].to_numpy(float)
hv = summary["hypervolume_fraction"].to_numpy(float)
ps = summary["pareto_size"].to_numpy(float)

fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2), facecolor="white")
fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.16, wspace=0.18)

# -------------------------
# Panel a: hypervolume
# -------------------------
ax = axes[0]
ax.plot(
    x, hv,
    marker="o",
    markersize=10.5,
    linewidth=3.0,
    color=COL_HV,
)

for _, row in summary.iterrows():
    yoff = 0.010 if row["iteration_index"] == 0 else 0.012
    xoff = 0.0
    if row["iteration_index"] == 0:
        xoff = -0.3

    ax.text(
        row["n_cumulative_candidates"] + xoff,
        row["hypervolume_fraction"] + yoff,
        short_stage_name(row["iteration_index"]),
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

ax.set_xlabel("Cumulative MD-evaluated candidates", labelpad=10, fontweight="bold")
ax.set_ylabel("Normalized dominated hypervolume", labelpad=10, fontweight="bold")
ax.set_ylim(0.0, max(hv) * 1.18)
ax.set_xlim(min(x) - 3, max(x) + 3)
format_axes(ax)
panel_label(ax, "a")

# -------------------------
# Panel b: Pareto size
# -------------------------
ax = axes[1]
ax.plot(
    x, ps,
    marker="s",
    markersize=10.2,
    linewidth=3.0,
    color=COL_PARETO_SIZE,
)

for _, row in summary.iterrows():
    yoff = 0.45 if row["iteration_index"] == 0 else 0.50
    ax.text(
        row["n_cumulative_candidates"],
        row["pareto_size"] + yoff,
        short_stage_name(row["iteration_index"]),
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

ax.set_xlabel("Cumulative MD-evaluated candidates", labelpad=10, fontweight="bold")
ax.set_ylabel("Pareto set size", labelpad=10, fontweight="bold")
ax.set_ylim(0, max(ps) + 2.5)
ax.set_xlim(min(x) - 3, max(x) + 3)
format_axes(ax)
panel_label(ax, "b")

# Save
base_name = "fig_al_progress_bolder"
pdf_path = os.path.join(OUTDIR, f"{base_name}.pdf")
svg_path = os.path.join(OUTDIR, f"{base_name}.svg")
png_path = os.path.join(OUTDIR, f"{base_name}.png")

fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\nSaved:")
print(pdf_path)
print(svg_path)
print(png_path)
