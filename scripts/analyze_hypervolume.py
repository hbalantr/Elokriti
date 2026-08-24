#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OBJECTIVES = ["elastic_modulus", "volumetric_shrinkage", "ffv"]
DIRECTIONS = ["max", "min", "min"]


def clean_numeric_column(series):
    """Convert values like '12.3%' or strings into numeric floats."""
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


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
    True for non-dominated points.
    Assumes all objectives are maximize.
    """
    n = Y.shape[0]
    is_non_dominated = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_non_dominated[i]:
            continue

        dominates = np.all(Y[i] >= Y, axis=1) & np.any(Y[i] > Y, axis=1)
        is_non_dominated[dominates] = False
        is_non_dominated[i] = True

    return is_non_dominated


def normalize_objectives(Y, lower, upper):
    span = np.maximum(upper - lower, 1e-12)
    return np.clip((Y - lower) / span, 0.0, 1.0)


def estimate_hypervolume_fraction(front_normalized, mc_samples):
    """
    Estimate dominated hypervolume fraction in [0,1]^M.
    Reference point is [0,0,0] after normalization.
    """
    if front_normalized.size == 0:
        return 0.0

    dominated = np.any(
        np.all(front_normalized[:, None, :] >= mc_samples[None, :, :], axis=2),
        axis=0,
    )
    return float(np.mean(dominated))


def parse_iteration_value(value):
    """
    Convert iteration labels to ordered integers.
    initial/init/0/d0 -> 0
    iter-1/iter1/1/d1 -> 1
    iter-2/iter2/2/d2 -> 2
    etc.
    """
    s = str(value).lower().strip()

    if s in {"initial", "init", "d0", "0"}:
        return 0

    s = s.replace("_", "-").replace(" ", "")

    if s.startswith("iter-"):
        return int(s.split("iter-")[-1])

    if s.startswith("iter"):
        return int(s.split("iter")[-1])

    if s.startswith("d") and s[1:].isdigit():
        return int(s[1:])

    if s.isdigit():
        return int(s)

    raise ValueError(f"Could not parse iteration label: {value}")


def compute_hv_for_rows(df_subset, lower, upper, mc_samples):
    signs = objective_signs(DIRECTIONS)

    Y_raw = df_subset[OBJECTIVES].to_numpy(dtype=float)
    Y_max = Y_raw * signs

    pareto_mask = pareto_mask_maximize(Y_max)
    pareto_Y = Y_max[pareto_mask]

    pareto_Y_norm = normalize_objectives(pareto_Y, lower, upper)

    hv = estimate_hypervolume_fraction(pareto_Y_norm, mc_samples)

    return hv, pareto_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evaluated_clean.csv")
    parser.add_argument("--output-prefix", default="al_hypervolume")
    parser.add_argument("--mc-samples", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_csv = Path(args.input)
    if not input_csv.exists():
        raise FileNotFoundError(f"Could not find input file: {input_csv}")

    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    required = ["reaction_id", "status", "iteration", *OBJECTIVES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Filter completed rows
    df["status"] = df["status"].astype(str).str.lower().str.strip()
    df = df[df["status"].isin(["completed", "complete", "valid", "done"])].copy()

    # Clean numeric objective columns
    for col in OBJECTIVES:
        df[col] = clean_numeric_column(df[col])

    df = df.dropna(subset=OBJECTIVES).copy()

    # Parse iteration order
    df["iteration_index"] = df["iteration"].apply(parse_iteration_value)
    df = df.sort_values(["iteration_index", "reaction_id"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid completed rows found.")

    signs = objective_signs(DIRECTIONS)

    # Important: use fixed bounds from all currently available evaluated data
    # so HV values across iterations are comparable.
    Y_all = df[OBJECTIVES].to_numpy(dtype=float) * signs
    lower = Y_all.min(axis=0)
    upper = Y_all.max(axis=0)

    rng = np.random.default_rng(args.seed)
    mc_samples = rng.random((args.mc_samples, len(OBJECTIVES)))

    summary_rows = []
    pareto_tables = []

    max_iter = int(df["iteration_index"].max())

    previous_hv = None

    for k in range(0, max_iter + 1):
        cumulative_df = df[df["iteration_index"] <= k].copy()

        if cumulative_df.empty:
            continue

        hv, pareto_mask = compute_hv_for_rows(
            cumulative_df,
            lower=lower,
            upper=upper,
            mc_samples=mc_samples,
        )

        delta_hv = np.nan if previous_hv is None else hv - previous_hv
        percent_gain_from_previous = (
            np.nan if previous_hv is None else 100.0 * delta_hv / max(previous_hv, 1e-12)
        )

        stage_name = "initial" if k == 0 else f"iter-{k}"

        summary_rows.append(
            {
                "stage": stage_name,
                "iteration_index": k,
                "n_cumulative_candidates": len(cumulative_df),
                "hypervolume_fraction": hv,
                "delta_hv_from_previous": delta_hv,
                "percent_gain_from_previous": percent_gain_from_previous,
                "pareto_size": int(pareto_mask.sum()),
            }
        )

        pareto_df = cumulative_df.loc[pareto_mask].copy()
        pareto_df["pareto_stage"] = stage_name
        pareto_tables.append(pareto_df)

        previous_hv = hv

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{args.output_prefix}_summary.csv", index=False)

    all_pareto = pd.concat(pareto_tables, ignore_index=True)
    all_pareto.to_csv(f"{args.output_prefix}_pareto_by_stage.csv", index=False)

    # Save final Pareto front separately
    final_stage = summary.iloc[-1]["stage"]
    final_pareto = all_pareto[all_pareto["pareto_stage"] == final_stage].copy()
    final_pareto.to_csv(f"{args.output_prefix}_final_pareto.csv", index=False)

    print("\n===== Hypervolume Across AL Iterations =====")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(f"  {args.output_prefix}_summary.csv")
    print(f"  {args.output_prefix}_pareto_by_stage.csv")
    print(f"  {args.output_prefix}_final_pareto.csv")

    # Plot
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(
        summary["n_cumulative_candidates"],
        summary["hypervolume_fraction"],
        marker="o",
        linewidth=2,
        label="AL trajectory",
    )
    plt.xlabel("Number of MD-evaluated candidates")
    plt.ylabel("Normalized dominated hypervolume")
    plt.title("Hypervolume improvement across AL iterations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = f"{args.output_prefix}_trajectory.png"
    plt.savefig(plot_path, dpi=300)
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()
