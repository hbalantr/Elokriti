#!/usr/bin/env python3
"""
Independent PI1070 candidate-selection benchmark.

Compares:
  1) EloKriti-style surrogate-guided selection
  2) uniform random selection
  3) greedy maximin space-filling selection

The three methods use the same initial set, batch schedule, evaluation budget,
objective normalization, and hypervolume reference within each seed.

Default objectives:
  maximize bulk_modulus
  maximize thermal_conductivity
  minimize density

Default campaign-matched budgets:
  10, 19, 27, 35, 41, 50, 57, 66, 71

Usage:
  python scripts/pi1070_selection_benchmark.py
  python scripts/pi1070_selection_benchmark.py --dataset data/PI1070.csv --seeds 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from polymer_bo_emulator.bo import parego_acquisition_scores
from polymer_bo_emulator.surrogate import FeatureConfig, MultiTargetGPSurrogate


OBJECTIVE_NAMES = ["bulk_modulus", "thermal_conductivity", "density"]
OBJECTIVE_DIRECTIONS = ["max", "max", "min"]
SIGNS = np.array([1.0, 1.0, -1.0], dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/PI1070.csv")
    p.add_argument("--output-dir", default="outputs/pi1070_selection_benchmark")
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--n-initial", type=int, default=10)
    p.add_argument(
        "--batch-sizes",
        default="9,8,8,6,9,7,9,5",
        help="Comma-separated batch sizes. Default reproduces successful dental-campaign counts.",
    )
    p.add_argument("--scalarizations", type=int, default=64)
    p.add_argument("--svd-components", type=int, default=8)
    p.add_argument("--max-tfidf-features", type=int, default=12000)
    p.add_argument("--gp-restarts", type=int, default=1)
    p.add_argument("--feature-seed", type=int, default=42)
    p.add_argument("--xi", type=float, default=0.01)
    return p.parse_args()


def pareto_mask_maximize(y: np.ndarray) -> np.ndarray:
    """Return mask of non-dominated rows for maximize-all objectives."""
    n = y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates = np.all(y[i] >= y, axis=1) & np.any(y[i] > y, axis=1)
        keep[dominates] = False
        keep[i] = True
    return keep


def normalize_full_space(y_max: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.maximum(hi - lo, 1e-12)
    return np.clip((y_max - lo) / span, 0.0, 1.0)


def hv2d_origin(points: np.ndarray) -> float:
    """Exact union area of rectangles from origin to 2D points in [0,1]^2."""
    if points.size == 0:
        return 0.0
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(pts > 0.0, axis=1)]
    if pts.size == 0:
        return 0.0

    y_levels = np.sort(np.unique(pts[:, 0]))[::-1]
    area = 0.0
    for i, y_hi in enumerate(y_levels):
        y_lo = y_levels[i + 1] if i + 1 < len(y_levels) else 0.0
        active = pts[pts[:, 0] >= y_hi - 1e-15]
        z_max = float(np.max(active[:, 1]))
        area += float(y_hi - y_lo) * z_max
    return area


def hv3d_origin(points: np.ndarray) -> float:
    """Exact dominated hypervolume in 3D relative to reference origin."""
    if points.size == 0:
        return 0.0
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(pts > 0.0, axis=1)]
    if pts.size == 0:
        return 0.0

    # Removing dominated points is not required for correctness, but speeds the slices.
    pts = pts[pareto_mask_maximize(pts)]

    x_levels = np.sort(np.unique(pts[:, 0]))[::-1]
    volume = 0.0
    for i, x_hi in enumerate(x_levels):
        x_lo = x_levels[i + 1] if i + 1 < len(x_levels) else 0.0
        active = pts[pts[:, 0] >= x_hi - 1e-15]
        yz_area = hv2d_origin(active[:, 1:3])
        volume += float(x_hi - x_lo) * yz_area
    return volume


def load_pi1070(path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    needed = ["smiles", *OBJECTIVE_NAMES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required PI1070 columns: {missing}")

    work = df[needed].copy()
    for c in OBJECTIVE_NAMES:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work["smiles"] = work["smiles"].astype(str)
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).reset_index(drop=True)

    reps = work["smiles"].astype(str).to_numpy()
    targets = work[OBJECTIVE_NAMES].to_numpy(dtype=float)
    return work, reps, targets


def build_space_filling_distances(
    reps: np.ndarray,
    svd_components: int,
    max_features: int,
    feature_seed: int,
) -> np.ndarray:
    """
    Unsupervised molecular-space embedding used only for the space-filling baseline.

    It uses the same broad representation family as the emulator:
    character n-gram TF-IDF -> truncated SVD -> scaling.
    No target property is used.
    """
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=max_features,
    )
    x_sparse = vec.fit_transform(reps)
    max_comp = max(2, min(x_sparse.shape) - 1)
    n_comp = min(svd_components, max_comp)
    svd = TruncatedSVD(n_components=n_comp, random_state=feature_seed)
    x = svd.fit_transform(x_sparse)
    x = StandardScaler().fit_transform(x)
    return cdist(x, x, metric="euclidean")


def metrics_for_selected(
    selected: list[int],
    y_max: np.ndarray,
    y_norm: np.ndarray,
    true_pareto_set: set[int],
    full_hv: float,
) -> tuple[float, float, int]:
    idx = np.asarray(selected, dtype=int)
    front_mask = pareto_mask_maximize(y_max[idx])
    front_norm = y_norm[idx][front_mask]
    hv = hv3d_origin(front_norm)
    hv_recovery = hv / full_hv if full_hv > 0 else np.nan
    n_true_found = len(set(selected).intersection(true_pareto_set))
    recall = n_true_found / len(true_pareto_set) if true_pareto_set else np.nan
    return float(hv_recovery), float(recall), int(n_true_found)


def run_bo(
    reps: np.ndarray,
    targets: np.ndarray,
    initial: np.ndarray,
    batch_sizes: list[int],
    seed: int,
    scalarizations: int,
    svd_components: int,
    max_tfidf_features: int,
    gp_restarts: int,
    feature_seed: int,
    xi: float,
    y_max: np.ndarray,
    y_norm: np.ndarray,
    true_pareto_set: set[int],
    full_hv: float,
) -> list[dict]:
    observed = [int(i) for i in initial]
    all_indices = np.arange(len(reps))
    unobserved = set(int(i) for i in all_indices if i not in observed)

    rows = []
    hv_rec, recall, n_found = metrics_for_selected(
        observed, y_max, y_norm, true_pareto_set, full_hv
    )
    rows.append(
        {
            "seed": seed,
            "method": "EloKriti",
            "round": 0,
            "budget": len(observed),
            "hv_recovery": hv_rec,
            "pareto_recall": recall,
            "true_pareto_found": n_found,
        }
    )

    feature_config = FeatureConfig(
        max_tfidf_features=max_tfidf_features,
        ngram_range=(2, 5),
        svd_components=svd_components,
        random_seed=feature_seed,
        use_rdkit_features=False,
    )

    for round_idx, requested_batch in enumerate(batch_sizes, start=1):
        if not unobserved:
            break

        obs = np.asarray(observed, dtype=int)
        unobs = np.asarray(sorted(unobserved), dtype=int)

        # Match the paper's suggestion workflow: fit GPs in maximize-space.
        y_train_max = targets[obs] * SIGNS

        model = MultiTargetGPSurrogate(
            feature_config=feature_config,
            gp_restarts=gp_restarts,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(
                texts=reps[obs],
                targets=y_train_max,
                target_names=OBJECTIVE_NAMES,
            )
            pred_mean_max, pred_std = model.predict(reps[unobs])

        scores = parego_acquisition_scores(
            observed_targets=y_train_max,
            predicted_mean=pred_mean_max,
            predicted_std=pred_std,
            objective_directions=["max"] * len(OBJECTIVE_NAMES),
            random_seed=seed * 1000 + round_idx,
            n_scalarizations=scalarizations,
            xi=xi,
        )

        batch_n = min(requested_batch, len(unobs))
        # Stable deterministic tie-breaking by candidate index.
        ranking = np.lexsort((unobs, -scores))
        chosen = unobs[ranking[:batch_n]]

        for idx in chosen:
            observed.append(int(idx))
            unobserved.remove(int(idx))

        hv_rec, recall, n_found = metrics_for_selected(
            observed, y_max, y_norm, true_pareto_set, full_hv
        )
        rows.append(
            {
                "seed": seed,
                "method": "EloKriti",
                "round": round_idx,
                "budget": len(observed),
                "hv_recovery": hv_rec,
                "pareto_recall": recall,
                "true_pareto_found": n_found,
            }
        )
    return rows


def run_random(
    n_samples: int,
    initial: np.ndarray,
    batch_sizes: list[int],
    seed: int,
    y_max: np.ndarray,
    y_norm: np.ndarray,
    true_pareto_set: set[int],
    full_hv: float,
) -> list[dict]:
    observed = [int(i) for i in initial]
    unobserved = [i for i in range(n_samples) if i not in set(observed)]
    rng = np.random.default_rng(seed + 100_000)

    rows = []
    hv_rec, recall, n_found = metrics_for_selected(
        observed, y_max, y_norm, true_pareto_set, full_hv
    )
    rows.append(
        {
            "seed": seed,
            "method": "Random",
            "round": 0,
            "budget": len(observed),
            "hv_recovery": hv_rec,
            "pareto_recall": recall,
            "true_pareto_found": n_found,
        }
    )

    remaining = np.asarray(unobserved, dtype=int)
    for round_idx, requested_batch in enumerate(batch_sizes, start=1):
        if len(remaining) == 0:
            break
        batch_n = min(requested_batch, len(remaining))
        chosen = rng.choice(remaining, size=batch_n, replace=False)
        chosen_set = set(int(i) for i in chosen)
        observed.extend(int(i) for i in chosen)
        remaining = np.asarray([i for i in remaining if int(i) not in chosen_set], dtype=int)

        hv_rec, recall, n_found = metrics_for_selected(
            observed, y_max, y_norm, true_pareto_set, full_hv
        )
        rows.append(
            {
                "seed": seed,
                "method": "Random",
                "round": round_idx,
                "budget": len(observed),
                "hv_recovery": hv_rec,
                "pareto_recall": recall,
                "true_pareto_found": n_found,
            }
        )
    return rows


def run_space_filling(
    distances: np.ndarray,
    initial: np.ndarray,
    batch_sizes: list[int],
    seed: int,
    y_max: np.ndarray,
    y_norm: np.ndarray,
    true_pareto_set: set[int],
    full_hv: float,
) -> list[dict]:
    observed = [int(i) for i in initial]
    selected_mask = np.zeros(distances.shape[0], dtype=bool)
    selected_mask[observed] = True

    rows = []
    hv_rec, recall, n_found = metrics_for_selected(
        observed, y_max, y_norm, true_pareto_set, full_hv
    )
    rows.append(
        {
            "seed": seed,
            "method": "Space-filling",
            "round": 0,
            "budget": len(observed),
            "hv_recovery": hv_rec,
            "pareto_recall": recall,
            "true_pareto_found": n_found,
        }
    )

    # Current distance from every point to the selected set.
    min_dist = np.min(distances[:, observed], axis=1)
    min_dist[selected_mask] = -np.inf

    for round_idx, requested_batch in enumerate(batch_sizes, start=1):
        chosen_this_round = []
        for _ in range(requested_batch):
            if np.all(selected_mask):
                break
            # Deterministic tie-breaking: np.argmax returns the first maximum.
            nxt = int(np.argmax(min_dist))
            selected_mask[nxt] = True
            observed.append(nxt)
            chosen_this_round.append(nxt)
            min_dist = np.minimum(min_dist, distances[:, nxt])
            min_dist[selected_mask] = -np.inf

        if not chosen_this_round:
            break

        hv_rec, recall, n_found = metrics_for_selected(
            observed, y_max, y_norm, true_pareto_set, full_hv
        )
        rows.append(
            {
                "seed": seed,
                "method": "Space-filling",
                "round": round_idx,
                "budget": len(observed),
                "hv_recovery": hv_rec,
                "pareto_recall": recall,
                "true_pareto_found": n_found,
            }
        )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (method, budget), g in raw.groupby(["method", "budget"], sort=True):
        n = len(g)
        hv_mean = g["hv_recovery"].mean()
        hv_sd = g["hv_recovery"].std(ddof=1)
        hv_se = hv_sd / np.sqrt(n) if n > 1 else np.nan
        rec_mean = g["pareto_recall"].mean()
        rec_sd = g["pareto_recall"].std(ddof=1)
        rec_se = rec_sd / np.sqrt(n) if n > 1 else np.nan
        records.append(
            {
                "method": method,
                "budget": int(budget),
                "n_seeds": n,
                "hv_recovery_mean": hv_mean,
                "hv_recovery_sd": hv_sd,
                "hv_recovery_ci95_low": hv_mean - 1.96 * hv_se if n > 1 else np.nan,
                "hv_recovery_ci95_high": hv_mean + 1.96 * hv_se if n > 1 else np.nan,
                "pareto_recall_mean": rec_mean,
                "pareto_recall_sd": rec_sd,
                "pareto_recall_ci95_low": rec_mean - 1.96 * rec_se if n > 1 else np.nan,
                "pareto_recall_ci95_high": rec_mean + 1.96 * rec_se if n > 1 else np.nan,
            }
        )
    return pd.DataFrame(records)


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(7.0, 4.8))
    for method in ["EloKriti", "Random", "Space-filling"]:
        g = summary[summary["method"] == method].sort_values("budget")
        mean_col = f"{metric}_mean"
        low_col = f"{metric}_ci95_low"
        high_col = f"{metric}_ci95_high"
        plt.plot(g["budget"], g[mean_col], marker="o", label=method)
        plt.fill_between(g["budget"], g[low_col], g[high_col], alpha=0.18)
    plt.xlabel("Number of evaluated polymers")
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> int:
    args = parse_args()
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    if not batch_sizes or any(x <= 0 for x in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    table, reps, targets = load_pi1070(args.dataset)
    n_samples = len(table)
    if n_samples < args.n_initial + sum(batch_sizes):
        raise ValueError(
            f"Dataset has {n_samples} valid rows, but benchmark requires "
            f"{args.n_initial + sum(batch_sizes)}."
        )

    # Fixed full-data objective transform and reference region, common to every method/seed.
    y_max = targets * SIGNS
    lower = np.min(y_max, axis=0)
    upper = np.max(y_max, axis=0)
    y_norm = normalize_full_space(y_max, lower, upper)

    true_mask = pareto_mask_maximize(y_max)
    true_pareto_indices = np.flatnonzero(true_mask)
    true_pareto_set = set(int(i) for i in true_pareto_indices)
    full_hv = hv3d_origin(y_norm[true_mask])

    ref = pd.DataFrame(
        {
            "n_valid_polymers": [n_samples],
            "n_true_pareto": [len(true_pareto_indices)],
            "full_dataset_hypervolume": [full_hv],
            "bulk_modulus_min_maxspace": [lower[0]],
            "bulk_modulus_max_maxspace": [upper[0]],
            "thermal_conductivity_min_maxspace": [lower[1]],
            "thermal_conductivity_max_maxspace": [upper[1]],
            "neg_density_min_maxspace": [lower[2]],
            "neg_density_max_maxspace": [upper[2]],
        }
    )
    ref.to_csv(out / "full_reference.csv", index=False)

    table.loc[true_mask].to_csv(out / "true_full_dataset_pareto.csv", index=False)

    # Build a target-free space-filling representation once.
    distances = build_space_filling_distances(
        reps=reps,
        svd_components=args.svd_components,
        max_features=args.max_tfidf_features,
        feature_seed=args.feature_seed,
    )

    all_rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        initial = rng.choice(np.arange(n_samples), size=args.n_initial, replace=False)

        all_rows.extend(
            run_bo(
                reps=reps,
                targets=targets,
                initial=initial,
                batch_sizes=batch_sizes,
                seed=seed,
                scalarizations=args.scalarizations,
                svd_components=args.svd_components,
                max_tfidf_features=args.max_tfidf_features,
                gp_restarts=args.gp_restarts,
                feature_seed=args.feature_seed,
                xi=args.xi,
                y_max=y_max,
                y_norm=y_norm,
                true_pareto_set=true_pareto_set,
                full_hv=full_hv,
            )
        )
        all_rows.extend(
            run_random(
                n_samples=n_samples,
                initial=initial,
                batch_sizes=batch_sizes,
                seed=seed,
                y_max=y_max,
                y_norm=y_norm,
                true_pareto_set=true_pareto_set,
                full_hv=full_hv,
            )
        )
        all_rows.extend(
            run_space_filling(
                distances=distances,
                initial=initial,
                batch_sizes=batch_sizes,
                seed=seed,
                y_max=y_max,
                y_norm=y_norm,
                true_pareto_set=true_pareto_set,
                full_hv=full_hv,
            )
        )

        print(f"Completed seed {seed + 1}/{args.seeds}")

    raw = pd.DataFrame(all_rows)
    raw.to_csv(out / "benchmark_raw.csv", index=False)

    summary = summarize(raw)
    summary.to_csv(out / "benchmark_summary.csv", index=False)

    plot_metric(
        summary,
        metric="hv_recovery",
        ylabel="Fraction of full-dataset hypervolume recovered",
        path=out / "hv_recovery.png",
    )
    plot_metric(
        summary,
        metric="pareto_recall",
        ylabel="Full-dataset Pareto recall",
        path=out / "pareto_recall.png",
    )

    # Convenient table at the campaign milestone budgets.
    milestone_budgets = [10, 19, 27, 35, 41, 50, 57, 66, 71]
    milestone = summary[summary["budget"].isin(milestone_budgets)].copy()
    milestone.to_csv(out / "milestone_summary.csv", index=False)

    print("\nDONE")
    print(f"Valid PI1070 polymers: {n_samples}")
    print(f"True full-data Pareto size: {len(true_pareto_indices)}")
    print(f"Exact full-data normalized HV: {full_hv:.6f}")
    print(f"Results written to: {out.resolve()}")
    print("\nFinal-budget summary:")
    final_budget = args.n_initial + sum(batch_sizes)
    print(
        summary[summary["budget"] == final_budget][
            [
                "method",
                "hv_recovery_mean",
                "hv_recovery_ci95_low",
                "hv_recovery_ci95_high",
                "pareto_recall_mean",
                "pareto_recall_ci95_low",
                "pareto_recall_ci95_high",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
