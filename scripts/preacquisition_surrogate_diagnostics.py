#!/usr/bin/env python3
"""
Chronological pre-acquisition surrogate diagnostics for the EloKriti campaign.

For each acquisition round, the original surrogate is trained only on
successfully completed candidates available before that round and then
evaluated on the candidates acquired in that round.

Outputs:
  preacquisition_predictions.csv
  preacquisition_metrics_overall.csv
  preacquisition_metrics_by_round.csv
  parity_elastic_modulus.png
  parity_volumetric_shrinkage.png
  parity_ffv.png
  calibration_summary.png

Usage:
  python scripts/preacquisition_surrogate_diagnostics.py \
      --evaluated data/evaluated_clean_no_expansion.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


INPUT_COLS = [
    "reactant_1",
    "molecular_weight_Boltzmann_average",
    "logP_Boltzmann_average",
    "TPSA_Boltzmann_average",
    "normalized_monomer_phi_Boltzmann_average",
    "normalized_backbone_phi_Boltzmann_average",
]

OBJECTIVES = ["elastic_modulus", "volumetric_shrinkage", "ffv"]
SIGNS = np.array([1.0, -1.0, -1.0])
ROUND_ORDER = ["initial", "iter-1", "iter-2", "iter-3", "iter-4", "iter-5", "iter-6", "iter-7", "iter-8"]


class OriginalSurrogate:
    """Standalone reproduction of the original surrogate implementation."""

    def __init__(self, random_seed: int = 42, svd_components: int = 8, gp_restarts: int = 1):
        self.random_seed = random_seed
        self.svd_components = svd_components
        self.gp_restarts = gp_restarts
        # IMPORTANT: lowercase is intentionally NOT disabled here.
        # This reproduces the original implementation exactly.
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=12000,
        )
        self.svd = None
        self.scaler = None
        self.models = []

    @staticmethod
    def _build_representation(df: pd.DataFrame) -> np.ndarray:
        return df[INPUT_COLS].astype(str).agg(" ".join, axis=1).to_numpy()

    def _fit_features(self, df: pd.DataFrame) -> np.ndarray:
        texts = self._build_representation(df)
        sparse = self.vectorizer.fit_transform(texts)
        min_dim = min(sparse.shape)
        if min_dim <= 3:
            dense = sparse.toarray()
            self.svd = None
        else:
            n_components = min(self.svd_components, min_dim - 1)
            self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_seed)
            dense = self.svd.fit_transform(sparse)
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(dense)

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        texts = self._build_representation(df)
        sparse = self.vectorizer.transform(texts)
        dense = sparse.toarray() if self.svd is None else self.svd.transform(sparse)
        return self.scaler.transform(dense)

    def fit(self, df: pd.DataFrame, y_signed: np.ndarray):
        X = self._fit_features(df)
        self.models = []
        for j in range(y_signed.shape[1]):
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=1.0, nu=2.5)
                + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=self.gp_restarts,
                random_state=self.random_seed,
            )
            gp.fit(X, y_signed[:, j])
            self.models.append(gp)
        return self

    def predict(self, df: pd.DataFrame):
        X = self._transform_features(df)
        means, stds = [], []
        for gp in self.models:
            mu, sd = gp.predict(X, return_std=True)
            means.append(mu)
            stds.append(sd)
        return np.column_stack(means), np.column_stack(stds)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluated", default="data/evaluated_clean_no_expansion.csv")
    p.add_argument("--outdir", default="outputs/preacquisition_surrogate_diagnostics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--svd", type=int, default=8)
    p.add_argument("--gp-restarts", type=int, default=1)
    return p.parse_args()


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    required = ["reaction_id", "iteration", *INPUT_COLS, *OBJECTIVES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["iteration"] = out["iteration"].astype(str).str.strip()
    for col in OBJECTIVES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[*INPUT_COLS, *OBJECTIVES, "iteration"]).copy()

    unknown = sorted(set(out["iteration"]) - set(ROUND_ORDER))
    if unknown:
        raise ValueError(f"Unknown iteration labels found: {unknown}. Expected labels: {ROUND_ORDER}")

    return out


def metric_row(target: str, y: np.ndarray, pred: np.ndarray, sd: np.ndarray, label: str) -> dict:
    err = pred - y
    spearman = spearmanr(y, pred, nan_policy="omit")
    pearson = pearsonr(y, pred) if len(y) >= 2 else (np.nan, np.nan)
    return {
        "group": label,
        "objective": target,
        "n": len(y),
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred) if len(y) >= 2 else np.nan,
        "Pearson_r": pearson[0],
        "Spearman_rho": spearman.statistic,
        "coverage_68pct": np.mean(np.abs(err) <= sd),
        "coverage_95pct": np.mean(np.abs(err) <= 1.96 * sd),
        "mean_predictive_std": np.mean(sd),
        "std_standardized_residual": np.std(err / np.maximum(sd, 1e-12), ddof=1) if len(y) > 1 else np.nan,
    }


def make_parity(pred_df: pd.DataFrame, target: str, outpath: Path):
    y = pred_df[f"actual_{target}"].to_numpy(float)
    p = pred_df[f"pred_{target}"].to_numpy(float)
    s = pred_df[f"std_{target}"].to_numpy(float)

    lo = min(np.min(y), np.min(p))
    hi = max(np.max(y), np.max(p))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.errorbar(y, p, yerr=1.96 * s, fmt="o", alpha=0.75, capsize=2)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--")
    ax.set_xlabel(f"MD {target}")
    ax.set_ylabel(f"Pre-acquisition GP prediction: {target}")
    ax.set_title(f"Pre-acquisition parity: {target}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = clean_and_validate(pd.read_csv(args.evaluated))

    rows = []
    for round_idx in range(1, len(ROUND_ORDER)):
        test_label = ROUND_ORDER[round_idx]
        prior_labels = set(ROUND_ORDER[:round_idx])

        train = df[df["iteration"].isin(prior_labels)].copy()
        test = df[df["iteration"] == test_label].copy()
        if test.empty:
            print(f"Skipping {test_label}: no successful evaluated rows found.")
            continue

        y_train_raw = train[OBJECTIVES].to_numpy(float)
        y_train_signed = y_train_raw * SIGNS

        model = OriginalSurrogate(
            random_seed=args.seed,
            svd_components=args.svd,
            gp_restarts=args.gp_restarts,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(train, y_train_signed)

        mu_signed, sd = model.predict(test)
        mu_raw = mu_signed * SIGNS

        for local_i, (_, r) in enumerate(test.iterrows()):
            rec = {
                "reaction_id": r["reaction_id"],
                "iteration": test_label,
                "n_train_before_round": len(train),
            }
            for j, obj in enumerate(OBJECTIVES):
                rec[f"actual_{obj}"] = float(r[obj])
                rec[f"pred_{obj}"] = float(mu_raw[local_i, j])
                rec[f"std_{obj}"] = float(sd[local_i, j])
                rec[f"z_{obj}"] = float((r[obj] - mu_raw[local_i, j]) / max(sd[local_i, j], 1e-12))
            rows.append(rec)

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise RuntimeError("No pre-acquisition predictions were generated. Check iteration labels.")

    pred_df.to_csv(outdir / "preacquisition_predictions.csv", index=False)

    overall = []
    for obj in OBJECTIVES:
        overall.append(metric_row(
            obj,
            pred_df[f"actual_{obj}"].to_numpy(float),
            pred_df[f"pred_{obj}"].to_numpy(float),
            pred_df[f"std_{obj}"].to_numpy(float),
            "all_acquired_successful_candidates",
        ))
    pd.DataFrame(overall).to_csv(outdir / "preacquisition_metrics_overall.csv", index=False)

    by_round = []
    for iteration, g in pred_df.groupby("iteration", sort=False):
        for obj in OBJECTIVES:
            by_round.append(metric_row(
                obj,
                g[f"actual_{obj}"].to_numpy(float),
                g[f"pred_{obj}"].to_numpy(float),
                g[f"std_{obj}"].to_numpy(float),
                iteration,
            ))
    pd.DataFrame(by_round).to_csv(outdir / "preacquisition_metrics_by_round.csv", index=False)

    for obj in OBJECTIVES:
        make_parity(pred_df, obj, outdir / f"parity_{obj}.png")

    # Compact coverage figure.
    m = pd.DataFrame(overall)
    x = np.arange(len(OBJECTIVES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(x - width / 2, m["coverage_68pct"], width, label="within ±1σ")
    ax.bar(x + width / 2, m["coverage_95pct"], width, label="within ±1.96σ")
    ax.axhline(0.68, linestyle="--", linewidth=1)
    ax.axhline(0.95, linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(OBJECTIVES, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Pre-acquisition predictive-interval coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "calibration_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nDONE")
    print(f"Predicted acquired candidates: {len(pred_df)}")
    print(f"Results written to: {outdir.resolve()}")
    print("\nOverall metrics:")
    print(pd.DataFrame(overall).to_string(index=False))


if __name__ == "__main__":
    main()
