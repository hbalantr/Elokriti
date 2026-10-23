#!/usr/bin/env python3
"""
Chronological pre-acquisition comparison of the original EloKriti GP surrogate
against a simple standardized linear-regression baseline.

For each acquisition round, both models are trained only on candidates
available before that round and evaluated on candidates acquired in that round.

Outputs:
  surrogate_baseline_roundwise.csv
  surrogate_baseline_aggregate.csv

Usage:
  python scripts/surrogate_baseline_comparison.py \
      --evaluated data/evaluated_clean_no_expansion.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

DESC_COLS = [
    "molecular_weight_Boltzmann_average",
    "logP_Boltzmann_average",
    "TPSA_Boltzmann_average",
    "normalized_monomer_phi_Boltzmann_average",
    "normalized_backbone_phi_Boltzmann_average",
]
OBJECTIVES = ["elastic_modulus", "volumetric_shrinkage", "ffv"]
SMILES_COL = "reactant_1"
SEED = 42


def parse_iteration(v):
    s = str(v).strip().lower().replace("_", "-").replace(" ", "")
    if s in {"initial", "init", "seed", "d0", "0"}:
        return 0
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
    raise ValueError(f"Cannot parse iteration {v!r}")


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    req = ["reaction_id", "iteration", SMILES_COL, *DESC_COLS, *OBJECTIVES]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns: {miss}")

    if "status" in df.columns:
        st = df["status"].astype(str).str.lower().str.strip()
        df = df[st.isin(["completed", "complete", "valid", "done"])].copy()

    for c in [*DESC_COLS, *OBJECTIVES]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )

    df = df.dropna(subset=[SMILES_COL, *DESC_COLS, *OBJECTIVES]).copy()
    df["_round"] = df["iteration"].apply(parse_iteration)
    df = df.sort_values(["_round", "reaction_id"]).reset_index(drop=True)
    return df


def original_text(df):
    rows = []
    for _, r in df.iterrows():
        vals = [str(r[SMILES_COL])] + [str(float(r[c])) for c in DESC_COLS]
        rows.append(" ".join(vals))
    return rows


def original_features(train, test):
    v = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=12000,
        # lowercase=True is sklearn default and reproduces the original campaign
    )
    A_tr = v.fit_transform(original_text(train))
    A_te = v.transform(original_text(test))

    n_components = min(8, max(2, len(train) - 2), min(A_tr.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    Z_tr = svd.fit_transform(A_tr)
    Z_te = svd.transform(A_te)

    sc = StandardScaler()
    return sc.fit_transform(Z_tr), sc.transform(Z_te)


def make_gp():
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=SEED,
    )


def metrics(y, p):
    rho = spearmanr(y, p).statistic if len(set(p)) > 1 else np.nan
    return {
        "n": len(y),
        "mae": mean_absolute_error(y, p),
        "rmse": np.sqrt(mean_squared_error(y, p)),
        "r2": r2_score(y, p),
        "spearman_rho": rho,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluated", default="data/evaluated_clean_no_expansion.csv")
    ap.add_argument("--output-dir", "--out-dir", dest="output_dir", default="outputs/surrogate_baseline_comparison")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.evaluated)

    records = []
    all_store = {
        (model, obj): {"y": [], "p": [], "std": []}
        for model in ["gp_original", "linear_descriptor_baseline"]
        for obj in OBJECTIVES
    }

    for r in sorted(x for x in df["_round"].unique() if x > 0):
        train = df[df["_round"] < r].copy()
        test = df[df["_round"] == r].copy()
        if test.empty:
            continue

        Xgp_tr, Xgp_te = original_features(train, test)

        dsc = StandardScaler()
        Xlin_tr = dsc.fit_transform(train[DESC_COLS].to_numpy(float))
        Xlin_te = dsc.transform(test[DESC_COLS].to_numpy(float))

        for obj in OBJECTIVES:
            ytr = train[obj].to_numpy(float)
            yte = test[obj].to_numpy(float)

            gp = make_gp()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                gp.fit(Xgp_tr, ytr)
            pgp, sgp = gp.predict(Xgp_te, return_std=True)

            lm = LinearRegression()
            lm.fit(Xlin_tr, ytr)
            plin = lm.predict(Xlin_te)

            for model, pred, std in [
                ("gp_original", pgp, sgp),
                ("linear_descriptor_baseline", plin, np.full(len(yte), np.nan)),
            ]:
                m = metrics(yte, pred)
                if model == "gp_original":
                    m["coverage_95"] = float(np.mean(np.abs(yte - pred) <= 1.96 * std))
                else:
                    m["coverage_95"] = np.nan

                records.append({
                    "round": r,
                    "model": model,
                    "objective": obj,
                    **m,
                })

                all_store[(model, obj)]["y"].extend(yte.tolist())
                all_store[(model, obj)]["p"].extend(pred.tolist())
                all_store[(model, obj)]["std"].extend(std.tolist())

    round_df = pd.DataFrame(records)
    round_df.to_csv(out / "surrogate_baseline_roundwise.csv", index=False)

    agg = []
    for (model, obj), d in all_store.items():
        y = np.asarray(d["y"], float)
        p = np.asarray(d["p"], float)
        m = metrics(y, p)
        if model == "gp_original":
            s = np.asarray(d["std"], float)
            m["coverage_95"] = float(np.mean(np.abs(y - p) <= 1.96 * s))
        else:
            m["coverage_95"] = np.nan

        agg.append({
            "model": model,
            "objective": obj,
            **m,
        })

    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(out / "surrogate_baseline_aggregate.csv", index=False)

    print("\n===== AGGREGATE PRE-ACQUISITION COMPARISON =====")
    print(agg_df.to_string(index=False))
    print(f"\nSaved in: {out.resolve()}")


if __name__ == "__main__":
    main()
