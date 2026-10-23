#!/usr/bin/env python3
"""
PI1070 scalarization and batch-selection sensitivity benchmark.

Tests:
  - 16, 32, 64, 128, and 256 random linear scalarizations
  - the original top-k batch rule
  - a diversity-aware 64-scalarization batch rule

Default campaign-matched budget:
  10 initial + [9, 8, 8, 6, 9, 7] = 57 evaluated polymers

Reported metrics:
  - Pareto recall
  - normalized 3D hypervolume recovery
  - supported/unsupported Pareto recall
  - within-batch SMILES-TFIDF cosine distance

Usage:
  python scripts/scalarization_batch_benchmark.py
  python scripts/scalarization_batch_benchmark.py \
      --data data/PI1070.csv \
      --seeds 20
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import norm

from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler


BASE_SEED = 42
OBJECTIVES = ["bulk_modulus", "thermal_conductivity", "density"]
SIGNS = np.array([1.0, 1.0, -1.0])
BATCH_SCHEDULE = [9, 8, 8, 6, 9, 7]   # 10 -> 57
SCALAR_COUNTS = [16, 32, 64, 128, 256]


def pareto_mask_maximize(Y: np.ndarray) -> np.ndarray:
    n = Y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(Y[i] >= Y, axis=1) & np.any(Y[i] > Y, axis=1)
        keep[dominated] = False
        keep[i] = True
    return keep


def hv2_origin(points: np.ndarray) -> float:
    """Exact dominated area in [0,1]^2 for maximize-space points vs origin."""
    if points.size == 0:
        return 0.0
    p = np.clip(points, 0.0, 1.0)
    p = p[pareto_mask_maximize(p)]
    xs = np.unique(p[:, 0])
    prev = 0.0
    area = 0.0
    for x in xs:
        active = p[p[:, 0] >= x]
        ymax = float(np.max(active[:, 1])) if len(active) else 0.0
        area += (x - prev) * ymax
        prev = x
    return float(area)


def hv3_origin(points: np.ndarray) -> float:
    """Exact dominated volume in [0,1]^3 for maximize-space points vs origin."""
    if points.size == 0:
        return 0.0
    p = np.clip(points, 0.0, 1.0)
    p = p[pareto_mask_maximize(p)]
    xs = np.unique(p[:, 0])
    prev = 0.0
    vol = 0.0
    for x in xs:
        active = p[p[:, 0] >= x][:, 1:]
        vol += (x - prev) * hv2_origin(active)
        prev = x
    return float(vol)


def normalize_full(Ymax: np.ndarray):
    lo = Ymax.min(axis=0)
    hi = Ymax.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    return np.clip((Ymax - lo) / span, 0.0, 1.0), lo, hi


def expected_improvement(mu, sigma, best, xi=0.01):
    s = np.maximum(sigma, 1e-12)
    imp = mu - best - xi
    z = imp / s
    out = imp * norm.cdf(z) + s * norm.pdf(z)
    out[sigma < 1e-12] = np.maximum(0.0, imp[sigma < 1e-12])
    return out


def make_gp(seed):
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
        random_state=seed,
    )


def build_round_features(smiles_eval, smiles_unobs):
    """Reproduce EloKriti-like campaign representation on SMILES strings."""
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=12000,
        # default lowercase=True, intentionally matching original implementation
    )
    A_eval = vec.fit_transform(smiles_eval)
    A_un = vec.transform(smiles_unobs)

    ncomp = max(1, min(8, max(1, len(smiles_eval) - 2), min(A_eval.shape) - 1))
    svd = TruncatedSVD(n_components=ncomp, random_state=BASE_SEED)
    X_eval = svd.fit_transform(A_eval)
    X_un = svd.transform(A_un)

    sc = StandardScaler()
    return sc.fit_transform(X_eval), sc.transform(X_un)


def acquisition_scores(Y_eval_raw, mu_raw, std_raw, n_scalar, rng_seed):
    """Averaged EI over Dirichlet random linear scalarizations."""
    Y_eval = Y_eval_raw * SIGNS
    mu = mu_raw * SIGNS

    center = Y_eval.mean(axis=0)
    scale = Y_eval.std(axis=0)
    scale[scale < 1e-8] = 1.0

    rng = np.random.default_rng(rng_seed)
    scores = np.zeros(len(mu), dtype=float)

    for _ in range(n_scalar):
        w = rng.dirichlet(np.ones(3))
        obs_s = ((Y_eval - center) / scale) @ w
        best = float(np.max(obs_s))

        mu_s = ((mu - center) / scale) @ w
        sd_s = np.sqrt(np.sum((w * (std_raw / scale)) ** 2, axis=1))
        scores += expected_improvement(mu_s, sd_s, best, xi=0.01)

    return scores / n_scalar


def mean_pairwise_cosine_distance(full_tfidf: csr_matrix, idx):
    idx = list(idx)
    if len(idx) < 2:
        return np.nan
    D = cosine_distances(full_tfidf[idx])
    upper = D[np.triu_indices_from(D, k=1)]
    return float(np.mean(upper))


def choose_topk(scores, unobs_idx, k):
    order = np.argsort(-scores)
    return np.asarray(unobs_idx)[order[:k]]


def choose_diverse(scores, unobs_idx, k, full_tfidf, shortlist_factor=5):
    """Acquisition shortlist + greedy maximin chemical diversity."""
    unobs_idx = np.asarray(unobs_idx)
    order = np.argsort(-scores)
    m = min(len(order), max(k, shortlist_factor * k))
    short_local = order[:m]
    short_global = unobs_idx[short_local]

    selected = [int(short_global[0])]
    remaining = [int(x) for x in short_global[1:]]

    while len(selected) < min(k, len(short_global)):
        D = cosine_distances(full_tfidf[remaining], full_tfidf[selected])
        min_d = D.min(axis=1)
        # Primary criterion = maximum minimum distance.
        # Acquisition score breaks near ties.
        best_d = np.max(min_d)
        candidate_pos = np.where(np.isclose(min_d, best_d, rtol=1e-10, atol=1e-12))[0]
        if len(candidate_pos) > 1:
            global_to_local = {int(g): int(l) for l, g in enumerate(unobs_idx)}
            tie_scores = np.array([scores[global_to_local[remaining[p]]] for p in candidate_pos])
            p = int(candidate_pos[np.argmax(tie_scores)])
        else:
            p = int(candidate_pos[0])

        selected.append(remaining.pop(p))

    return np.asarray(selected[:k], dtype=int)


def classify_supported_pareto(Ynorm, pareto_mask, n_weights=50000, seed=12345):
    """Approximate supported Pareto points under positive linear scalarizations."""
    pareto_idx = np.flatnonzero(pareto_mask)
    supported = set()
    rng = np.random.default_rng(seed)

    chunk = 2000
    left = n_weights
    while left > 0:
        c = min(chunk, left)
        W = rng.dirichlet(np.ones(3), size=c)
        S = Ynorm @ W.T
        winners = np.argmax(S, axis=0)
        for x in np.unique(winners):
            if pareto_mask[x]:
                supported.add(int(x))
        left -= c

    supported_mask = np.zeros(len(Ynorm), dtype=bool)
    if supported:
        supported_mask[list(supported)] = True
    supported_mask &= pareto_mask
    unsupported_mask = pareto_mask & (~supported_mask)
    return supported_mask, unsupported_mask


def evaluate_selected(selected, Ynorm, true_pareto, supported, unsupported, full_hv):
    selected = np.asarray(selected, dtype=int)
    sel_set = set(selected.tolist())

    p_idx = set(np.flatnonzero(true_pareto).tolist())
    s_idx = set(np.flatnonzero(supported).tolist())
    u_idx = set(np.flatnonzero(unsupported).tolist())

    recall = len(sel_set & p_idx) / max(1, len(p_idx))
    srec = len(sel_set & s_idx) / max(1, len(s_idx))
    urec = len(sel_set & u_idx) / max(1, len(u_idx))

    hv = hv3_origin(Ynorm[selected])
    return recall, srec, urec, hv / max(full_hv, 1e-15)


def run_one(df, Yraw, Ynorm, full_tfidf, true_pareto, supported, unsupported,
            full_hv, init_idx, method, n_scalar, seed):
    selected = list(map(int, init_idx))
    rows = []
    batch_rows = []

    # Initial point metrics
    rec, srec, urec, hvrec = evaluate_selected(
        selected, Ynorm, true_pareto, supported, unsupported, full_hv
    )
    rows.append(dict(
        seed=seed, method=method, n_scalarizations=n_scalar,
        budget=len(selected), pareto_recall=rec,
        supported_pareto_recall=srec, unsupported_pareto_recall=urec,
        hv_recovery=hvrec
    ))

    for round_no, batch_size in enumerate(BATCH_SCHEDULE, start=1):
        mask = np.ones(len(df), dtype=bool)
        mask[selected] = False
        unobs_idx = np.flatnonzero(mask)

        Xev, Xun = build_round_features(
            df.loc[selected, "smiles"].astype(str).tolist(),
            df.loc[unobs_idx, "smiles"].astype(str).tolist(),
        )

        mu = np.zeros((len(unobs_idx), 3), dtype=float)
        sd = np.zeros((len(unobs_idx), 3), dtype=float)

        for j in range(3):
            gp = make_gp(BASE_SEED + seed * 100 + round_no * 10 + j)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                gp.fit(Xev, Yraw[selected, j])
            mu[:, j], sd[:, j] = gp.predict(Xun, return_std=True)

        acq = acquisition_scores(
            Yraw[selected], mu, sd, n_scalar,
            rng_seed=100000 + seed * 1000 + round_no,
        )

        if method.startswith("linear_topk"):
            batch = choose_topk(acq, unobs_idx, batch_size)
        elif method == "linear_diverse_n64":
            batch = choose_diverse(acq, unobs_idx, batch_size, full_tfidf, shortlist_factor=5)
        else:
            raise ValueError(method)

        div = mean_pairwise_cosine_distance(full_tfidf, batch)
        batch_rows.append(dict(
            seed=seed, method=method, n_scalarizations=n_scalar,
            round=round_no, batch_size=len(batch),
            mean_pairwise_tfidf_cosine_distance=div
        ))

        selected.extend(batch.tolist())

        rec, srec, urec, hvrec = evaluate_selected(
            selected, Ynorm, true_pareto, supported, unsupported, full_hv
        )
        rows.append(dict(
            seed=seed, method=method, n_scalarizations=n_scalar,
            budget=len(selected), pareto_recall=rec,
            supported_pareto_recall=srec, unsupported_pareto_recall=urec,
            hv_recovery=hvrec
        ))

    return rows, batch_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/PI1070.csv")
    ap.add_argument("--output-dir", "--out-dir", dest="output_dir", default="outputs/scalarization_batch_benchmark")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--supported-weights", type=int, default=50000)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()

    required = ["smiles", *OBJECTIVES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"PI1070 is missing required columns: {missing}")

    for c in OBJECTIVES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)

    Yraw = df[OBJECTIVES].to_numpy(float)
    Ymax = Yraw * SIGNS
    Ynorm, _, _ = normalize_full(Ymax)

    true_pareto = pareto_mask_maximize(Ynorm)
    full_hv = hv3_origin(Ynorm[true_pareto])

    supported, unsupported = classify_supported_pareto(
        Ynorm, true_pareto, n_weights=args.supported_weights
    )

    print(f"Valid PI1070 rows: {len(df)}")
    print(f"True Pareto size: {true_pareto.sum()}")
    print(f"Approx. linearly supported Pareto points: {supported.sum()}")
    print(f"Approx. unsupported Pareto points: {unsupported.sum()}")
    print(f"Exact full-dataset normalized HV: {full_hv:.6f}")

    # Fixed, target-free, case-preserving representation used ONLY for
    # batch-diversity measurement/greedy diversification.
    div_vec = TfidfVectorizer(
        analyzer="char", ngram_range=(2,5), max_features=12000, lowercase=False
    )
    full_tfidf = div_vec.fit_transform(df["smiles"].astype(str))

    methods = [(f"linear_topk_n{n}", n) for n in SCALAR_COUNTS]
    methods += [("linear_diverse_n64", 64)]

    all_rows = []
    all_batch = []

    for seed in range(args.seeds):
        rng = np.random.default_rng(7000 + seed)
        init_idx = rng.choice(len(df), size=10, replace=False)

        print(f"\nSeed {seed+1}/{args.seeds}")
        for method, n_scalar in methods:
            print(f"  {method}")
            rows, batch = run_one(
                df, Yraw, Ynorm, full_tfidf,
                true_pareto, supported, unsupported, full_hv,
                init_idx, method, n_scalar, seed
            )
            all_rows.extend(rows)
            all_batch.extend(batch)

    traj = pd.DataFrame(all_rows)
    batches = pd.DataFrame(all_batch)
    traj.to_csv(out / "scalarization_batch_trajectories.csv", index=False)
    batches.to_csv(out / "scalarization_batch_diversity.csv", index=False)

    # Final N=57 summary
    final = traj[traj["budget"] == 57].copy()
    summary = (
        final.groupby(["method", "n_scalarizations"], as_index=False)
        .agg(
            pareto_recall_mean=("pareto_recall", "mean"),
            pareto_recall_sd=("pareto_recall", "std"),
            supported_recall_mean=("supported_pareto_recall", "mean"),
            supported_recall_sd=("supported_pareto_recall", "std"),
            unsupported_recall_mean=("unsupported_pareto_recall", "mean"),
            unsupported_recall_sd=("unsupported_pareto_recall", "std"),
            hv_recovery_mean=("hv_recovery", "mean"),
            hv_recovery_sd=("hv_recovery", "std"),
        )
    )

    div_summary = (
        batches.groupby(["method", "n_scalarizations"], as_index=False)
        .agg(
            mean_batch_diversity=("mean_pairwise_tfidf_cosine_distance", "mean"),
            sd_batch_diversity=("mean_pairwise_tfidf_cosine_distance", "std"),
        )
    )

    summary = summary.merge(div_summary, on=["method", "n_scalarizations"], how="left")
    summary.to_csv(out / "scalarization_batch_summary_N57.csv", index=False)

    print("\n===== N=57 SUMMARY =====")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs in: {out.resolve()}")


if __name__ == "__main__":
    main()
