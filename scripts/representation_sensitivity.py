#!/usr/bin/env python3
"""
Chronological representation-sensitivity analysis for the EloKriti campaign.

For each acquisition round, models are trained only on candidates completed
before that round and evaluated on the candidates acquired in that round.

Representations:
  - original lower-case concatenated TF-IDF/SVD8
  - case-preserving concatenated TF-IDF/SVD8
  - case-preserving SMILES + continuous descriptors (SVD 4/8/16/32)
  - descriptor-only
  - Morgan-512 fingerprint + descriptors
  - canonical case-preserving SMILES + descriptors

Usage:
  python scripts/representation_sensitivity.py
  python scripts/representation_sensitivity.py \
      --evaluated data/evaluated_clean_no_expansion.csv \
      --repeats 10
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

DESC_COLS = [
    "molecular_weight_Boltzmann_average",
    "logP_Boltzmann_average",
    "TPSA_Boltzmann_average",
    "normalized_monomer_phi_Boltzmann_average",
    "normalized_backbone_phi_Boltzmann_average",
]
OBJECTIVES = ["elastic_modulus", "volumetric_shrinkage", "ffv"]
SMILES_COL = "reactant_1"
ID_COL = "reaction_id"
BASE_SEED = 42


def parse_iteration(value) -> int:
    s = str(value).strip().lower()
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
    raise ValueError(f"Cannot parse iteration value: {value!r}")


def load_campaign(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = [ID_COL, "iteration", SMILES_COL, *DESC_COLS, *OBJECTIVES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if "status" in df.columns:
        status = df["status"].astype(str).str.lower().str.strip()
        df = df[status.isin(["completed", "complete", "valid", "done"])].copy()

    for c in [*DESC_COLS, *OBJECTIVES]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )

    df = df.dropna(subset=[SMILES_COL, *DESC_COLS, *OBJECTIVES]).copy()
    df["_round"] = df["iteration"].apply(parse_iteration)
    df = df.sort_values(["_round", ID_COL]).reset_index(drop=True)

    if df[ID_COL].astype(str).duplicated().any():
        raise ValueError("Duplicate reaction_id values found.")

    print("Completed-label counts by round:")
    print(df.groupby("_round").size().to_string())
    print(f"\nTotal completed rows: {len(df)}")
    print(f"Initial seed rows: {(df['_round'] == 0).sum()}")
    print(f"Pre-acquisition test rows (rounds > 0): {(df['_round'] > 0).sum()}")
    return df


def _safe_n_components(cap: int, sparse_shape, n_train: int) -> int:
    return max(1, min(cap, max(1, n_train - 2), max(1, min(sparse_shape) - 1)))


def _canonicalize(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles), False
    return Chem.MolToSmiles(mol, canonical=True), True


def _morgan_matrix(smiles_values, n_bits=512, radius=2):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles_values), n_bits), dtype=float)
    ok = np.zeros(len(smiles_values), dtype=bool)
    for i, smi in enumerate(smiles_values):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        arr = np.zeros(n_bits, dtype=float)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
        ok[i] = True
    return X, ok


@dataclass
class FeatureBuilder:
    name: str

    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame):
        tr_smiles = train[SMILES_COL].astype(str).tolist()
        te_smiles = test[SMILES_COL].astype(str).tolist()
        Xd_tr = train[DESC_COLS].to_numpy(dtype=float)
        Xd_te = test[DESC_COLS].to_numpy(dtype=float)

        if self.name == "original_lowercase_concat_svd8":
            def txt(df):
                out = []
                for _, r in df.iterrows():
                    vals = [str(r[SMILES_COL])] + [str(float(r[c])) for c in DESC_COLS]
                    out.append(" ".join(vals))
                return out
            v = TfidfVectorizer(analyzer="char", ngram_range=(2,5), max_features=12000)
            A_tr = v.fit_transform(txt(train)); A_te = v.transform(txt(test))
            nc = _safe_n_components(8, A_tr.shape, len(train))
            svd = TruncatedSVD(n_components=nc, random_state=BASE_SEED)
            Z_tr = svd.fit_transform(A_tr); Z_te = svd.transform(A_te)
            sc = StandardScaler()
            return sc.fit_transform(Z_tr), sc.transform(Z_te)

        if self.name == "casepres_concat_svd8":
            def txt(df):
                out = []
                for _, r in df.iterrows():
                    vals = [str(r[SMILES_COL])] + [str(float(r[c])) for c in DESC_COLS]
                    out.append(" ".join(vals))
                return out
            v = TfidfVectorizer(analyzer="char", ngram_range=(2,5), max_features=12000, lowercase=False)
            A_tr = v.fit_transform(txt(train)); A_te = v.transform(txt(test))
            nc = _safe_n_components(8, A_tr.shape, len(train))
            svd = TruncatedSVD(n_components=nc, random_state=BASE_SEED)
            Z_tr = svd.fit_transform(A_tr); Z_te = svd.transform(A_te)
            sc = StandardScaler()
            return sc.fit_transform(Z_tr), sc.transform(Z_te)

        if self.name.startswith("separated_casepres_svd"):
            cap = int(self.name.split("svd")[-1])
            v = TfidfVectorizer(analyzer="char", ngram_range=(2,5), max_features=12000, lowercase=False)
            A_tr = v.fit_transform(tr_smiles); A_te = v.transform(te_smiles)
            nc = _safe_n_components(cap, A_tr.shape, len(train))
            svd = TruncatedSVD(n_components=nc, random_state=BASE_SEED)
            Z_tr = svd.fit_transform(A_tr); Z_te = svd.transform(A_te)
            scz = StandardScaler()
            Z_tr = scz.fit_transform(Z_tr); Z_te = scz.transform(Z_te)
            scd = StandardScaler()
            D_tr = scd.fit_transform(Xd_tr); D_te = scd.transform(Xd_te)
            return np.hstack([Z_tr, D_tr]), np.hstack([Z_te, D_te])

        if self.name == "descriptor_only":
            sc = StandardScaler()
            return sc.fit_transform(Xd_tr), sc.transform(Xd_te)

        if self.name == "morgan512_plus_desc":
            M_tr, _ = _morgan_matrix(tr_smiles, 512, 2)
            M_te, _ = _morgan_matrix(te_smiles, 512, 2)
            scd = StandardScaler()
            D_tr = scd.fit_transform(Xd_tr); D_te = scd.transform(Xd_te)
            return np.hstack([M_tr, D_tr]), np.hstack([M_te, D_te])

        if self.name == "canonical_casepres_svd8":
            tr_can = [_canonicalize(s)[0] for s in tr_smiles]
            te_can = [_canonicalize(s)[0] for s in te_smiles]
            v = TfidfVectorizer(analyzer="char", ngram_range=(2,5), max_features=12000, lowercase=False)
            A_tr = v.fit_transform(tr_can); A_te = v.transform(te_can)
            nc = _safe_n_components(8, A_tr.shape, len(train))
            svd = TruncatedSVD(n_components=nc, random_state=BASE_SEED)
            Z_tr = svd.fit_transform(A_tr); Z_te = svd.transform(A_te)
            scz = StandardScaler()
            Z_tr = scz.fit_transform(Z_tr); Z_te = scz.transform(Z_te)
            scd = StandardScaler()
            D_tr = scd.fit_transform(Xd_tr); D_te = scd.transform(Xd_te)
            return np.hstack([Z_tr, D_tr]), np.hstack([Z_te, D_te])

        raise ValueError(self.name)


REPRESENTATIONS = [
    "original_lowercase_concat_svd8",
    "casepres_concat_svd8",
    "separated_casepres_svd4",
    "separated_casepres_svd8",
    "separated_casepres_svd16",
    "separated_casepres_svd32",
    "descriptor_only",
    "morgan512_plus_desc",
    "canonical_casepres_svd8",
]


def make_gp(seed: int):
    kernel = ConstantKernel(1.0, (1e-3,1e3))*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8,1e1))
    return GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, normalize_y=True,
        n_restarts_optimizer=1, random_state=seed
    )


def calc_metrics(y_true, y_pred):
    rho = spearmanr(y_true, y_pred).statistic
    return dict(
        n=len(y_true),
        mae=mean_absolute_error(y_true, y_pred),
        rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
        r2=r2_score(y_true, y_pred),
        spearman_rho=rho,
    )


def chronological_predictions(df, rep_name, repeat):
    pred = {o: [] for o in OBJECTIVES}
    true = {o: [] for o in OBJECTIVES}

    for r in sorted(x for x in df["_round"].unique() if x > 0):
        train = df[df["_round"] < r].copy()
        test = df[df["_round"] == r].copy()
        if test.empty:
            continue

        Xtr, Xte = FeatureBuilder(rep_name).fit_transform(train, test)

        for j, obj in enumerate(OBJECTIVES):
            gp = make_gp(BASE_SEED + 1000*repeat + 17*j + r)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                gp.fit(Xtr, train[obj].to_numpy(float))
            mu = gp.predict(Xte)
            pred[obj].extend(mu.tolist())
            true[obj].extend(test[obj].to_numpy(float).tolist())

    rows = []
    for obj in OBJECTIVES:
        rows.append({
            "representation": rep_name,
            "repeat": repeat,
            "objective": obj,
            **calc_metrics(np.asarray(true[obj]), np.asarray(pred[obj]))
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluated", default="data/evaluated_clean_no_expansion.csv")
    ap.add_argument("--output-dir", "--out-dir", dest="output_dir", default="outputs/representation_sensitivity")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_campaign(args.evaluated)

    rows = []
    for rep in REPRESENTATIONS:
        print(f"\n=== {rep} ===")
        for repeat in range(args.repeats):
            rr = chronological_predictions(df, rep, repeat)
            rows.extend(rr)
            print(
                f"repeat {repeat+1:02d}: " +
                " | ".join(f"{x['objective']} rho={x['spearman_rho']:.3f}" for x in rr)
            )

    all_df = pd.DataFrame(rows)
    all_df.to_csv(out/"representation_sensitivity_all_repeats.csv", index=False)

    summary = (
        all_df.groupby(["representation","objective"], as_index=False)
        .agg(
            n=("n","first"),
            mae_mean=("mae","mean"), mae_sd=("mae","std"),
            rmse_mean=("rmse","mean"), rmse_sd=("rmse","std"),
            r2_mean=("r2","mean"), r2_sd=("r2","std"),
            spearman_mean=("spearman_rho","mean"),
            spearman_sd=("spearman_rho","std"),
        )
    )
    summary.to_csv(out/"representation_sensitivity_summary.csv", index=False)

    piv = summary.pivot(index="representation", columns="objective", values="spearman_mean").reset_index()
    piv.to_csv(out/"representation_sensitivity_spearman_table.csv", index=False)

    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))
    print("\n===== SPEARMAN TABLE =====")
    print(piv.to_string(index=False))
    print(f"\nSaved in: {out.resolve()}")


if __name__ == "__main__":
    main()
