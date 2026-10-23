#!/usr/bin/env python3
"""
Chronological structure-similarity audit for the EloKriti active-learning campaign.

For every candidate acquired after the initial seed, this script compares its
structure only with candidates whose labels were available before that round.

Checks:
  - exact canonical-SMILES overlap
  - maximum Morgan-fingerprint Tanimoto similarity to the preceding training set
  - shared non-empty Bemis-Murcko scaffold with the preceding training set

Outputs:
  structure_similarity_audit.csv
  structure_similarity_summary.csv

Usage:
  python scripts/structure_similarity_audit.py \
      --evaluated data/evaluated_clean_no_expansion.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ID_COL = "reaction_id"
SMILES_COL = "reactant_1"


def parse_iteration(value) -> int:
    s = str(value).strip().lower().replace("_", "-").replace(" ", "")
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
    raise ValueError(f"Cannot parse iteration value: {value!r}")


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluated", default="data/evaluated_clean_no_expansion.csv")
    ap.add_argument("--output-dir", default="outputs/structure_similarity_audit")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--fp-bits", type=int, default=2048)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.evaluated)
    df.columns = df.columns.str.strip()

    required = [ID_COL, "iteration", SMILES_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if "status" in df.columns:
        status = df["status"].astype(str).str.lower().str.strip()
        keep = status.isin(["completed", "complete", "valid", "done"])
        # If status labels are absent/unrecognized, retain rows rather than silently dropping all data.
        if keep.any():
            df = df[keep].copy()

    df["_round"] = df["iteration"].apply(parse_iteration)
    df["_canonical_smiles"] = df[SMILES_COL].map(canonical_smiles)
    df["_scaffold"] = df["_canonical_smiles"].map(murcko_scaffold)
    df = df.sort_values(["_round", ID_COL]).reset_index(drop=True)

    if df[ID_COL].astype(str).duplicated().any():
        raise ValueError("Duplicate reaction_id values found.")

    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.radius,
        fpSize=args.fp_bits,
    )
    df["_fp"] = [
        fpgen.GetFingerprint(Chem.MolFromSmiles(s))
        for s in df["_canonical_smiles"]
    ]

    rows = []

    for round_id in sorted(r for r in df["_round"].unique() if r > 0):
        train = df[df["_round"] < round_id]
        test = df[df["_round"] == round_id]

        if train.empty or test.empty:
            continue

        train_smiles = train["_canonical_smiles"].tolist()
        train_smiles_set = set(train_smiles)
        train_scaffolds = set(s for s in train["_scaffold"] if s)
        train_fps = train["_fp"].tolist()

        for _, row in test.iterrows():
            sims = np.asarray(
                DataStructs.BulkTanimotoSimilarity(row["_fp"], train_fps),
                dtype=float,
            )
            best_pos = int(np.argmax(sims))
            best = train.iloc[best_pos]

            rows.append({
                "reaction_id": row[ID_COL],
                "iteration": row["iteration"],
                "n_prior_labeled": len(train),
                "canonical_smiles": row["_canonical_smiles"],
                "exact_canonical_smiles_in_prior": row["_canonical_smiles"] in train_smiles_set,
                "max_morgan_tanimoto_to_prior": float(sims[best_pos]),
                "nearest_prior_reaction_id": best[ID_COL],
                "nearest_prior_canonical_smiles": best["_canonical_smiles"],
                "murcko_scaffold": row["_scaffold"],
                "shared_nonempty_murcko_scaffold_with_prior": (
                    bool(row["_scaffold"]) and row["_scaffold"] in train_scaffolds
                ),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No post-initial candidates were available for the audit.")

    result.to_csv(out / "structure_similarity_audit.csv", index=False)

    sims = result["max_morgan_tanimoto_to_prior"].to_numpy(float)
    summary = pd.DataFrame([{
        "n_post_initial_candidates": len(result),
        "n_exact_canonical_smiles_overlap": int(result["exact_canonical_smiles_in_prior"].sum()),
        "n_shared_nonempty_murcko_scaffold": int(result["shared_nonempty_murcko_scaffold_with_prior"].sum()),
        "max_similarity_mean": float(np.mean(sims)),
        "max_similarity_median": float(np.median(sims)),
        "max_similarity_min": float(np.min(sims)),
        "max_similarity_max": float(np.max(sims)),
        "n_similarity_ge_0.50": int(np.sum(sims >= 0.50)),
        "fraction_similarity_ge_0.50": float(np.mean(sims >= 0.50)),
    }])
    summary.to_csv(out / "structure_similarity_summary.csv", index=False)

    print("\n===== STRUCTURE-SIMILARITY AUDIT =====")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs in: {out.resolve()}")


if __name__ == "__main__":
    main()
