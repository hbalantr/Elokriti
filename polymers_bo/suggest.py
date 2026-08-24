"""Suggest next MD candidates using GP surrogates and ParEGO-inspired
random linear-scalarization expected improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from polymers_bo.data import load_candidate_and_evaluated, split_unobserved_candidates
from polymers_bo.surrogate import FeatureConfig, MultiTargetGPSurrogate
from polymers_bo.bo import parego_acquisition_scores


@dataclass
class SuggestConfig:
    candidate_csv: str
    evaluated_csv: str
    output_csv: str = "next_candidates.csv"

    id_col: str = "reaction_id"
    smiles_col: str = "reactant_1"

    input_columns: Sequence[str] = (
        "reactant_1",
        "molecular_weight_Boltzmann_average",
        "logP_Boltzmann_average",
        "TPSA_Boltzmann_average",
        "normalized_monomer_phi_Boltzmann_average",
        "normalized_backbone_phi_Boltzmann_average",
    )

    objective_names: Sequence[str] = (
        "elastic_modulus",
        "volumetric_shrinkage",
        "ffv",
    )

    objective_directions: Sequence[str] = (
        "max",
        "min",
        "min",
    )

    top_k: int = 10
    random_seed: int = 42
    n_scalarizations: int = 64
    svd_components: int = 8
    gp_restarts: int = 1
    xi: float = 0.01


def _direction_to_sign(direction: str) -> float:
    direction = direction.lower().strip()
    if direction == "max":
        return 1.0
    if direction == "min":
        return -1.0
    raise ValueError(f"Invalid objective direction: {direction}")


def suggest_next_candidates(config: SuggestConfig) -> pd.DataFrame:
    """Fit the surrogate models and rank the next MD candidates."""

    candidate_df, evaluated_df = load_candidate_and_evaluated(
        candidate_csv=config.candidate_csv,
        evaluated_csv=config.evaluated_csv,
        id_col=config.id_col,
        input_columns=config.input_columns,
        objective_columns=config.objective_names,
        smiles_col=config.smiles_col,
    )

    unobserved_df = split_unobserved_candidates(
        candidate_df=candidate_df,
        evaluated_df=evaluated_df,
        id_col=config.id_col,
    )

    if evaluated_df.shape[0] < 8:
        raise ValueError(f"Need at least 8 evaluated rows; found {evaluated_df.shape[0]}.")

    if unobserved_df.empty:
        raise ValueError("No unevaluated candidates remain.")

    signs = np.array(
        [_direction_to_sign(d) for d in config.objective_directions],
        dtype=float,
    )

    y_raw = evaluated_df[list(config.objective_names)].to_numpy(dtype=float)
    y_train = y_raw * signs

    feature_config = FeatureConfig(
        max_tfidf_features=12000,
        ngram_range=(2, 5),
        svd_components=min(
            config.svd_components,
            max(2, evaluated_df.shape[0] - 2),
        ),
        random_seed=config.random_seed,
        use_rdkit_features=False,
    )

    model = MultiTargetGPSurrogate(
        feature_config=feature_config,
        gp_restarts=config.gp_restarts,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(
            texts=evaluated_df["representation"].astype(str).to_numpy(),
            targets=y_train,
            target_names=list(config.objective_names),
        )

    pred_mean_max, pred_std = model.predict(
        texts=unobserved_df["representation"].astype(str).to_numpy(),
    )

    acquisition = parego_acquisition_scores(
        observed_targets=y_train,
        predicted_mean=pred_mean_max,
        predicted_std=pred_std,
        objective_directions=["max"] * len(config.objective_names),
        random_seed=config.random_seed,
        n_scalarizations=config.n_scalarizations,
        xi=config.xi,
    )

    suggested = unobserved_df.copy()

    for i, name in enumerate(config.objective_names):
        suggested[f"pred_{name}"] = pred_mean_max[:, i] * signs[i]
        suggested[f"unc_{name}"] = pred_std[:, i]

    suggested["acquisition_score"] = acquisition
    suggested = suggested.sort_values(
        "acquisition_score",
        ascending=False,
    ).reset_index(drop=True)

    top = suggested.head(config.top_k).copy()
    top.to_csv(config.output_csv, index=False)

    print(f"Evaluated rows used: {evaluated_df.shape[0]}")
    print(f"Unevaluated candidates: {unobserved_df.shape[0]}")
    print(f"Saved suggestions: {config.output_csv}")

    return top