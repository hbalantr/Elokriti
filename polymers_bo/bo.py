"""Multi-objective Bayesian optimization utilities for EloKriti.

Candidate acquisition uses ParEGO-inspired random linear scalarization
with expected improvement averaged over multiple Dirichlet-sampled
weight vectors.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


def _direction_to_sign(direction: str) -> int:
    if direction == "max":
        return 1
    if direction == "min":
        return -1
    raise ValueError(f"Invalid objective direction: {direction}")


def objective_signs(objective_directions: Sequence[str]) -> np.ndarray:
    """Map objective directions to +/-1 signs (maximize space)."""

    return np.array(
        [_direction_to_sign(direction) for direction in objective_directions],
        dtype=float,
    )


def normalize_maximize_objectives(
    maximize_objectives: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Normalize maximize-space objectives to [0, 1] per dimension."""

    spans = np.maximum(upper_bounds - lower_bounds, 1e-12)
    normalized = (maximize_objectives - lower_bounds) / spans
    return np.clip(normalized, 0.0, 1.0)


def estimate_hypervolume_fraction(
    front_normalized: np.ndarray,
    mc_samples: np.ndarray,
) -> float:
    """Estimate dominated hypervolume fraction in [0,1]^M using Monte Carlo samples."""

    if front_normalized.size == 0:
        return 0.0

    dominated_mask = np.any(
        np.all(
            front_normalized[:, None, :] >= mc_samples[None, :, :],
            axis=2,
        ),
        axis=0,
    )

    return float(np.mean(dominated_mask))


def expected_improvement(
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    best_observed: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected improvement acquisition for a scalar objective."""

    safe_std = np.maximum(predicted_std, 1e-12)
    improvement = predicted_mean - best_observed - xi
    z_score = improvement / safe_std

    ei = improvement * norm.cdf(z_score) + safe_std * norm.pdf(z_score)

    deterministic_mask = predicted_std < 1e-12
    ei[deterministic_mask] = np.maximum(
        0.0,
        improvement[deterministic_mask],
    )

    return ei


def parego_acquisition_scores(
    observed_targets: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    objective_directions: Sequence[str],
    random_seed: int,
    n_scalarizations: int = 64,
    xi: float = 0.01,
) -> np.ndarray:
    """Average EI across random scalarizations for multi-objective ranking."""

    if observed_targets.shape[1] != predicted_mean.shape[1]:
        raise ValueError(
            "Observed and predicted targets must have the same number of objectives."
        )

    if predicted_mean.shape != predicted_std.shape:
        raise ValueError("predicted_mean and predicted_std shapes must match.")

    signs = objective_signs(objective_directions)
    observed_maximize = observed_targets * signs
    predicted_maximize_mean = predicted_mean * signs

    center = observed_maximize.mean(axis=0)
    scale = observed_maximize.std(axis=0)
    scale[scale < 1e-8] = 1.0

    rng = np.random.default_rng(random_seed)
    scores = np.zeros(predicted_mean.shape[0], dtype=float)

    for _ in range(max(1, n_scalarizations)):
        weights = rng.dirichlet(np.ones(observed_targets.shape[1]))

        observed_scalarized = ((observed_maximize - center) / scale) @ weights
        best_observed = float(np.max(observed_scalarized))

        pred_scalarized_mean = ((predicted_maximize_mean - center) / scale) @ weights
        pred_scalarized_std = np.sqrt(
            np.sum((weights * (predicted_std / scale)) ** 2, axis=1)
        )

        scores += expected_improvement(
            predicted_mean=pred_scalarized_mean,
            predicted_std=pred_scalarized_std,
            best_observed=best_observed,
            xi=xi,
        )

    return scores / max(1, n_scalarizations)


def pareto_mask_maximize(objectives: np.ndarray) -> np.ndarray:
    """Return True for non-dominated points in maximize-all objective space."""

    n_points = objectives.shape[0]
    is_non_dominated = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not is_non_dominated[i]:
            continue

        dominates = np.all(objectives[i] >= objectives, axis=1) & np.any(
            objectives[i] > objectives,
            axis=1,
        )

        is_non_dominated[dominates] = False
        is_non_dominated[i] = True

    return is_non_dominated


def build_pareto_table(
    observed_table: pd.DataFrame,
    observed_targets: np.ndarray,
    objective_names: Sequence[str],
    objective_directions: Sequence[str],
) -> pd.DataFrame:
    """Return non-dominated observed candidates with objective values."""

    signs = objective_signs(objective_directions)
    signed = observed_targets * signs
    mask = pareto_mask_maximize(signed)

    pareto = observed_table.loc[mask].copy()

    for idx, name in enumerate(objective_names):
        pareto[name] = observed_targets[mask, idx]

    return pareto.reset_index(drop=True)