"""Data loading and cleaning utilities for backbone AL/BO pipeline."""

from __future__ import annotations

from typing import Sequence, Tuple
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV and strip column whitespace."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Convert values like '12.01%' or strings to numeric floats."""
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def build_representation(df: pd.DataFrame, input_columns: Sequence[str]) -> pd.Series:
    """Build model-ready text representation from selected input columns."""
    missing = [c for c in input_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing input columns: {missing}")

    return df[list(input_columns)].astype(str).agg(" ".join, axis=1)


def validate_columns(
    candidate_df: pd.DataFrame,
    evaluated_df: pd.DataFrame,
    id_col: str,
    input_columns: Sequence[str],
    objective_columns: Sequence[str],
) -> None:
    """Check required columns exist."""
    candidate_required = [id_col, *input_columns]
    evaluated_required = [id_col, *input_columns, *objective_columns]

    missing_candidate = [c for c in candidate_required if c not in candidate_df.columns]
    missing_evaluated = [c for c in evaluated_required if c not in evaluated_df.columns]

    if missing_candidate:
        raise KeyError(f"Candidate CSV missing columns: {missing_candidate}")

    if missing_evaluated:
        raise KeyError(f"Evaluated CSV missing columns: {missing_evaluated}")


def load_candidate_and_evaluated(
    candidate_csv: str,
    evaluated_csv: str,
    id_col: str,
    input_columns: Sequence[str],
    objective_columns: Sequence[str],
    smiles_col: str = "reactant_1",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, clean, validate candidate/evaluated tables.

    Returns
    -------
    candidate_df
        Candidate pool with representation column.
    evaluated_df
        Completed evaluated rows with representation and numeric objectives.
    """

    candidate_df = load_csv(candidate_csv)
    evaluated_df = load_csv(evaluated_csv)

    validate_columns(
        candidate_df=candidate_df,
        evaluated_df=evaluated_df,
        id_col=id_col,
        input_columns=input_columns,
        objective_columns=objective_columns,
    )

    for col in objective_columns:
        evaluated_df[col] = clean_numeric_column(evaluated_df[col])

    evaluated_df = evaluated_df.dropna(
        subset=[id_col, smiles_col, *objective_columns]
    ).copy()

    evaluated_df = evaluated_df.drop_duplicates(subset=[id_col], keep="last")

    candidate_df["representation"] = build_representation(candidate_df, input_columns)
    evaluated_df["representation"] = build_representation(evaluated_df, input_columns)

    return candidate_df.reset_index(drop=True), evaluated_df.reset_index(drop=True)


def split_unobserved_candidates(
    candidate_df: pd.DataFrame,
    evaluated_df: pd.DataFrame,
    id_col: str,
) -> pd.DataFrame:
    """Remove already evaluated IDs from candidate pool."""
    evaluated_ids = set(evaluated_df[id_col].astype(str))

    unobserved_df = candidate_df[
        ~candidate_df[id_col].astype(str).isin(evaluated_ids)
    ].copy()

    return unobserved_df.reset_index(drop=True)
