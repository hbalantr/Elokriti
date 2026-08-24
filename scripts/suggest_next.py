"""Command-line wrapper for suggesting next AL/BO candidates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from polymers_bo.suggest import SuggestConfig, suggest_next_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", default="candidate_pool.csv")
    parser.add_argument("--evaluated-csv", default="evaluated_clean.csv")
    parser.add_argument("--output-csv", default="next_candidates.csv")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scalarizations", type=int, default=64)

    args = parser.parse_args()

    config = SuggestConfig(
        candidate_csv=args.candidate_csv,
        evaluated_csv=args.evaluated_csv,
        output_csv=args.output_csv,
        top_k=args.top_k,
        random_seed=args.seed,
        n_scalarizations=args.scalarizations,
    )

    top = suggest_next_candidates(config)

    print(
        top[
            [
                "reaction_id",
                "reactant_1",
                "pred_elastic_modulus",
                "pred_volumetric_shrinkage",
                "pred_ffv",
                "acquisition_score",
            ]
        ]
    )


if __name__ == "__main__":
    main()
