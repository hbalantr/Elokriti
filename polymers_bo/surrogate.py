"""Surrogate model built from text featurization + per-target Gaussian Processes."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    max_tfidf_features: int = 12000
    ngram_range: Tuple[int, int] = (2, 5)
    svd_components: int = 8
    random_seed: int = 42
    use_rdkit_features: bool = False
    rdkit_fingerprint_bits: int = 512
    rdkit_fingerprint_radius: int = 2
    rdkit_descriptor_names: List[str] = field(default_factory=list)


class MultiTargetGPSurrogate:
    """Independent GP per target over shared text-derived features."""

    def __init__(
        self,
        feature_config: FeatureConfig,
        gp_restarts: int = 1,
    ) -> None:
        self.feature_config = feature_config
        self.gp_restarts = gp_restarts
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=feature_config.ngram_range,
            max_features=feature_config.max_tfidf_features,
        )
        self.svd: Optional[TruncatedSVD] = None
        self.scaler: Optional[StandardScaler] = None
        self.models: Dict[str, GaussianProcessRegressor] = {}
        self.target_names: List[str] = []
        self._rdkit_descriptor_order: List[str] = list(feature_config.rdkit_descriptor_names)

    @staticmethod
    def _normalize_polymer_smiles(smiles: str) -> str:
        """Normalize polymer strings for more stable RDKit parsing."""

        normalized = str(smiles).strip()
        if not normalized or normalized.lower() in {"nan", "none"}:
            return ""
        # PI1070-style repeat units often use bare '*' as terminal wildcard.
        # Convert to bracket wildcard form first, then fallback below if needed.
        return re.sub(r"(?<!\[)\*(?!\])", "[*]", normalized)

    @classmethod
    def _smiles_parse_candidates(cls, smiles: str) -> List[str]:
        """Generate parse fallbacks for polymer repeat-unit strings."""

        normalized = cls._normalize_polymer_smiles(smiles)
        if not normalized:
            return []

        candidates: List[str] = []
        for candidate in (
            normalized,
            normalized.replace("[*]", "C").replace("*", "C"),
            normalized.replace("[*]", "").replace("*", ""),
        ):
            compact = candidate.strip()
            if compact and compact not in candidates:
                candidates.append(compact)
        return candidates

    @classmethod
    def _safe_mol_from_smiles(cls, smiles: str, chem_module: object) -> Optional[object]:
        """Parse polymer-like SMILES robustly with simple fallbacks."""

        for candidate in cls._smiles_parse_candidates(smiles):
            mol = chem_module.MolFromSmiles(candidate)
            if mol is not None:
                return mol
        return None

    def _rdkit_features(self, smiles_values: Sequence[str]) -> np.ndarray:
        """Build optional RDKit descriptor + fingerprint feature matrix."""

        if not self.feature_config.use_rdkit_features:
            return np.empty((len(smiles_values), 0), dtype=float)

        try:
            from rdkit import Chem, DataStructs, RDLogger  # type: ignore
            from rdkit.Chem import Descriptors, rdFingerprintGenerator  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "RDKit features requested but rdkit is not installed. "
                "Install with: .venv/bin/pip install rdkit"
            ) from exc
        RDLogger.DisableLog("rdApp.*")

        descriptor_functions = []
        for name in self._rdkit_descriptor_order:
            if not hasattr(Descriptors, name):
                raise ValueError(f"Unknown RDKit descriptor name: {name}")
            descriptor_functions.append(getattr(Descriptors, name))

        n_bits = self.feature_config.rdkit_fingerprint_bits
        radius = self.feature_config.rdkit_fingerprint_radius
        n_desc = len(descriptor_functions)
        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

        rows: List[np.ndarray] = []
        for smiles in smiles_values:
            mol = self._safe_mol_from_smiles(str(smiles), Chem)
            fingerprint_bits = np.zeros(n_bits, dtype=float)
            descriptor_values = np.zeros(n_desc, dtype=float)
            if mol is not None:
                fp = morgan_generator.GetFingerprint(mol)
                DataStructs.ConvertToNumpyArray(fp, fingerprint_bits)
                descriptor_values = np.array(
                    [fn(mol) for fn in descriptor_functions],
                    dtype=float,
                )
                descriptor_values = np.nan_to_num(descriptor_values, nan=0.0, posinf=0.0, neginf=0.0)
            rows.append(np.concatenate([descriptor_values, fingerprint_bits], axis=0))
        return np.vstack(rows)

    def _fit_features(
        self,
        texts: Iterable[str],
        rdkit_smiles: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        text_list = list(texts)
        sparse_features = self.vectorizer.fit_transform(text_list)
        min_dim = min(sparse_features.shape)
        if min_dim <= 3:
            dense_features = sparse_features.toarray()
            self.svd = None
        else:
            n_components = min(self.feature_config.svd_components, min_dim - 1)
            self.svd = TruncatedSVD(
                n_components=n_components,
                random_state=self.feature_config.random_seed,
            )
            dense_features = self.svd.fit_transform(sparse_features)

        if self.feature_config.use_rdkit_features:
            smiles_values = list(rdkit_smiles) if rdkit_smiles is not None else text_list
            rdkit_matrix = self._rdkit_features(smiles_values)
            dense_features = np.hstack([dense_features, rdkit_matrix])

        self.scaler = StandardScaler()
        return self.scaler.fit_transform(dense_features)

    def _transform_features(
        self,
        texts: Iterable[str],
        rdkit_smiles: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        text_list = list(texts)
        sparse_features = self.vectorizer.transform(text_list)
        if self.svd is None:
            dense_features = sparse_features.toarray()
        else:
            dense_features = self.svd.transform(sparse_features)

        if self.feature_config.use_rdkit_features:
            smiles_values = list(rdkit_smiles) if rdkit_smiles is not None else text_list
            rdkit_matrix = self._rdkit_features(smiles_values)
            dense_features = np.hstack([dense_features, rdkit_matrix])

        if self.scaler is None:
            raise RuntimeError("Scaler is not fitted.")
        return self.scaler.transform(dense_features)

    def fit(
        self,
        texts: Sequence[str],
        targets: np.ndarray,
        target_names: Sequence[str],
        rdkit_smiles: Optional[Sequence[str]] = None,
    ) -> "MultiTargetGPSurrogate":
        if len(texts) != targets.shape[0]:
            raise ValueError("Number of texts and target rows must match.")
        if rdkit_smiles is not None and len(rdkit_smiles) != len(texts):
            raise ValueError("Number of rdkit_smiles entries must match number of texts.")

        features = self._fit_features(texts, rdkit_smiles=rdkit_smiles)
        self.target_names = list(target_names)
        self.models = {}

        for target_index, target_name in enumerate(self.target_names):
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=1.0, nu=2.5)
                + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
            )
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=self.gp_restarts,
                random_state=self.feature_config.random_seed,
            )
            model.fit(features, targets[:, target_index])
            self.models[target_name] = model
        return self

    def predict(
        self,
        texts: Sequence[str],
        rdkit_smiles: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("Model is not fitted.")
        if rdkit_smiles is not None and len(rdkit_smiles) != len(texts):
            raise ValueError("Number of rdkit_smiles entries must match number of texts.")

        features = self._transform_features(texts, rdkit_smiles=rdkit_smiles)
        means: List[np.ndarray] = []
        stds: List[np.ndarray] = []
        for target_name in self.target_names:
            prediction_mean, prediction_std = self.models[target_name].predict(
                features, return_std=True
            )
            means.append(prediction_mean)
            stds.append(prediction_std)
        return np.column_stack(means), np.column_stack(stds)