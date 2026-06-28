from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .inorganic_fingerprints import (
    DEFAULT_COMPOSITION_STATISTICS,
    DEFAULT_ELEMENTAL_PROPERTIES,
    featurize_compositions,
)
from .organic_fingerprints import DEFAULT_RDKIT_DESCRIPTORS, featurize_smiles
from .structure_fingerprints import (
    DEFAULT_STRUCTURE_FEATURES,
    featurize_structures,
    structure_feature_names,
)

__all__ = [
    "CompositionFeaturizer",
    "PolymerSmilesFeaturizer",
    "SmilesFeaturizer",
    "StructureFeaturizer",
]


class CompositionFeaturizer(TransformerMixin, BaseEstimator):
    """Scikit-learn-style transformer for inorganic composition descriptors.

    The transformer wraps :func:`matgpr.featurize_compositions`, preserving the
    same pymatgen-backed elemental-property descriptors while adding familiar
    ``fit``, ``transform``, ``fit_transform``, and ``get_feature_names_out``
    methods.
    """

    def __init__(
        self,
        *,
        formula_column: str | int | None = None,
        properties: Sequence[str] = DEFAULT_ELEMENTAL_PROPERTIES,
        statistics: Sequence[str] = DEFAULT_COMPOSITION_STATISTICS,
        column_prefix: str | None = None,
        errors: str = "raise",
        cache_dir: str | Path | None = None,
        return_dataframe: bool = True,
    ):
        self.formula_column = formula_column
        self.properties = properties
        self.statistics = statistics
        self.column_prefix = column_prefix
        self.errors = errors
        self.cache_dir = cache_dir
        self.return_dataframe = return_dataframe

    def fit(self, X, y=None):
        """Store input-column metadata and deterministic descriptor names."""
        _validate_errors(self.errors)
        self.formula_column_ = _resolve_column(X, self.formula_column, kind="formula")
        feature_names = _composition_feature_names(
            properties=self.properties,
            statistics=self.statistics,
            column_prefix=self.column_prefix,
        )
        self.feature_names_out_ = np.asarray(feature_names, dtype=object)
        self.n_features_out_ = len(self.feature_names_out_)
        return self

    def transform(self, X):
        """Transform formulas into numeric composition descriptors."""
        check_is_fitted(self, "feature_names_out_")
        formulas, index = _extract_values(X, self.formula_column_, kind="formula")
        result = featurize_compositions(
            formulas,
            properties=self.properties,
            statistics=self.statistics,
            errors=self.errors,
            cache_dir=self.cache_dir,
        )
        features = result.features.copy()
        features.columns = self.feature_names_out_
        features.index = index
        self.failed_ = result.failed
        self.cache_keys_ = result.cache_keys.set_axis(index)
        self.cache_hit_ = result.cache_hit.set_axis(index)
        self.last_result_ = result
        return _format_transform_output(features, self.return_dataframe)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output descriptor names."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class StructureFeaturizer(TransformerMixin, BaseEstimator):
    """Scikit-learn-style transformer for global crystal-structure descriptors."""

    def __init__(
        self,
        *,
        structure_column: str | int | None = None,
        features: Sequence[str] = DEFAULT_STRUCTURE_FEATURES,
        column_prefix: str | None = None,
        errors: str = "raise",
        cache_dir: str | Path | None = None,
        return_dataframe: bool = True,
    ):
        self.structure_column = structure_column
        self.features = features
        self.column_prefix = column_prefix
        self.errors = errors
        self.cache_dir = cache_dir
        self.return_dataframe = return_dataframe

    def fit(self, X, y=None):
        """Store input-column metadata and deterministic descriptor names."""
        _validate_errors(self.errors)
        self.structure_column_ = _resolve_column(X, self.structure_column, kind="structure")
        self.feature_names_out_ = np.asarray(
            structure_feature_names(self.features, column_prefix=self.column_prefix),
            dtype=object,
        )
        self.n_features_out_ = len(self.feature_names_out_)
        return self

    def transform(self, X):
        """Transform structures into numeric lattice and packing descriptors."""
        check_is_fitted(self, "feature_names_out_")
        structures, index = _extract_values(X, self.structure_column_, kind="structure")
        result = featurize_structures(
            structures,
            features=self.features,
            errors=self.errors,
            cache_dir=self.cache_dir,
        )
        features = result.features.copy()
        features.columns = self.feature_names_out_
        features.index = index
        self.failed_ = result.failed
        self.cache_keys_ = result.cache_keys.set_axis(index)
        self.cache_hit_ = result.cache_hit.set_axis(index)
        self.last_result_ = result
        return _format_transform_output(features, self.return_dataframe)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output descriptor names."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class SmilesFeaturizer(TransformerMixin, BaseEstimator):
    """Scikit-learn-style transformer for molecule or polymer SMILES.

    Molecule SMILES are canonicalized directly with RDKit. If
    ``smiles_type="polymer"``, repeat-unit SMILES must contain exactly two
    ``[*]`` dummy atoms and are converted to the cyclic-trimer surrogate used by
    :func:`matgpr.featurize_smiles`.
    """

    def __init__(
        self,
        *,
        smiles_column: str | int | None = None,
        smiles_type: str = "molecule",
        fingerprint_type: str = "morgan",
        n_bits: int = 1024,
        radius: int = 2,
        descriptors: Sequence[str] = DEFAULT_RDKIT_DESCRIPTORS,
        column_prefix: str | None = None,
        errors: str = "raise",
        cache_dir: str | Path | None = None,
        include_canonical_smiles: bool = False,
        return_dataframe: bool = True,
    ):
        self.smiles_column = smiles_column
        self.smiles_type = smiles_type
        self.fingerprint_type = fingerprint_type
        self.n_bits = n_bits
        self.radius = radius
        self.descriptors = descriptors
        self.column_prefix = column_prefix
        self.errors = errors
        self.cache_dir = cache_dir
        self.include_canonical_smiles = include_canonical_smiles
        self.return_dataframe = return_dataframe

    def fit(self, X, y=None):
        """Store input-column metadata and deterministic fingerprint names."""
        _validate_errors(self.errors)
        self.smiles_column_ = _resolve_column(X, self.smiles_column, kind="SMILES")
        empty_result = featurize_smiles(
            [],
            smiles_type=self.smiles_type,
            fingerprint_type=self.fingerprint_type,
            n_bits=self.n_bits,
            radius=self.radius,
            descriptors=self.descriptors,
            column_prefix=self.column_prefix,
            errors=self.errors,
            cache_dir=self.cache_dir,
        )
        feature_names = list(empty_result.features.columns)
        if self.include_canonical_smiles:
            feature_names = [empty_result.canonical_smiles.name, *feature_names]
        self.feature_names_out_ = np.asarray(feature_names, dtype=object)
        self.n_features_out_ = len(self.feature_names_out_)
        return self

    def transform(self, X):
        """Transform SMILES strings into RDKit fingerprints or descriptors."""
        check_is_fitted(self, "feature_names_out_")
        smiles_values, index = _extract_values(X, self.smiles_column_, kind="SMILES")
        result = featurize_smiles(
            smiles_values,
            smiles_type=self.smiles_type,
            fingerprint_type=self.fingerprint_type,
            n_bits=self.n_bits,
            radius=self.radius,
            descriptors=self.descriptors,
            column_prefix=self.column_prefix,
            errors=self.errors,
            cache_dir=self.cache_dir,
        )
        features = result.features.copy()
        features.index = index
        canonical_smiles = result.canonical_smiles.set_axis(index)

        if self.include_canonical_smiles:
            features = pd.concat([canonical_smiles, features], axis=1)
            features.columns = self.feature_names_out_

        self.canonical_smiles_ = canonical_smiles
        self.failed_ = result.failed
        self.cache_keys_ = result.cache_keys.set_axis(index)
        self.cache_hit_ = result.cache_hit.set_axis(index)
        self.last_result_ = result
        return _format_transform_output(features, self.return_dataframe)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output fingerprint or descriptor names."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class PolymerSmilesFeaturizer(SmilesFeaturizer):
    """Convenience transformer for two-ended polymer repeat-unit SMILES."""

    def __init__(
        self,
        *,
        smiles_column: str | int | None = None,
        fingerprint_type: str = "morgan",
        n_bits: int = 1024,
        radius: int = 2,
        descriptors: Sequence[str] = DEFAULT_RDKIT_DESCRIPTORS,
        column_prefix: str | None = None,
        errors: str = "raise",
        cache_dir: str | Path | None = None,
        include_canonical_smiles: bool = False,
        return_dataframe: bool = True,
    ):
        super().__init__(
            smiles_column=smiles_column,
            smiles_type="polymer",
            fingerprint_type=fingerprint_type,
            n_bits=n_bits,
            radius=radius,
            descriptors=descriptors,
            column_prefix=column_prefix,
            errors=errors,
            cache_dir=cache_dir,
            include_canonical_smiles=include_canonical_smiles,
            return_dataframe=return_dataframe,
        )


def _composition_feature_names(
    *,
    properties: Sequence[str],
    statistics: Sequence[str],
    column_prefix: str | None,
) -> list[str]:
    names = [f"{property_name}_{statistic}" for property_name in properties for statistic in statistics]
    if column_prefix is None:
        return names
    return [f"{column_prefix}_{name}" for name in names]


def _validate_errors(errors: str) -> None:
    if errors not in {"raise", "coerce"}:
        raise ValueError("errors must be either 'raise' or 'coerce'")


def _resolve_column(X, column: str | int | None, *, kind: str) -> str | int | None:
    if isinstance(X, pd.DataFrame):
        if column is None:
            if X.shape[1] != 1:
                raise ValueError(
                    f"{kind} input dataframe has {X.shape[1]} columns; provide a column name"
                )
            return X.columns[0]
        _validate_dataframe_column(X, column, kind=kind)
        return column

    if isinstance(X, pd.Series):
        return column

    if kind == "structure" and column is None and not isinstance(X, np.ndarray):
        return column

    array = np.asarray(X, dtype=object)
    if array.ndim == 1:
        return column
    if array.ndim == 2:
        if column is None:
            if array.shape[1] != 1:
                raise ValueError(
                    f"{kind} input array has {array.shape[1]} columns; provide an integer column"
                )
            return 0
        if not isinstance(column, int):
            raise ValueError(f"{kind} array input requires an integer column index")
        if column < 0 or column >= array.shape[1]:
            raise ValueError(f"{kind} column index {column} is out of bounds")
        return column

    raise ValueError(f"{kind} input must be one-dimensional or a two-dimensional table")


def _extract_values(X, column: str | int | None, *, kind: str) -> tuple[Sequence[object], pd.Index]:
    if isinstance(X, pd.DataFrame):
        _validate_dataframe_column(X, column, kind=kind)
        values = X.iloc[:, column] if isinstance(column, int) else X[column]
        return values.tolist(), X.index

    if isinstance(X, pd.Series):
        return X.tolist(), X.index

    if kind == "structure" and column is None and not isinstance(X, np.ndarray):
        values = list(X)
        return values, pd.RangeIndex(len(values))

    array = np.asarray(X, dtype=object)
    if array.ndim == 1:
        return array.tolist(), pd.RangeIndex(array.shape[0])
    if array.ndim == 2:
        column_index = 0 if column is None else column
        if not isinstance(column_index, int):
            raise ValueError(f"{kind} array input requires an integer column index")
        if column_index < 0 or column_index >= array.shape[1]:
            raise ValueError(f"{kind} column index {column_index} is out of bounds")
        return array[:, column_index].tolist(), pd.RangeIndex(array.shape[0])

    raise ValueError(f"{kind} input must be one-dimensional or a two-dimensional table")


def _validate_dataframe_column(X: pd.DataFrame, column: str | int | None, *, kind: str) -> None:
    if column is None:
        raise ValueError(f"{kind} dataframe input requires a fitted column")
    if isinstance(column, int):
        if column < 0 or column >= X.shape[1]:
            raise ValueError(f"{kind} column index {column} is out of bounds")
        return
    if column not in X.columns:
        raise KeyError(f"{kind} column '{column}' not found")


def _format_transform_output(features: pd.DataFrame, return_dataframe: bool):
    if return_dataframe:
        return features
    return features.to_numpy()
