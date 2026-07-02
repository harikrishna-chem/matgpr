from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .inorganic_fingerprints import (
    CompositionFingerprintResult,
    DEFAULT_COMPOSITION_STATISTICS,
    DEFAULT_ELEMENTAL_PROPERTIES,
    clean_formula,
    featurize_compositions,
)
from .optional_dependencies import require_optional_dependency
from .organic_fingerprints import DEFAULT_RDKIT_DESCRIPTORS, featurize_smiles
from .structure_fingerprints import (
    DEFAULT_STRUCTURE_FEATURES,
    featurize_structures,
    structure_feature_names,
)

__all__ = [
    "CompositionFeaturizer",
    "DEFAULT_MAGPIE_PROPERTIES",
    "DEFAULT_MAGPIE_STATISTICS",
    "MagpieCompositionFeaturizer",
    "PolymerSmilesFeaturizer",
    "SmilesFeaturizer",
    "StructureFeaturizer",
    "append_magpie_composition_features",
    "featurize_magpie_compositions",
]

DEFAULT_MAGPIE_PROPERTIES: tuple[str, ...] = (
    "Number",
    "MendeleevNumber",
    "AtomicWeight",
    "MeltingT",
    "Column",
    "Row",
    "CovalentRadius",
    "Electronegativity",
    "NsValence",
    "NpValence",
    "NdValence",
    "NfValence",
    "NValence",
    "NsUnfilled",
    "NpUnfilled",
    "NdUnfilled",
    "NfUnfilled",
    "NUnfilled",
    "GSvolume_pa",
    "GSbandgap",
    "GSmagmom",
    "SpaceGroupNumber",
)

DEFAULT_MAGPIE_STATISTICS: tuple[str, ...] = (
    "minimum",
    "maximum",
    "range",
    "mean",
    "avg_dev",
    "mode",
)


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


class MagpieCompositionFeaturizer(TransformerMixin, BaseEstimator):
    """Scikit-learn-style wrapper for matminer Magpie composition descriptors.

    Magpie descriptors summarize elemental properties with composition-weighted
    statistics. This transformer keeps `matminer` optional: the backend is
    imported only when ``fit`` or the low-level Magpie helper functions are
    called.
    """

    def __init__(
        self,
        *,
        formula_column: str | int | None = None,
        properties: Sequence[str] | None = None,
        statistics: Sequence[str] | None = None,
        column_prefix: str | None = "magpie",
        impute_nan: bool = True,
        errors: str = "raise",
        return_dataframe: bool = True,
    ):
        self.formula_column = formula_column
        self.properties = properties
        self.statistics = statistics
        self.column_prefix = column_prefix
        self.impute_nan = impute_nan
        self.errors = errors
        self.return_dataframe = return_dataframe

    def fit(self, X, y=None):
        """Store input-column metadata and Magpie descriptor names."""
        _validate_errors(self.errors)
        self.formula_column_ = _resolve_column(X, self.formula_column, kind="formula")
        self.magpie_featurizer_ = _build_magpie_featurizer(
            properties=self.properties,
            statistics=self.statistics,
            impute_nan=self.impute_nan,
        )
        self.feature_names_out_ = np.asarray(
            _magpie_feature_names(
                self.magpie_featurizer_.feature_labels(),
                column_prefix=self.column_prefix,
            ),
            dtype=object,
        )
        self.n_features_out_ = len(self.feature_names_out_)
        return self

    def transform(self, X):
        """Transform formulas into matminer Magpie composition descriptors."""
        check_is_fitted(self, ["feature_names_out_", "magpie_featurizer_"])
        formulas, index = _extract_values(X, self.formula_column_, kind="formula")
        result = _featurize_magpie_compositions_with_backend(
            formulas,
            featurizer=self.magpie_featurizer_,
            feature_names=self.feature_names_out_,
            errors=self.errors,
        )
        features = result.features.copy()
        features.index = index
        self.failed_ = result.failed
        self.last_result_ = result
        return _format_transform_output(features, self.return_dataframe)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output Magpie descriptor names."""
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


def featurize_magpie_compositions(
    formulas: Sequence[object],
    *,
    properties: Sequence[str] | None = None,
    statistics: Sequence[str] | None = None,
    column_prefix: str | None = "magpie",
    impute_nan: bool = True,
    errors: str = "raise",
) -> CompositionFingerprintResult:
    """Featurize inorganic formulas with matminer Magpie descriptors.

    Parameters
    ----------
    formulas
        Formula strings or ``pymatgen.Composition`` objects.
    properties, statistics
        Optional Magpie elemental properties and summary statistics. When
        omitted, matminer's standard Magpie property/statistic set is used.
    column_prefix
        Prefix added to cleaned descriptor names. The default ``"magpie"``
        avoids collisions with lightweight `matgpr` composition descriptors.
    impute_nan
        Passed to matminer's ``ElementProperty`` featurizer.
    errors
        ``"raise"`` stops at the first invalid formula. ``"coerce"`` returns
        rows filled with ``NaN`` and records failures in ``failed``.
    """
    _validate_errors(errors)
    featurizer = _build_magpie_featurizer(
        properties=properties,
        statistics=statistics,
        impute_nan=impute_nan,
    )
    feature_names = _magpie_feature_names(
        featurizer.feature_labels(),
        column_prefix=column_prefix,
    )
    return _featurize_magpie_compositions_with_backend(
        formulas,
        featurizer=featurizer,
        feature_names=feature_names,
        errors=errors,
    )


def append_magpie_composition_features(
    data: pd.DataFrame,
    *,
    formula_column: str = "composition",
    drop_formula_column: bool = False,
    properties: Sequence[str] | None = None,
    statistics: Sequence[str] | None = None,
    column_prefix: str | None = "magpie",
    impute_nan: bool = True,
    errors: str = "raise",
) -> pd.DataFrame:
    """Append matminer Magpie descriptors to a dataframe."""
    if formula_column not in data.columns:
        raise KeyError(f"Formula column '{formula_column}' not found")

    result = featurize_magpie_compositions(
        data[formula_column],
        properties=properties,
        statistics=statistics,
        column_prefix=column_prefix,
        impute_nan=impute_nan,
        errors=errors,
    )
    features = result.features.set_index(data.index)
    base = data.drop(columns=[formula_column]) if drop_formula_column else data.copy()
    return pd.concat([base, features], axis=1)


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


def _build_magpie_featurizer(
    *,
    properties: Sequence[str] | None,
    statistics: Sequence[str] | None,
    impute_nan: bool,
):
    require_optional_dependency("matminer", purpose="Matminer Magpie composition descriptors")
    composition_module = require_optional_dependency(
        "matminer.featurizers.composition",
        purpose="Matminer Magpie composition descriptors",
        extra="materials-extra",
        package_name="matminer",
    )
    element_property = composition_module.ElementProperty
    return element_property(
        data_source="magpie",
        features=list(DEFAULT_MAGPIE_PROPERTIES if properties is None else properties),
        stats=list(DEFAULT_MAGPIE_STATISTICS if statistics is None else statistics),
        impute_nan=impute_nan,
    )


def _featurize_magpie_compositions_with_backend(
    formulas: Sequence[object],
    *,
    featurizer,
    feature_names: Sequence[str],
    errors: str,
) -> CompositionFingerprintResult:
    rows: list[list[float]] = []
    failures: list[dict[str, object]] = []

    for index, formula in enumerate(formulas):
        try:
            composition = _as_pymatgen_composition(formula)
            rows.append([float(value) for value in featurizer.featurize(composition)])
        except Exception as exc:
            if errors == "raise":
                raise ValueError(
                    f"Could not featurize formula with Magpie at position {index}: {formula!r}"
                ) from exc
            failures.append(
                {
                    "index": index,
                    "formula": formula,
                    "error": str(exc),
                }
            )
            rows.append([np.nan] * len(feature_names))

    return CompositionFingerprintResult(
        features=pd.DataFrame(rows, columns=feature_names),
        failed=pd.DataFrame(failures, columns=["index", "formula", "error"]),
        cache_keys=None,
        cache_hit=None,
    )


def _as_pymatgen_composition(value: object):
    if hasattr(value, "element_composition") and hasattr(value, "get_el_amt_dict"):
        return value
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        raise ValueError("formula is missing")
    formula = clean_formula(value)
    if not formula:
        raise ValueError("formula is empty")
    return Composition(formula)


def _magpie_feature_names(
    labels: Sequence[str],
    *,
    column_prefix: str | None,
) -> list[str]:
    names = [_clean_magpie_label(label) for label in labels]
    if column_prefix is None:
        return names
    prefix = _snake_case(column_prefix)
    return [f"{prefix}_{name}" for name in names]


def _clean_magpie_label(label: str) -> str:
    pieces = str(label).split()
    if len(pieces) >= 3 and pieces[0] == "MagpieData":
        statistic = _snake_case(pieces[1])
        property_name = _snake_case(" ".join(pieces[2:]))
        return f"{property_name}_{statistic}"
    return _snake_case(label)


def _snake_case(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = text.strip("_").lower()
    return text or "feature"


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
