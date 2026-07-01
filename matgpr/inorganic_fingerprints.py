from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .fingerprint_cache import (
    fingerprint_cache_key,
    read_fingerprint_cache_record,
    write_fingerprint_cache_record,
)

__all__ = [
    "CompositionFingerprintResult",
    "DEFAULT_COMPOSITION_STATISTICS",
    "DEFAULT_ELEMENTAL_PROPERTIES",
    "append_composition_fingerprints",
    "append_element_fractions",
    "clean_formula",
    "composition_fingerprint",
    "default_element_symbols",
    "element_fraction_fingerprint",
    "featurize_compositions",
    "featurize_element_fractions",
]

DEFAULT_ELEMENTAL_PROPERTIES: tuple[str, ...] = (
    "atomic_number",
    "atomic_mass",
    "X",
    "atomic_radius",
    "row",
    "column",
    "s_val",
    "p_val",
    "d_val",
    "f_val",
)

DEFAULT_COMPOSITION_STATISTICS: tuple[str, ...] = (
    "min",
    "max",
    "range",
    "fwm",
    "ad",
    "std",
)

_ORBITAL_PATTERN = re.compile(r"(\d+)([spdf])(\d+)")


@dataclass(frozen=True)
class CompositionFingerprintResult:
    """Result returned by composition featurization helpers.

    Attributes
    ----------
    features
        Numeric composition descriptors indexed like the input formulas.
    failed
        Formulas that could not be parsed or featurized. Empty when
        ``errors="raise"`` and no exception was raised.
    cache_keys
        Deterministic cache keys for each input row.
    cache_hit
        Boolean flags indicating whether each row was loaded from cache.
    """

    features: pd.DataFrame
    failed: pd.DataFrame
    cache_keys: pd.Series | None = None
    cache_hit: pd.Series | None = None


def clean_formula(formula: object) -> str:
    """Normalize common text artifacts before parsing an inorganic formula."""
    if formula is None or (isinstance(formula, float) and np.isnan(formula)):
        return ""
    return str(formula).replace("\u00a0", "").strip()


def composition_fingerprint(
    formula: object,
    *,
    properties: Sequence[str] = DEFAULT_ELEMENTAL_PROPERTIES,
    statistics: Sequence[str] = DEFAULT_COMPOSITION_STATISTICS,
) -> dict[str, float]:
    """Create statistical elemental-property descriptors for one composition.

    The default descriptor set follows a compact materials-informatics pattern:
    each elemental property is summarized by composition-weighted statistics
    such as the fraction-weighted mean (``fwm``), absolute deviation (``ad``),
    minimum, maximum, range, and standard deviation.

    Parameters
    ----------
    formula
        Inorganic composition string, for example ``"B4C"`` or
        ``"Ag0.05Gd0.048Pd0.902"``.
    properties
        Element-level properties to summarize. Supported defaults include
        atomic number, mass, electronegativity, radius, periodic-table row and
        column, and s/p/d/f valence counts.
    statistics
        Summary statistics to compute for each property.
    """
    composition = _parse_composition(formula)
    element_amounts = composition.get_el_amt_dict()
    total_amount = float(sum(element_amounts.values()))
    if total_amount <= 0:
        raise ValueError(f"Composition '{formula}' has no positive element amounts")

    elements = [_element(symbol) for symbol in element_amounts]
    fractions = np.array([element_amounts[element.symbol] / total_amount for element in elements], dtype=float)

    features: dict[str, float] = {}
    for property_name in properties:
        values = np.array([_element_property(element, property_name) for element in elements], dtype=float)
        for statistic in statistics:
            features[f"{property_name}_{statistic}"] = _weighted_statistic(
                values,
                fractions,
                statistic,
                property_name=property_name,
            )
    return features


def default_element_symbols(max_atomic_number: int = 118) -> tuple[str, ...]:
    """Return periodic-table element symbols up to ``max_atomic_number``."""
    if not isinstance(max_atomic_number, int) or max_atomic_number < 1:
        raise ValueError("max_atomic_number must be a positive integer")
    if max_atomic_number > 118:
        raise ValueError("max_atomic_number must be less than or equal to 118")
    return tuple(_element_from_atomic_number(z).symbol for z in range(1, max_atomic_number + 1))


def element_fraction_fingerprint(
    formula: object,
    *,
    elements: Sequence[str] | None = None,
) -> dict[str, float]:
    """Create a fixed-length elemental-fraction vector for one formula.

    Each output feature is named ``element_fraction_<symbol>``. If ``elements``
    is omitted, the vector spans all elements up to atomic number 118. Passing a
    smaller element list is useful for project-specific models, but every
    element present in ``formula`` must be included in that list.
    """
    symbols = _validate_element_symbols(default_element_symbols() if elements is None else elements)
    return _element_fraction_values(formula, symbols)


def featurize_element_fractions(
    formulas: Iterable[object],
    *,
    elements: Sequence[str] | None = None,
    errors: str = "raise",
    cache_dir: str | Path | None = None,
) -> CompositionFingerprintResult:
    """Featurize formulas as fixed element-fraction vectors.

    Element-fraction vectors are the recommended inputs for
    :class:`matgpr.ElementFractionKernel`. They preserve direct compositional
    information, unlike statistical elemental-property descriptors.
    """
    if errors not in {"raise", "coerce"}:
        raise ValueError("errors must be either 'raise' or 'coerce'")

    symbols = _validate_element_symbols(default_element_symbols() if elements is None else elements)
    feature_names = _element_fraction_feature_names(symbols)
    cache_parameters = {"elements": list(symbols)}

    rows: list[Mapping[str, float]] = []
    failures: list[dict[str, object]] = []
    cache_keys: list[str] = []
    cache_hits: list[bool] = []

    for index, formula in enumerate(formulas):
        normalized_formula = clean_formula(formula)
        cache_key = fingerprint_cache_key(
            namespace="element_fraction",
            value=normalized_formula,
            parameters=cache_parameters,
        )
        cache_keys.append(cache_key)

        try:
            cached_features = _read_cached_feature_row(
                cache_dir,
                namespace="element_fraction",
                cache_key=cache_key,
                feature_names=feature_names,
            )
            if cached_features is not None:
                rows.append(cached_features)
                cache_hits.append(True)
                continue

            features = _element_fraction_values(formula, symbols)
            rows.append(features)
            cache_hits.append(False)
            _write_cached_feature_row(
                cache_dir,
                namespace="element_fraction",
                cache_key=cache_key,
                input_value=normalized_formula,
                features=features,
                metadata=cache_parameters,
            )
        except Exception as exc:
            if errors == "raise":
                raise ValueError(
                    f"Could not featurize element fractions at position {index}: {formula!r}"
                ) from exc
            failures.append(
                {
                    "index": index,
                    "formula": formula,
                    "cache_key": cache_key,
                    "error": str(exc),
                }
            )
            rows.append({name: np.nan for name in feature_names})
            cache_hits.append(False)

    return CompositionFingerprintResult(
        features=pd.DataFrame(rows, columns=feature_names),
        failed=pd.DataFrame(failures, columns=["index", "formula", "cache_key", "error"]),
        cache_keys=pd.Series(cache_keys, name="element_fraction_cache_key"),
        cache_hit=pd.Series(cache_hits, name="element_fraction_cache_hit"),
    )


def featurize_compositions(
    formulas: Iterable[object],
    *,
    properties: Sequence[str] = DEFAULT_ELEMENTAL_PROPERTIES,
    statistics: Sequence[str] = DEFAULT_COMPOSITION_STATISTICS,
    errors: str = "raise",
    cache_dir: str | Path | None = None,
) -> CompositionFingerprintResult:
    """Featurize a sequence of inorganic formulas.

    Parameters
    ----------
    formulas
        Iterable of formula strings.
    properties, statistics
        Descriptor controls passed to ``composition_fingerprint``.
    errors
        ``"raise"`` stops at the first invalid formula. ``"coerce"`` returns
        rows filled with ``NaN`` and records failures in ``failed``.
    cache_dir
        Optional directory for persistent row-level JSON cache files. Cache keys
        are deterministic hashes of the normalized formula and descriptor
        settings.
    """
    if errors not in {"raise", "coerce"}:
        raise ValueError("errors must be either 'raise' or 'coerce'")

    rows: list[Mapping[str, float]] = []
    failures: list[dict[str, object]] = []
    cache_keys: list[str] = []
    cache_hits: list[bool] = []
    feature_names = [f"{property_name}_{statistic}" for property_name in properties for statistic in statistics]
    cache_parameters = {
        "properties": list(properties),
        "statistics": list(statistics),
    }

    for index, formula in enumerate(formulas):
        normalized_formula = clean_formula(formula)
        cache_key = fingerprint_cache_key(
            namespace="composition",
            value=normalized_formula,
            parameters=cache_parameters,
        )
        cache_keys.append(cache_key)

        try:
            cached_features = _read_cached_feature_row(
                cache_dir,
                namespace="composition",
                cache_key=cache_key,
                feature_names=feature_names,
            )
            if cached_features is not None:
                rows.append(cached_features)
                cache_hits.append(True)
                continue

            features = composition_fingerprint(
                formula,
                properties=properties,
                statistics=statistics,
            )
            rows.append(features)
            cache_hits.append(False)
            _write_cached_feature_row(
                cache_dir,
                namespace="composition",
                cache_key=cache_key,
                input_value=normalized_formula,
                features=features,
                metadata=cache_parameters,
            )
        except Exception as exc:
            if errors == "raise":
                raise ValueError(f"Could not featurize formula at position {index}: {formula!r}") from exc
            failures.append(
                {
                    "index": index,
                    "formula": formula,
                    "cache_key": cache_key,
                    "error": str(exc),
                }
            )
            rows.append({name: np.nan for name in feature_names})
            cache_hits.append(False)

    return CompositionFingerprintResult(
        features=pd.DataFrame(rows, columns=feature_names),
        failed=pd.DataFrame(failures, columns=["index", "formula", "cache_key", "error"]),
        cache_keys=pd.Series(cache_keys, name="composition_cache_key"),
        cache_hit=pd.Series(cache_hits, name="composition_cache_hit"),
    )


def append_composition_fingerprints(
    data: pd.DataFrame,
    *,
    formula_column: str = "composition",
    drop_formula_column: bool = False,
    errors: str = "raise",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Append inorganic composition descriptors to a dataframe.

    The returned dataframe preserves the original index and columns, then adds
    descriptor columns produced by ``featurize_compositions``.
    """
    if formula_column not in data.columns:
        raise KeyError(f"Formula column '{formula_column}' not found")

    result = featurize_compositions(data[formula_column], errors=errors, cache_dir=cache_dir)
    features = result.features.set_index(data.index)
    base = data.drop(columns=[formula_column]) if drop_formula_column else data.copy()
    return pd.concat([base, features], axis=1)


def append_element_fractions(
    data: pd.DataFrame,
    *,
    formula_column: str = "composition",
    elements: Sequence[str] | None = None,
    drop_formula_column: bool = False,
    errors: str = "raise",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Append element-fraction columns to a dataframe."""
    if formula_column not in data.columns:
        raise KeyError(f"Formula column '{formula_column}' not found")

    result = featurize_element_fractions(
        data[formula_column],
        elements=elements,
        errors=errors,
        cache_dir=cache_dir,
    )
    features = result.features.set_index(data.index)
    base = data.drop(columns=[formula_column]) if drop_formula_column else data.copy()
    return pd.concat([base, features], axis=1)


def _read_cached_feature_row(
    cache_dir: str | Path | None,
    *,
    namespace: str,
    cache_key: str,
    feature_names: Sequence[str],
) -> dict[str, float] | None:
    record = read_fingerprint_cache_record(cache_dir, namespace=namespace, cache_key=cache_key)
    if record is None:
        return None
    features = record.get("features")
    if not isinstance(features, Mapping):
        return None
    if any(name not in features for name in feature_names):
        return None
    return {name: _cached_float(features[name]) for name in feature_names}


def _write_cached_feature_row(
    cache_dir: str | Path | None,
    *,
    namespace: str,
    cache_key: str,
    input_value: object,
    features: Mapping[str, float],
    metadata: Mapping[str, object],
) -> None:
    write_fingerprint_cache_record(
        cache_dir,
        namespace=namespace,
        cache_key=cache_key,
        record={
            "input": input_value,
            "features": dict(features),
            "metadata": dict(metadata),
        },
    )


def _cached_float(value: object) -> float:
    if value is None:
        return np.nan
    return float(value)


def _parse_composition(formula: object):
    try:
        from pymatgen.core import Composition
    except ImportError as exc:
        raise ImportError(
            "pymatgen is required for inorganic composition fingerprints. "
            "Install it with `conda install -c conda-forge pymatgen`."
        ) from exc

    normalized = clean_formula(formula)
    if not normalized:
        raise ValueError("Formula is empty")
    return Composition(normalized)


def _element_from_atomic_number(atomic_number: int):
    try:
        from pymatgen.core import Element
    except ImportError as exc:
        raise ImportError(
            "pymatgen is required for inorganic composition fingerprints. "
            "Install it with `conda install -c conda-forge pymatgen`."
        ) from exc

    return Element.from_Z(atomic_number)


def _element(symbol: str):
    try:
        from pymatgen.core import Element
    except ImportError as exc:
        raise ImportError(
            "pymatgen is required for inorganic composition fingerprints. "
            "Install it with `conda install -c conda-forge pymatgen`."
        ) from exc

    return Element(symbol)


def _validate_element_symbols(elements: Sequence[str]) -> tuple[str, ...]:
    symbols = tuple(str(symbol).strip() for symbol in elements)
    if not symbols:
        raise ValueError("elements must contain at least one element symbol")
    if any(not symbol for symbol in symbols):
        raise ValueError("elements must contain non-empty element symbols")
    if len(set(symbols)) != len(symbols):
        raise ValueError("elements must not contain duplicate symbols")
    for symbol in symbols:
        _element(symbol)
    return symbols


def _element_fraction_values(formula: object, symbols: Sequence[str]) -> dict[str, float]:
    composition = _parse_composition(formula)
    element_amounts = composition.get_el_amt_dict()
    total_amount = float(sum(element_amounts.values()))
    if total_amount <= 0:
        raise ValueError(f"Composition '{formula}' has no positive element amounts")

    missing_symbols = sorted(set(element_amounts) - set(symbols))
    if missing_symbols:
        missing = ", ".join(missing_symbols)
        raise ValueError(f"elements does not include symbols present in formula: {missing}")

    return {
        f"element_fraction_{symbol}": float(element_amounts.get(symbol, 0.0)) / total_amount
        for symbol in symbols
    }


def _element_fraction_feature_names(elements: Sequence[str]) -> list[str]:
    return [f"element_fraction_{symbol}" for symbol in elements]


def _element_property(element, property_name: str) -> float:
    if property_name == "atomic_number":
        return float(element.Z)
    if property_name == "atomic_mass":
        return float(element.atomic_mass)
    if property_name == "X":
        return _as_float_or_nan(element.X)
    if property_name == "atomic_radius":
        return _as_float_or_nan(element.atomic_radius)
    if property_name == "row":
        return float(element.row)
    if property_name == "column":
        return float(element.group)
    if property_name in {"s_val", "p_val", "d_val", "f_val"}:
        return float(_orbital_valence_count(element, property_name[0]))
    raise ValueError(f"Unsupported elemental property '{property_name}'")


def _orbital_valence_count(element, orbital: str) -> int:
    counts = {"s": 0, "p": 0, "d": 0, "f": 0}
    for _, orbital_name, electron_count in _ORBITAL_PATTERN.findall(element.electronic_structure):
        counts[orbital_name] += int(electron_count)
    return counts[orbital]


def _weighted_statistic(
    values: np.ndarray,
    fractions: np.ndarray,
    statistic: str,
    *,
    property_name: str,
) -> float:
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.nan

    values = values[finite_mask]
    fractions = fractions[finite_mask]
    fractions = fractions / fractions.sum()
    mean = float(np.dot(fractions, values))

    if statistic == "min":
        return float(np.min(values))
    if statistic == "max":
        return float(np.max(values))
    if statistic == "range":
        return float(np.max(values) - np.min(values))
    if statistic == "fwm":
        return mean
    if statistic == "ad":
        return float(np.dot(fractions, np.abs(values - mean)))
    if statistic == "std":
        variance = np.dot(fractions, (values - mean) ** 2)
        return float(np.sqrt(max(variance, 0.0)))

    raise ValueError(f"Unsupported statistic '{statistic}' for property '{property_name}'")


def _as_float_or_nan(value: object) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
