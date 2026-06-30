from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import gcd
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CandidatePoolDiagnostics",
    "build_cartesian_candidate_grid",
    "build_composition_candidate_grid",
    "exclude_existing_candidates",
    "summarize_candidate_category_coverage",
    "summarize_candidate_feature_coverage",
    "summarize_candidate_pool",
    "summarize_candidate_duplicates",
    "split_candidate_features",
]


@dataclass(frozen=True)
class CandidatePoolDiagnostics:
    """Diagnostic tables for a finite candidate pool.

    Attributes
    ----------
    overview
        One-row pool-level summary with candidate counts, missing-feature
        counts, and duplicate-key counts.
    numeric_features
        Per-feature numeric descriptor summary.
    categorical_features
        Per-column categorical metadata summary.
    duplicate_keys
        Duplicate key groups, empty when no duplicate groups are found or no
        key columns were supplied.
    """

    overview: pd.DataFrame
    numeric_features: pd.DataFrame
    categorical_features: pd.DataFrame
    duplicate_keys: pd.DataFrame

    def overview_frame(self) -> pd.DataFrame:
        """Return a copy of the pool-level overview table."""
        return self.overview.copy()

    def numeric_feature_frame(self) -> pd.DataFrame:
        """Return a copy of the numeric feature summary table."""
        return self.numeric_features.copy()

    def categorical_feature_frame(self) -> pd.DataFrame:
        """Return a copy of the categorical metadata summary table."""
        return self.categorical_features.copy()

    def duplicate_key_frame(self) -> pd.DataFrame:
        """Return a copy of the duplicate-key diagnostics table."""
        return self.duplicate_keys.copy()


def build_cartesian_candidate_grid(
    parameter_space: Mapping[str, Iterable[Any] | Any],
    *,
    fixed_values: Mapping[str, Any] | None = None,
    candidate_id_column: str | None = "candidate_id",
    candidate_id_prefix: str = "candidate",
    max_candidates: int | None = 100_000,
) -> pd.DataFrame:
    """Build a finite candidate table from a Cartesian product of choices.

    Use this for processing conditions, formulation variables, solvents,
    categorical choices, or any finite design space that should be ranked by
    Bayesian optimization.

    Parameters
    ----------
    parameter_space
        Mapping from output column name to candidate values. Scalars and strings
        are treated as one-value choices.
    fixed_values
        Optional columns with constant metadata added to every candidate row.
    candidate_id_column
        Optional candidate identifier column. Set to `None` to omit IDs.
    candidate_id_prefix
        Prefix used when `candidate_id_column` is not `None`.
    max_candidates
        Optional safety limit. A `ValueError` is raised before returning very
        large grids.
    """
    columns, levels = _validate_parameter_space(parameter_space)
    fixed = _validate_fixed_values(fixed_values)
    conflicts = sorted(set(columns).intersection(fixed))
    if conflicts:
        raise ValueError(f"fixed_values conflict with parameter columns: {conflicts}")
    candidate_id_column = _normalize_candidate_id_column(candidate_id_column)
    _validate_candidate_id_column(candidate_id_column, columns + tuple(fixed))
    _validate_candidate_id_prefix(candidate_id_prefix)
    _validate_max_candidates(max_candidates)

    candidate_count = int(np.prod([len(level) for level in levels], dtype=object))
    if max_candidates is not None and candidate_count > max_candidates:
        raise ValueError(
            "Cartesian candidate grid would contain "
            f"{candidate_count} rows, above max_candidates={max_candidates}"
        )

    rows = [dict(zip(columns, values)) for values in product(*levels)]
    candidates = pd.DataFrame(rows, columns=list(columns))
    for column, value in fixed.items():
        candidates[column] = value

    _insert_candidate_ids(candidates, candidate_id_column, prefix=candidate_id_prefix)
    return candidates.reset_index(drop=True)


def build_composition_candidate_grid(
    elements: Sequence[str],
    *,
    step: float = 0.25,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0,
    min_components: int = 1,
    max_components: int | None = None,
    formula_column: str = "formula",
    fraction_prefix: str = "frac_",
    candidate_id_column: str | None = "candidate_id",
    candidate_id_prefix: str = "composition",
    max_candidates: int | None = 100_000,
) -> pd.DataFrame:
    """Build inorganic composition candidates on an element-fraction simplex.

    The grid uses integer stoichiometric units implied by `step`. For example,
    `step=0.25` creates compositions with fractions in increments of one
    quarter and formulas such as `"Al3Ni"` or `"AlNi"`.

    Parameters
    ----------
    elements
        Element symbols defining the composition space.
    step
        Fractional spacing. The value must divide 1.0, such as `0.5`, `0.25`,
        `0.2`, or `0.1`.
    min_fraction, max_fraction
        Bounds applied to each nonzero element fraction.
    min_components, max_components
        Bounds on the number of elements present in a candidate.
    formula_column
        Name of the generated reduced-formula column.
    fraction_prefix
        Prefix for element-fraction columns.
    candidate_id_column
        Optional candidate identifier column. Set to `None` to omit IDs.
    candidate_id_prefix
        Prefix used when `candidate_id_column` is not `None`.
    max_candidates
        Optional safety limit for generated rows.
    """
    symbols = _validate_element_symbols(elements)
    total_units = _fraction_step_units(step)
    _validate_fraction_bounds(min_fraction, max_fraction)
    min_components, max_components = _validate_component_bounds(
        min_components,
        max_components,
        n_elements=len(symbols),
    )
    if not str(formula_column).strip():
        raise ValueError("formula_column must be non-empty")
    if not str(fraction_prefix).strip():
        raise ValueError("fraction_prefix must be non-empty")
    formula_column = str(formula_column).strip()
    fraction_prefix = str(fraction_prefix).strip()

    fraction_columns = tuple(f"{fraction_prefix}{symbol}" for symbol in symbols)
    reserved_columns = (formula_column, "n_components") + fraction_columns
    candidate_id_column = _normalize_candidate_id_column(candidate_id_column)
    _validate_candidate_id_column(candidate_id_column, reserved_columns)
    _validate_candidate_id_prefix(candidate_id_prefix)
    _validate_max_candidates(max_candidates)

    rows: list[dict[str, Any]] = []
    for counts in _integer_simplex_points(len(symbols), total_units):
        fractions = np.asarray(counts, dtype=float) / float(total_units)
        n_components = int(np.count_nonzero(counts))
        if n_components < min_components or n_components > max_components:
            continue
        nonzero = fractions[fractions > 0.0]
        if np.any(nonzero < min_fraction) or np.any(nonzero > max_fraction):
            continue

        row: dict[str, Any] = {
            formula_column: _reduced_formula(symbols, counts),
            "n_components": n_components,
        }
        row.update({column: fraction for column, fraction in zip(fraction_columns, fractions)})
        rows.append(row)
        if max_candidates is not None and len(rows) > max_candidates:
            raise ValueError(
                "Composition candidate grid exceeded "
                f"max_candidates={max_candidates}"
            )

    candidates = pd.DataFrame(rows, columns=[formula_column, "n_components", *fraction_columns])
    _insert_candidate_ids(candidates, candidate_id_column, prefix=candidate_id_prefix)
    return candidates.reset_index(drop=True)


def exclude_existing_candidates(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    key_columns: Sequence[str],
    *,
    keep_indicator: bool = False,
    indicator_column: str = "matgpr_is_observed",
) -> pd.DataFrame:
    """Remove or annotate candidates that already exist in observed data.

    Parameters
    ----------
    candidates
        Candidate table to filter.
    observed
        Existing measured, simulated, or already-selected rows.
    key_columns
        Columns used to identify duplicates, such as `("formula",)` or
        `("formula", "temperature_c")`.
    keep_indicator
        If `True`, keep all rows and add `indicator_column`. If `False`, return
        only candidates not present in `observed`.
    indicator_column
        Name of the boolean indicator column used when `keep_indicator=True`.
    """
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")
    if not isinstance(observed, pd.DataFrame):
        raise TypeError("observed must be a pandas DataFrame")
    keys = _validate_columns(key_columns, candidates, name="key_columns")
    missing_observed = [column for column in keys if column not in observed.columns]
    if missing_observed:
        raise ValueError(f"observed is missing key columns: {missing_observed}")
    if not str(indicator_column).strip():
        raise ValueError("indicator_column must be non-empty")

    observed_keys = set(_row_keys(observed, keys))
    is_observed = pd.Series(
        [key in observed_keys for key in _row_keys(candidates, keys)],
        index=candidates.index,
        dtype=bool,
    )

    result = candidates.copy()
    if keep_indicator:
        result[indicator_column] = is_observed.to_numpy(dtype=bool)
        return result.reset_index(drop=True)
    return result.loc[~is_observed].reset_index(drop=True)


def split_candidate_features(
    candidates: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    metadata_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a candidate table into numeric features and candidate metadata.

    The returned `(X_candidates, candidate_data)` tuple can be passed directly
    to `suggest_next_experiments`.
    """
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")
    features = _validate_columns(feature_columns, candidates, name="feature_columns")
    non_numeric = [
        column
        for column in features
        if not pd.api.types.is_numeric_dtype(candidates[column])
    ]
    if non_numeric:
        raise ValueError(f"feature_columns must be numeric: {non_numeric}")

    if metadata_columns is None:
        metadata = tuple(column for column in candidates.columns if column not in features)
    else:
        metadata = _validate_columns(metadata_columns, candidates, name="metadata_columns")

    X_candidates = candidates.loc[:, list(features)].copy()
    candidate_data = candidates.loc[:, list(metadata)].copy()
    return X_candidates.reset_index(drop=True), candidate_data.reset_index(drop=True)


def summarize_candidate_pool(
    candidates: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
    key_columns: Sequence[str] | None = None,
) -> CandidatePoolDiagnostics:
    """Summarize a finite candidate pool before Bayesian optimization.

    This helper is intended for search-space audits: descriptor completeness,
    numeric feature ranges, categorical metadata diversity, and duplicate
    candidate keys. It does not rank candidates and does not require BoTorch.
    """
    frame = _validate_candidate_frame(candidates, name="candidates")
    features = _resolve_numeric_feature_columns(frame, feature_columns)
    categorical = _resolve_categorical_columns(
        frame,
        categorical_columns,
        exclude_columns=features,
    )
    duplicates = (
        summarize_candidate_duplicates(frame, key_columns)
        if key_columns is not None
        else _empty_duplicate_summary()
    )
    numeric_summary = _summarize_numeric_candidate_features(frame, features)
    categorical_summary = _summarize_categorical_candidate_columns(frame, categorical)
    overview = _summarize_candidate_pool_overview(
        frame,
        feature_columns=features,
        categorical_columns=categorical,
        duplicate_keys=duplicates,
    )
    return CandidatePoolDiagnostics(
        overview=overview,
        numeric_features=numeric_summary,
        categorical_features=categorical_summary,
        duplicate_keys=duplicates,
    )


def summarize_candidate_feature_coverage(
    candidates: pd.DataFrame,
    reference_data: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Compare numeric candidate-feature ranges with reference/training data.

    The returned table reports whether the finite candidate pool covers the
    reference range for each feature, and how much of the candidate pool lies
    outside the reference feature range. Use this before BO to identify
    extrapolative candidate pools or features with poor search-space coverage.
    """
    candidate_frame = _validate_candidate_frame(candidates, name="candidates")
    reference_frame = _validate_candidate_frame(reference_data, name="reference_data")
    features = _validate_columns(feature_columns, candidate_frame, name="feature_columns")
    missing_reference = [column for column in features if column not in reference_frame.columns]
    if missing_reference:
        raise ValueError(f"reference_data is missing feature columns: {missing_reference}")
    _validate_nonnegative_finite(tolerance, name="tolerance")

    rows: list[dict[str, Any]] = []
    for feature in features:
        candidate_values, candidate_invalid = _finite_numeric_values(
            candidate_frame[feature],
            label=f"candidates[{feature!r}]",
        )
        reference_values, reference_invalid = _finite_numeric_values(
            reference_frame[feature],
            label=f"reference_data[{feature!r}]",
        )
        rows.append(
            _feature_coverage_row(
                feature,
                candidate_values=candidate_values,
                reference_values=reference_values,
                candidate_invalid_count=candidate_invalid,
                reference_invalid_count=reference_invalid,
                tolerance=float(tolerance),
            )
        )
    return pd.DataFrame(rows)


def summarize_candidate_category_coverage(
    candidates: pd.DataFrame,
    reference_data: pd.DataFrame,
    categorical_columns: Sequence[str],
    *,
    max_levels: int = 20,
) -> pd.DataFrame:
    """Compare categorical candidate levels with reference/training data.

    This is useful for finite BO pools that include solvents, processing
    labels, synthesis routes, supplier choices, or discrete formulation
    classes. The summary highlights candidate-only levels and reference levels
    that are absent from the candidate pool.
    """
    candidate_frame = _validate_candidate_frame(candidates, name="candidates")
    reference_frame = _validate_candidate_frame(reference_data, name="reference_data")
    columns = _validate_columns(categorical_columns, candidate_frame, name="categorical_columns")
    missing_reference = [column for column in columns if column not in reference_frame.columns]
    if missing_reference:
        raise ValueError(f"reference_data is missing categorical columns: {missing_reference}")
    max_levels = _validate_positive_int(max_levels, name="max_levels")

    rows = []
    for column in columns:
        candidate_levels = _nonmissing_level_set(candidate_frame[column])
        reference_levels = _nonmissing_level_set(reference_frame[column])
        shared = candidate_levels.intersection(reference_levels)
        candidate_only = candidate_levels.difference(reference_levels)
        reference_only = reference_levels.difference(candidate_levels)
        reference_count = len(reference_levels)
        rows.append(
            {
                "column": column,
                "candidate_unique_count": len(candidate_levels),
                "reference_unique_count": reference_count,
                "shared_unique_count": len(shared),
                "reference_levels_covered_fraction": (
                    np.nan if reference_count == 0 else len(shared) / reference_count
                ),
                "candidate_new_level_count": len(candidate_only),
                "reference_missing_level_count": len(reference_only),
                "candidate_new_levels": _format_levels(candidate_only, max_levels=max_levels),
                "reference_missing_levels": _format_levels(reference_only, max_levels=max_levels),
            }
        )
    return pd.DataFrame(rows)


def summarize_candidate_duplicates(
    candidates: pd.DataFrame,
    key_columns: Sequence[str],
) -> pd.DataFrame:
    """Return duplicate candidate-key groups sorted by duplicate count."""
    frame = _validate_candidate_frame(candidates, name="candidates")
    keys = _validate_columns(key_columns, frame, name="key_columns")
    if frame.empty:
        return _empty_duplicate_summary(keys)

    duplicates = (
        frame.groupby(list(keys), dropna=False, sort=False)
        .size()
        .reset_index(name="matgpr_duplicate_count")
    )
    duplicates = duplicates.loc[duplicates["matgpr_duplicate_count"] > 1].copy()
    if duplicates.empty:
        return _empty_duplicate_summary(keys)

    duplicates["matgpr_duplicate_fraction"] = (
        duplicates["matgpr_duplicate_count"] / frame.shape[0]
    )
    return duplicates.sort_values(
        "matgpr_duplicate_count",
        ascending=False,
    ).reset_index(drop=True)


def _validate_candidate_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return frame.copy()


def _resolve_numeric_feature_columns(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    if feature_columns is None:
        return tuple(
            column
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
        )
    features = _validate_columns(feature_columns, frame, name="feature_columns")
    non_numeric = [
        column
        for column in features
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(f"feature_columns must be numeric: {non_numeric}")
    return features


def _resolve_categorical_columns(
    frame: pd.DataFrame,
    categorical_columns: Sequence[str] | None,
    *,
    exclude_columns: Sequence[str],
) -> tuple[str, ...]:
    excluded = set(exclude_columns)
    if categorical_columns is None:
        return tuple(
            column
            for column in frame.columns
            if column not in excluded and not pd.api.types.is_numeric_dtype(frame[column])
        )
    return _validate_columns(categorical_columns, frame, name="categorical_columns")


def _summarize_numeric_candidate_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    columns = [
        "feature",
        "dtype",
        "finite_count",
        "missing_or_invalid_count",
        "missing_or_invalid_fraction",
        "unique_count",
        "min",
        "q05",
        "median",
        "q95",
        "max",
        "mean",
        "std",
    ]
    if not feature_columns:
        return pd.DataFrame(columns=columns)

    rows = []
    n_rows = frame.shape[0]
    for feature in feature_columns:
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        invalid_count = int(n_rows - finite.shape[0])
        if finite.size == 0:
            rows.append(
                {
                    "feature": feature,
                    "dtype": str(frame[feature].dtype),
                    "finite_count": 0,
                    "missing_or_invalid_count": invalid_count,
                    "missing_or_invalid_fraction": _safe_fraction(invalid_count, n_rows),
                    "unique_count": 0,
                    "min": np.nan,
                    "q05": np.nan,
                    "median": np.nan,
                    "q95": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                }
            )
            continue

        rows.append(
            {
                "feature": feature,
                "dtype": str(frame[feature].dtype),
                "finite_count": int(finite.shape[0]),
                "missing_or_invalid_count": invalid_count,
                "missing_or_invalid_fraction": _safe_fraction(invalid_count, n_rows),
                "unique_count": int(pd.Series(finite).nunique(dropna=True)),
                "min": float(np.min(finite)),
                "q05": float(np.quantile(finite, 0.05)),
                "median": float(np.quantile(finite, 0.50)),
                "q95": float(np.quantile(finite, 0.95)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite, ddof=1)) if finite.shape[0] > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _summarize_categorical_candidate_columns(
    frame: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    columns = [
        "column",
        "dtype",
        "non_missing_count",
        "missing_count",
        "missing_fraction",
        "unique_count",
        "top_value",
        "top_count",
        "top_fraction",
    ]
    if not categorical_columns:
        return pd.DataFrame(columns=columns)

    rows = []
    n_rows = frame.shape[0]
    for column in categorical_columns:
        series = frame[column]
        missing_mask = series.isna()
        non_missing = series.loc[~missing_mask]
        value_counts = non_missing.astype(str).value_counts(dropna=True)
        if value_counts.empty:
            top_value = pd.NA
            top_count = 0
        else:
            top_value = value_counts.index[0]
            top_count = int(value_counts.iloc[0])
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_missing_count": int(non_missing.shape[0]),
                "missing_count": int(missing_mask.sum()),
                "missing_fraction": _safe_fraction(int(missing_mask.sum()), n_rows),
                "unique_count": len(_nonmissing_level_set(series)),
                "top_value": top_value,
                "top_count": top_count,
                "top_fraction": _safe_fraction(top_count, n_rows),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _summarize_candidate_pool_overview(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    duplicate_keys: pd.DataFrame,
) -> pd.DataFrame:
    n_candidates = frame.shape[0]
    feature_missing_rows = _count_rows_with_invalid_features(frame, feature_columns)
    duplicate_key_groups = int(duplicate_keys.shape[0])
    if "matgpr_duplicate_count" in duplicate_keys.columns:
        duplicate_candidate_rows = int(duplicate_keys["matgpr_duplicate_count"].sum())
        duplicate_excess_rows = int((duplicate_keys["matgpr_duplicate_count"] - 1).sum())
    else:
        duplicate_candidate_rows = 0
        duplicate_excess_rows = 0

    return pd.DataFrame(
        [
            {
                "matgpr_n_candidates": n_candidates,
                "matgpr_n_columns": frame.shape[1],
                "matgpr_n_numeric_features": len(feature_columns),
                "matgpr_n_categorical_columns": len(categorical_columns),
                "matgpr_feature_missing_rows": feature_missing_rows,
                "matgpr_feature_complete_fraction": (
                    np.nan
                    if n_candidates == 0
                    else 1.0 - (feature_missing_rows / n_candidates)
                ),
                "matgpr_duplicate_key_groups": duplicate_key_groups,
                "matgpr_duplicate_candidate_rows": duplicate_candidate_rows,
                "matgpr_duplicate_excess_rows": duplicate_excess_rows,
            }
        ]
    )


def _count_rows_with_invalid_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> int:
    if not feature_columns or frame.empty:
        return 0
    numeric = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    invalid = ~np.isfinite(values)
    return int(invalid.any(axis=1).sum())


def _empty_duplicate_summary(key_columns: Sequence[str] | None = None) -> pd.DataFrame:
    keys = [] if key_columns is None else list(key_columns)
    return pd.DataFrame(
        columns=[*keys, "matgpr_duplicate_count", "matgpr_duplicate_fraction"]
    )


def _finite_numeric_values(series: pd.Series, *, label: str) -> tuple[np.ndarray, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    invalid_count = int(values.shape[0] - finite.shape[0])
    if finite.size == 0:
        raise ValueError(f"{label} must contain at least one finite numeric value")
    return finite, invalid_count


def _feature_coverage_row(
    feature: str,
    *,
    candidate_values: np.ndarray,
    reference_values: np.ndarray,
    candidate_invalid_count: int,
    reference_invalid_count: int,
    tolerance: float,
) -> dict[str, Any]:
    candidate_min = float(np.min(candidate_values))
    candidate_max = float(np.max(candidate_values))
    reference_min = float(np.min(reference_values))
    reference_max = float(np.max(reference_values))
    candidate_span = candidate_max - candidate_min
    reference_span = reference_max - reference_min

    reference_lower = reference_min - tolerance
    reference_upper = reference_max + tolerance
    candidate_lower = candidate_min - tolerance
    candidate_upper = candidate_max + tolerance

    return {
        "feature": feature,
        "candidate_count": int(candidate_values.shape[0]),
        "candidate_missing_or_invalid_count": candidate_invalid_count,
        "reference_count": int(reference_values.shape[0]),
        "reference_missing_or_invalid_count": reference_invalid_count,
        "candidate_min": candidate_min,
        "candidate_max": candidate_max,
        "candidate_span": candidate_span,
        "reference_min": reference_min,
        "reference_max": reference_max,
        "reference_span": reference_span,
        "reference_range_covered_fraction": _range_covered_fraction(
            reference_min,
            reference_max,
            cover_min=candidate_lower,
            cover_max=candidate_upper,
        ),
        "candidate_outside_reference_fraction": float(
            np.mean(
                (candidate_values < reference_lower)
                | (candidate_values > reference_upper)
            )
        ),
        "candidate_below_reference_fraction": float(
            np.mean(candidate_values < reference_lower)
        ),
        "candidate_above_reference_fraction": float(
            np.mean(candidate_values > reference_upper)
        ),
        "reference_outside_candidate_fraction": float(
            np.mean(
                (reference_values < candidate_lower)
                | (reference_values > candidate_upper)
            )
        ),
    }


def _range_covered_fraction(
    target_min: float,
    target_max: float,
    *,
    cover_min: float,
    cover_max: float,
) -> float:
    target_span = target_max - target_min
    if np.isclose(target_span, 0.0):
        return float(cover_min <= target_min <= cover_max)
    overlap = max(0.0, min(target_max, cover_max) - max(target_min, cover_min))
    return float(min(1.0, overlap / target_span))


def _nonmissing_level_set(series: pd.Series) -> set[str]:
    levels = set()
    for value in series:
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        levels.add(str(value))
    return levels


def _format_levels(levels: set[str], *, max_levels: int) -> str:
    ordered = sorted(levels)
    shown = ordered[:max_levels]
    suffix = "" if len(ordered) <= max_levels else f"; ...(+{len(ordered) - max_levels} more)"
    return "; ".join(shown) + suffix


def _safe_fraction(numerator: int, denominator: int) -> float:
    return np.nan if denominator == 0 else float(numerator / denominator)


def _validate_nonnegative_finite(value: float, *, name: str) -> None:
    try:
        is_valid = bool(np.isfinite(value)) and value >= 0
    except TypeError:
        is_valid = False
    if not is_valid:
        raise ValueError(f"{name} must be a non-negative finite value")


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _validate_parameter_space(
    parameter_space: Mapping[str, Iterable[Any] | Any],
) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
    if not isinstance(parameter_space, Mapping):
        raise TypeError("parameter_space must be a mapping")
    if not parameter_space:
        raise ValueError("parameter_space must contain at least one parameter")

    columns: list[str] = []
    levels: list[tuple[Any, ...]] = []
    for column, values in parameter_space.items():
        name = str(column).strip()
        if not name:
            raise ValueError("parameter names must be non-empty")
        if name in columns:
            raise ValueError(f"Duplicate parameter name: {name!r}")
        value_tuple = _as_choice_tuple(values)
        if not value_tuple:
            raise ValueError(f"parameter_space[{name!r}] must contain at least one value")
        columns.append(name)
        levels.append(value_tuple)
    return tuple(columns), tuple(levels)


def _validate_fixed_values(fixed_values: Mapping[str, Any] | None) -> dict[str, Any]:
    if fixed_values is None:
        return {}
    if not isinstance(fixed_values, Mapping):
        raise TypeError("fixed_values must be a mapping when provided")
    fixed: dict[str, Any] = {}
    for column, value in fixed_values.items():
        name = str(column).strip()
        if not name:
            raise ValueError("fixed_values column names must be non-empty")
        if name in fixed:
            raise ValueError(f"Duplicate fixed value column: {name!r}")
        fixed[name] = value
    return fixed


def _as_choice_tuple(values: Iterable[Any] | Any) -> tuple[Any, ...]:
    if isinstance(values, str) or not isinstance(values, Iterable):
        return (values,)
    return tuple(values)


def _validate_element_symbols(elements: Sequence[str]) -> tuple[str, ...]:
    if isinstance(elements, str) or not isinstance(elements, Sequence):
        raise TypeError("elements must be a sequence of element symbols")
    if not elements:
        raise ValueError("elements must contain at least one element symbol")

    from pymatgen.core.periodic_table import Element

    symbols: list[str] = []
    for element in elements:
        symbol = str(element).strip()
        if not symbol:
            raise ValueError("element symbols must be non-empty")
        try:
            symbol = Element(symbol).symbol
        except Exception as exc:
            raise ValueError(f"Invalid element symbol: {element!r}") from exc
        if symbol in symbols:
            raise ValueError(f"Duplicate element symbol: {symbol!r}")
        symbols.append(symbol)
    return tuple(symbols)


def _fraction_step_units(step: float) -> int:
    if not np.isfinite(step) or step <= 0.0 or step > 1.0:
        raise ValueError("step must be a finite value in the interval (0, 1]")
    inverse = 1.0 / float(step)
    units = int(round(inverse))
    if not np.isclose(units * float(step), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("step must divide 1.0 into an integer number of intervals")
    return units


def _validate_fraction_bounds(min_fraction: float, max_fraction: float) -> None:
    if not np.isfinite(min_fraction) or not np.isfinite(max_fraction):
        raise ValueError("min_fraction and max_fraction must be finite")
    if min_fraction < 0.0 or max_fraction > 1.0 or min_fraction > max_fraction:
        raise ValueError("fraction bounds must satisfy 0 <= min_fraction <= max_fraction <= 1")


def _validate_component_bounds(
    min_components: int,
    max_components: int | None,
    *,
    n_elements: int,
) -> tuple[int, int]:
    if not isinstance(min_components, int) or min_components < 1:
        raise ValueError("min_components must be a positive integer")
    if max_components is None:
        max_components = n_elements
    if not isinstance(max_components, int) or max_components < 1:
        raise ValueError("max_components must be a positive integer")
    if min_components > max_components:
        raise ValueError("min_components must be <= max_components")
    if max_components > n_elements:
        raise ValueError("max_components must be <= number of elements")
    return min_components, max_components


def _integer_simplex_points(n_dimensions: int, total: int) -> Iterable[tuple[int, ...]]:
    if n_dimensions == 1:
        yield (total,)
        return
    for value in range(total, -1, -1):
        for rest in _integer_simplex_points(n_dimensions - 1, total - value):
            yield (value, *rest)


def _reduced_formula(symbols: Sequence[str], counts: Sequence[int]) -> str:
    nonzero = [(symbol, int(count)) for symbol, count in zip(symbols, counts) if count > 0]
    divisor = nonzero[0][1]
    for _, count in nonzero[1:]:
        divisor = gcd(divisor, count)
    reduced = [(symbol, count // divisor) for symbol, count in nonzero]
    return "".join(symbol if count == 1 else f"{symbol}{count}" for symbol, count in reduced)


def _insert_candidate_ids(
    candidates: pd.DataFrame,
    candidate_id_column: str | None,
    *,
    prefix: str,
) -> None:
    if candidate_id_column is None:
        return
    prefix = str(prefix).strip()
    width = max(6, len(str(max(len(candidates), 1))))
    candidate_ids = [f"{prefix}_{index:0{width}d}" for index in range(1, len(candidates) + 1)]
    candidates.insert(0, candidate_id_column, candidate_ids)


def _normalize_candidate_id_column(candidate_id_column: str | None) -> str | None:
    if candidate_id_column is None:
        return None
    return str(candidate_id_column).strip()


def _validate_candidate_id_column(
    candidate_id_column: str | None,
    existing_columns: Sequence[str],
) -> None:
    if candidate_id_column is None:
        return
    name = str(candidate_id_column).strip()
    if not name:
        raise ValueError("candidate_id_column must be non-empty when provided")
    if name in existing_columns:
        raise ValueError(f"candidate_id_column conflicts with existing column {name!r}")


def _validate_candidate_id_prefix(candidate_id_prefix: str) -> None:
    if not str(candidate_id_prefix).strip():
        raise ValueError("candidate_id_prefix must be non-empty")


def _validate_max_candidates(max_candidates: int | None) -> None:
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive when provided")


def _validate_columns(
    columns: Sequence[str],
    frame: pd.DataFrame,
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(columns, str) or not isinstance(columns, Sequence):
        raise TypeError(f"{name} must be a sequence of column names")
    resolved = tuple(str(column).strip() for column in columns)
    if not resolved:
        raise ValueError(f"{name} must contain at least one column")
    if any(not column for column in resolved):
        raise ValueError(f"{name} must not contain empty column names")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicate columns")
    missing = [column for column in resolved if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} are missing: {missing}")
    return resolved


def _row_keys(frame: pd.DataFrame, key_columns: Sequence[str]) -> list[tuple[Any, ...]]:
    return [
        tuple(_hashable_key_value(value) for value in row)
        for row in frame.loc[:, list(key_columns)].itertuples(index=False, name=None)
    ]


def _hashable_key_value(value: Any) -> Any:
    try:
        is_missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        is_missing = False
    if is_missing:
        return ("__matgpr_missing__",)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return tuple(value)
    return value
