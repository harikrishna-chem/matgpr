from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from math import gcd
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "build_cartesian_candidate_grid",
    "build_composition_candidate_grid",
    "exclude_existing_candidates",
    "split_candidate_features",
]


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
