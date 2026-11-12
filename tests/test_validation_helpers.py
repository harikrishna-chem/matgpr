"""Focused tests for shared private validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

import matgpr.reporting as reporting
import matgpr.validation as validation
import matgpr.visualization as visualization
from matgpr._validation import to_1d_finite


@pytest.mark.unit
def test_to_1d_finite_converts_and_flattens_values() -> None:
    result = to_1d_finite([[1, 2], [3, 4]], "values")

    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0, 4.0]))
    assert result.dtype == np.dtype(float)
    assert result.ndim == 1


@pytest.mark.unit
def test_to_1d_finite_rejects_empty_input_with_exact_message() -> None:
    with pytest.raises(ValueError, match="^values must contain at least one value$"):
        to_1d_finite([], "values")


@pytest.mark.unit
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_to_1d_finite_rejects_nonfinite_input_with_exact_message(
    invalid_value: float,
) -> None:
    with pytest.raises(ValueError, match="^values must contain only finite values$"):
        to_1d_finite([1.0, invalid_value], "values")


@pytest.mark.unit
def test_local_finite_vector_helpers_use_canonical_owner() -> None:
    assert reporting._to_1d_numeric is to_1d_finite
    assert validation._to_1d_float is to_1d_finite
    assert visualization._to_1d_float_array is to_1d_finite
