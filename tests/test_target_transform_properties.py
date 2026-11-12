from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from matgpr import (
    BoundedTargetTransform,
    IdentityTargetTransform,
    LogTargetTransform,
    PhysicsResidualTransform,
    StandardizedTargetTransform,
)
from matgpr.gpytorch_gpr import GPyTorchPrediction

pytestmark = pytest.mark.scientific

_FINITE = st.floats(
    min_value=-1.0e4,
    max_value=1.0e4,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_POSITIVE = st.floats(
    min_value=1.0e-5,
    max_value=1.0e4,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_NONNEGATIVE_STD = st.floats(
    min_value=0.0,
    max_value=3.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@st.composite
def _finite_arrays(
    draw: st.DrawFn,
    *,
    min_size: int = 1,
    max_size: int = 20,
) -> np.ndarray:
    return np.asarray(
        draw(st.lists(_FINITE, min_size=min_size, max_size=max_size)),
        dtype=float,
    )


@st.composite
def _nonconstant_arrays(draw: st.DrawFn) -> np.ndarray:
    values = np.asarray(
        draw(st.lists(_FINITE, min_size=2, max_size=20)),
        dtype=float,
    )
    assume(float(np.ptp(values)) > 1.0e-8)
    assume(float(np.std(values, ddof=0)) > 1.0e-10)
    return values


@st.composite
def _positive_arrays(draw: st.DrawFn) -> np.ndarray:
    return np.asarray(
        draw(st.lists(_POSITIVE, min_size=1, max_size=20)),
        dtype=float,
    )


@st.composite
def _paired_finite_arrays(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    size = draw(st.integers(min_value=1, max_value=20))
    values = draw(st.lists(_FINITE, min_size=size, max_size=size))
    baseline = draw(st.lists(_FINITE, min_size=size, max_size=size))
    return np.asarray(values, dtype=float), np.asarray(baseline, dtype=float)


@st.composite
def _bounded_case(draw: st.DrawFn) -> tuple[float, float, np.ndarray]:
    lower = draw(_FINITE)
    width = draw(
        st.floats(
            min_value=1.0e-3,
            max_value=1.0e4,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        )
    )
    fractions = np.asarray(
        draw(
            st.lists(
                st.floats(
                    min_value=1.0e-6,
                    max_value=1.0 - 1.0e-6,
                    allow_nan=False,
                    allow_infinity=False,
                    width=64,
                ),
                min_size=1,
                max_size=20,
            )
        ),
        dtype=float,
    )
    return lower, lower + width, lower + width * fractions


@st.composite
def _mean_std_arrays(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    pairs = draw(
        st.lists(
            st.tuples(
                st.floats(
                    min_value=-10.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                    width=64,
                ),
                _NONNEGATIVE_STD,
            ),
            min_size=1,
            max_size=20,
        )
    )
    return (
        np.asarray([pair[0] for pair in pairs], dtype=float),
        np.asarray([pair[1] for pair in pairs], dtype=float),
    )


@given(values=_finite_arrays())
def test_identity_round_trip_preserves_values_shape_and_order(values: np.ndarray) -> None:
    transform = IdentityTargetTransform()

    transformed = transform.fit_transform(values)
    restored = transform.inverse_transform(transformed)

    np.testing.assert_array_equal(transformed, values)
    np.testing.assert_array_equal(restored, values)
    assert transformed.shape == values.shape


@given(values=_nonconstant_arrays())
def test_standardized_round_trip_and_moments_hold_for_finite_targets(
    values: np.ndarray,
) -> None:
    transform = StandardizedTargetTransform()

    transformed = transform.fit_transform(values)
    restored = transform.inverse_transform(transformed)

    np.testing.assert_allclose(restored, values, rtol=1.0e-12, atol=1.0e-10)
    assert float(np.mean(transformed)) == pytest.approx(0.0, abs=1.0e-10)
    assert float(np.std(transformed, ddof=0)) == pytest.approx(1.0, rel=1.0e-10)
    assert np.all(np.diff(transform.transform(np.sort(values))) >= 0.0)


@given(
    shifted_values=_positive_arrays(),
    offset=st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)
def test_log_round_trip_is_finite_and_monotonic(
    shifted_values: np.ndarray,
    offset: float,
) -> None:
    transform = LogTargetTransform(offset=offset)
    values = shifted_values - offset

    transformed = transform.fit_transform(values)
    restored = transform.inverse_transform(transformed)

    np.testing.assert_allclose(restored, values, rtol=1.0e-11, atol=1.0e-10)
    assert np.isfinite(transformed).all()
    assert np.all(np.diff(transform.transform(np.sort(values))) >= 0.0)


@given(case=_bounded_case())
def test_bounded_round_trip_stays_inside_physical_interval(
    case: tuple[float, float, np.ndarray],
) -> None:
    lower, upper, values = case
    transform = BoundedTargetTransform(lower_bound=lower, upper_bound=upper)

    transformed = transform.fit_transform(values)
    restored = transform.inverse_transform(transformed)

    np.testing.assert_allclose(restored, values, rtol=1.0e-10, atol=1.0e-9)
    assert np.isfinite(transformed).all()
    assert np.all(restored > lower)
    assert np.all(restored < upper)
    assert np.all(np.diff(transform.transform(np.sort(values))) >= 0.0)


@given(case=_paired_finite_arrays())
def test_physics_residual_round_trip_preserves_row_correspondence(
    case: tuple[np.ndarray, np.ndarray],
) -> None:
    values, baseline = case
    transform = PhysicsResidualTransform()

    residual = transform.fit_transform(values, baseline=baseline)
    restored = transform.inverse_transform(residual, baseline=baseline)

    np.testing.assert_allclose(restored, values, rtol=5.0e-15, atol=1.0e-12)
    np.testing.assert_allclose(residual, values - baseline, rtol=0.0, atol=1.0e-12)


@given(mean_std=_mean_std_arrays())
def test_log_prediction_uncertainty_is_finite_nonnegative_and_aligned(
    mean_std: tuple[np.ndarray, np.ndarray],
) -> None:
    mean, std = mean_std
    transform = LogTargetTransform(offset=0.25)
    prediction = GPyTorchPrediction(
        mean=mean,
        std=std,
        lower=mean - 1.96 * std,
        upper=mean + 1.96 * std,
    )

    restored = transform.inverse_prediction(prediction)

    assert restored.mean.shape == mean.shape
    assert restored.std is not None and restored.std.shape == std.shape
    assert restored.lower is not None and restored.lower.shape == mean.shape
    assert restored.upper is not None and restored.upper.shape == mean.shape
    assert np.isfinite(restored.mean).all()
    assert np.isfinite(restored.std).all()
    assert np.all(restored.std >= 0.0)
    assert np.all(restored.lower <= restored.upper)


@given(
    mean_std=_mean_std_arrays(),
    lower=st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
    width=st.floats(
        min_value=1.0e-3,
        max_value=1.0e3,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)
def test_bounded_prediction_moments_respect_support_and_variance_bound(
    mean_std: tuple[np.ndarray, np.ndarray],
    lower: float,
    width: float,
) -> None:
    mean, std = mean_std
    upper = lower + width
    transform = BoundedTargetTransform(
        lower_bound=lower,
        upper_bound=upper,
        n_quadrature_points=20,
    )

    restored = transform.inverse_prediction(GPyTorchPrediction(mean=mean, std=std))

    assert restored.std is not None
    assert restored.mean.shape == mean.shape
    assert restored.std.shape == std.shape
    assert np.isfinite(restored.mean).all()
    assert np.isfinite(restored.std).all()
    assert np.all(restored.mean >= lower)
    assert np.all(restored.mean <= upper)
    assert np.all(restored.std >= 0.0)
    assert np.all(restored.std <= width / 2.0 + 1.0e-10)


@given(
    values=_nonconstant_arrays(),
    include_std=st.booleans(),
    include_lower=st.booleans(),
    include_upper=st.booleans(),
)
def test_inverse_prediction_preserves_optional_field_presence_and_shapes(
    values: np.ndarray,
    include_std: bool,
    include_lower: bool,
    include_upper: bool,
) -> None:
    transform = StandardizedTargetTransform().fit(values)
    mean = transform.transform(values)
    prediction = GPyTorchPrediction(
        mean=mean,
        std=np.full(mean.shape, 0.25) if include_std else None,
        lower=mean - 0.5 if include_lower else None,
        upper=mean + 0.5 if include_upper else None,
    )

    restored = transform.inverse_prediction(prediction)

    assert restored.mean.shape == mean.shape
    assert (restored.std is not None) is include_std
    assert (restored.lower is not None) is include_lower
    assert (restored.upper is not None) is include_upper
    for array in (restored.std, restored.lower, restored.upper):
        if array is not None:
            assert array.shape == mean.shape
