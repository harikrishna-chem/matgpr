from __future__ import annotations

import numpy as np
import pytest

from matgpr import (
    BoundedTargetTransform,
    IdentityTargetTransform,
    LogTargetTransform,
    PhysicsResidualTransform,
    StandardizedTargetTransform,
)
from matgpr.gpytorch_gpr import GPyTorchPrediction


@pytest.mark.scientific
def test_standardization_round_trip_and_uncertainty_use_training_scale() -> None:
    y = np.array([-3.0, 1.0, 5.0, 9.0])
    transform = StandardizedTargetTransform()

    transformed = transform.fit_transform(y)
    restored = transform.inverse_transform(transformed)
    restored_std = transform.inverse_std([0.0, 1.0], [0.25, 0.5])

    np.testing.assert_allclose(np.mean(transformed), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.std(transformed, ddof=0), 1.0, atol=1e-15)
    np.testing.assert_allclose(restored, y, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(restored_std, np.std(y, ddof=0) * np.array([0.25, 0.5]))
    assert transform.mean_ == pytest.approx(np.mean(y))
    assert transform.scale_ == pytest.approx(np.std(y, ddof=0))


@pytest.mark.scientific
def test_log_transform_matches_lognormal_moments_and_quantiles() -> None:
    transform = LogTargetTransform(offset=0.5)
    transformed_mean = np.array([np.log(2.0), np.log(5.0)])
    transformed_std = np.array([0.15, 0.35])
    prediction = GPyTorchPrediction(
        mean=transformed_mean,
        std=transformed_std,
        lower=transformed_mean - 1.96 * transformed_std,
        upper=transformed_mean + 1.96 * transformed_std,
    )

    inverted = transform.inverse_prediction(prediction)

    variance = transformed_std**2
    expected_mean = np.exp(transformed_mean + 0.5 * variance) - transform.offset
    expected_std = np.sqrt(np.expm1(variance) * np.exp(2.0 * transformed_mean + variance))
    np.testing.assert_allclose(inverted.mean, expected_mean)
    np.testing.assert_allclose(inverted.std, expected_std)
    np.testing.assert_allclose(
        inverted.lower,
        np.exp(prediction.lower) - transform.offset,
    )
    np.testing.assert_allclose(
        inverted.upper,
        np.exp(prediction.upper) - transform.offset,
    )
    assert (inverted.lower < inverted.mean).all()
    assert (inverted.mean < inverted.upper).all()


@pytest.mark.scientific
def test_bounded_transform_zero_noise_and_symmetry_are_physically_bounded() -> None:
    transform = BoundedTargetTransform(
        lower_bound=10.0,
        upper_bound=30.0,
        n_quadrature_points=40,
    )

    zero_noise = transform.inverse_prediction(
        GPyTorchPrediction(mean=np.array([0.0]), std=np.array([0.0]))
    )
    symmetric = transform.inverse_prediction(
        GPyTorchPrediction(mean=np.array([0.0]), std=np.array([1.2]))
    )

    np.testing.assert_allclose(zero_noise.mean, [20.0], atol=1e-13)
    np.testing.assert_allclose(zero_noise.std, [0.0], atol=1e-12)
    np.testing.assert_allclose(symmetric.mean, [20.0], atol=1e-12)
    assert 0.0 < symmetric.std[0] < 10.0


@pytest.mark.scientific
def test_bounded_transform_small_noise_matches_delta_method_limit() -> None:
    transform = BoundedTargetTransform(
        lower_bound=10.0,
        upper_bound=30.0,
        n_quadrature_points=40,
    )
    transformed_mean = np.array([0.7])
    transformed_std = np.array([1.0e-4])

    prediction = transform.inverse_prediction(
        GPyTorchPrediction(mean=transformed_mean, std=transformed_std)
    )

    probability = 1.0 / (1.0 + np.exp(-transformed_mean))
    expected_mean = transform.lower_bound + 20.0 * probability
    expected_std = 20.0 * probability * (1.0 - probability) * transformed_std
    np.testing.assert_allclose(prediction.mean, expected_mean, rtol=1.0e-8, atol=1.0e-12)
    np.testing.assert_allclose(prediction.std, expected_std, rtol=1.0e-7, atol=1.0e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "transform",
    [
        IdentityTargetTransform(),
        StandardizedTargetTransform().fit([1.0, 3.0]),
        PhysicsResidualTransform(),
    ],
)
def test_inverse_std_rejects_mismatched_mean_and_std_lengths(transform: object) -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        transform.inverse_std([0.0, 1.0], [0.2])  # type: ignore[attr-defined]


@pytest.mark.unit
@pytest.mark.parametrize("offset", [np.nan, np.inf, -np.inf])
def test_log_transform_rejects_non_finite_offset(offset: float) -> None:
    with pytest.raises(ValueError, match="offset must be finite"):
        LogTargetTransform(offset=offset)


@pytest.mark.unit
def test_transform_failure_paths_are_explicit() -> None:
    with pytest.raises(ValueError, match="must be fitted before use"):
        StandardizedTargetTransform().transform([1.0])
    with pytest.raises(ValueError, match="constant target"):
        StandardizedTargetTransform().fit([2.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        IdentityTargetTransform().inverse_std([0.0], [-0.1])
    with pytest.raises(ValueError, match="strictly between"):
        BoundedTargetTransform(0.0, 1.0).transform([0.0, 0.5])


@pytest.mark.unit
def test_inverse_prediction_without_uncertainty_preserves_optional_fields() -> None:
    prediction = GPyTorchPrediction(mean=np.array([np.log(3.0)]))

    inverted = LogTargetTransform(offset=1.0).inverse_prediction(prediction)

    np.testing.assert_allclose(inverted.mean, [2.0])
    assert inverted.std is None
    assert inverted.lower is None
    assert inverted.upper is None
