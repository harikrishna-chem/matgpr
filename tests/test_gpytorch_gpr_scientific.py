"""Scientific and failure-path characterization for exact GPyTorch GPR."""

from __future__ import annotations

from statistics import NormalDist

import gpytorch
import numpy as np
import pytest
import torch

from matgpr import gaussian_nlpd, uncertainty_diagnostics
from matgpr.gpytorch_gpr import ExactGPRModel, fit_gpytorch_gpr, predict_gpytorch_gpr

pytestmark = [pytest.mark.scientific]


def _fixed_hyperparameter_gp() -> tuple[ExactGPRModel, gpytorch.likelihoods.GaussianLikelihood]:
    train_x = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    train_y = torch.tensor([-0.5, 0.25, 1.0], dtype=torch.float64)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    likelihood.initialize(noise=0.15)
    model = ExactGPRModel(train_x, train_y, likelihood, kernel="rbf").double()
    model.mean_module.initialize(constant=0.3)
    model.covar_module.initialize(outputscale=1.4)
    model.covar_module.base_kernel.initialize(lengthscale=0.75)
    model.target_mean.fill_(12.0)
    model.target_std.fill_(2.5)
    return model, likelihood


def _rbf_covariance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    squared_distance = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return 1.4 * np.exp(-0.5 * squared_distance / 0.75**2)


def test_fixed_gpytorch_posterior_matches_matrix_algebra() -> None:
    model, likelihood = _fixed_hyperparameter_gp()
    train_x = np.array([[-1.0], [0.0], [1.0]])
    train_y = np.array([-0.5, 0.25, 1.0])
    test_x = np.array([[-0.5], [0.5]])

    prediction = predict_gpytorch_gpr(
        model,
        likelihood,
        test_x,
        include_observation_noise=False,
        return_prediction=True,
    )

    train_covariance = _rbf_covariance(train_x, train_x) + 0.15 * np.eye(3)
    cross_covariance = _rbf_covariance(test_x, train_x)
    test_covariance = _rbf_covariance(test_x, test_x)
    centered_target = train_y - 0.3
    expected_mean = 0.3 + cross_covariance @ np.linalg.solve(train_covariance, centered_target)
    expected_covariance = test_covariance - cross_covariance @ np.linalg.solve(
        train_covariance, cross_covariance.T
    )

    np.testing.assert_allclose(prediction.mean, expected_mean * 2.5 + 12.0, atol=1e-7)
    np.testing.assert_allclose(
        prediction.std,
        np.sqrt(np.diag(expected_covariance)) * 2.5,
        atol=1e-7,
    )


def test_observation_prediction_adds_likelihood_noise_in_target_units() -> None:
    model, likelihood = _fixed_hyperparameter_gp()
    test_x = np.array([[-0.5], [0.5]])

    latent = predict_gpytorch_gpr(
        model,
        likelihood,
        test_x,
        include_observation_noise=False,
        return_prediction=True,
    )
    observed = predict_gpytorch_gpr(
        model,
        likelihood,
        test_x,
        include_observation_noise=True,
        return_prediction=True,
    )

    np.testing.assert_allclose(observed.mean, latent.mean, atol=1e-12)
    np.testing.assert_allclose(
        observed.std**2 - latent.std**2,
        np.full(2, 0.15 * 2.5**2),
        atol=1e-7,
    )


def test_confidence_interval_uses_requested_normal_quantile() -> None:
    model, likelihood = _fixed_hyperparameter_gp()

    prediction = predict_gpytorch_gpr(
        model,
        likelihood,
        [[0.25]],
        confidence_level=0.90,
        include_observation_noise=False,
        return_prediction=True,
    )

    z_value = NormalDist().inv_cdf(0.95)
    np.testing.assert_allclose(prediction.lower, prediction.mean - z_value * prediction.std)
    np.testing.assert_allclose(prediction.upper, prediction.mean + z_value * prediction.std)
    assert np.all(prediction.lower < prediction.mean)
    assert np.all(prediction.mean < prediction.upper)


def test_mean_only_prediction_omits_uncertainty() -> None:
    model, likelihood = _fixed_hyperparameter_gp()

    prediction = predict_gpytorch_gpr(
        model,
        likelihood,
        [[0.25]],
        return_std=False,
        return_prediction=True,
    )

    assert prediction.mean.shape == (1,)
    assert prediction.std is None
    assert prediction.lower is None
    assert prediction.upper is None


def test_gpytorch_prediction_feeds_uncertainty_diagnostics() -> None:
    model, likelihood = _fixed_hyperparameter_gp()
    test_x = np.array([[-0.75], [0.0], [0.75]])
    y_true = np.array([10.9, 12.6, 14.0])

    prediction = predict_gpytorch_gpr(
        model,
        likelihood,
        test_x,
        include_observation_noise=True,
        return_prediction=True,
    )
    diagnostics = uncertainty_diagnostics(y_true, prediction.mean, prediction.std)

    assert np.all(np.isfinite(prediction.mean))
    assert np.all(np.isfinite(prediction.std))
    assert np.all(prediction.std > 0)
    assert np.isfinite(diagnostics["NLPD"])
    assert 0.0 <= diagnostics["observed_coverage"] <= 1.0


def test_constant_target_policy_is_explicit() -> None:
    x = np.arange(4.0).reshape(-1, 1)
    y = np.full(4, 3.5)

    with pytest.raises(ValueError, match="zero standard deviation"):
        fit_gpytorch_gpr(x, y, training_iter=1)

    result = fit_gpytorch_gpr(x, y, standardize_y=False, training_iter=1)
    prediction = result.predict([[1.5]])
    assert np.all(np.isfinite(prediction.mean))
    assert np.all(np.isfinite(prediction.std))


@pytest.mark.parametrize(
    ("option", "value", "error_type", "message"),
    [
        ("lr", np.nan, ValueError, "lr must be finite and positive"),
        ("lr", True, TypeError, "lr must be a real number"),
        ("training_iter", 1.5, TypeError, "training_iter must be an integer"),
        ("training_iter", True, TypeError, "training_iter must be an integer"),
        ("log_every", 1.5, TypeError, "log_every must be an integer"),
        ("initial_noise", np.inf, ValueError, "initial_noise must be finite and positive"),
    ],
)
def test_fit_rejects_invalid_numeric_options(
    option: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    x = np.arange(3.0).reshape(-1, 1)
    y = np.array([0.0, 0.5, 1.0])
    options = {"training_iter": 1, option: value}

    with pytest.raises(error_type, match=message):
        fit_gpytorch_gpr(x, y, **options)


def test_fit_accepts_single_column_target() -> None:
    x = np.arange(3.0).reshape(-1, 1)
    y = np.array([[0.0], [0.5], [1.0]])

    result = fit_gpytorch_gpr(x, y, training_iter=1)

    assert result.model.train_targets.shape == (3,)


def test_fit_rejects_multi_column_target_instead_of_flattening_it() -> None:
    x = np.arange(3.0).reshape(-1, 1)
    y = np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]])

    with pytest.raises(ValueError, match="y_train must be one-dimensional"):
        fit_gpytorch_gpr(x, y, training_iter=1)


def test_fit_rejects_feature_matrix_without_columns() -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        fit_gpytorch_gpr(np.empty((3, 0)), np.arange(3.0), training_iter=1)


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (np.array([0.0, 1.0]), "X must be a 2D feature matrix"),
        (np.zeros((2, 2)), "model was fitted with 1 features"),
        (np.array([[np.nan]]), "X must contain only finite values"),
        (np.empty((0, 1)), "X must contain at least one sample"),
    ],
)
def test_predict_rejects_invalid_feature_matrices(features: np.ndarray, message: str) -> None:
    model, likelihood = _fixed_hyperparameter_gp()

    with pytest.raises(ValueError, match=message):
        predict_gpytorch_gpr(model, likelihood, features)


@pytest.mark.parametrize("target_std", [0.0, -1.0, np.nan, np.inf])
def test_predict_rejects_invalid_target_standardization(target_std: float) -> None:
    model, likelihood = _fixed_hyperparameter_gp()
    model.target_std.fill_(target_std)

    with pytest.raises(ValueError, match="target_std must be finite and positive"):
        predict_gpytorch_gpr(model, likelihood, [[0.0]])


def test_predict_rejects_nonfinite_target_mean() -> None:
    model, likelihood = _fixed_hyperparameter_gp()
    model.target_mean.fill_(np.nan)

    with pytest.raises(ValueError, match="target_mean must be finite"):
        predict_gpytorch_gpr(model, likelihood, [[0.0]])


@pytest.mark.parametrize("min_std", [np.nan, np.inf])
def test_gaussian_nlpd_rejects_nonfinite_minimum_standard_deviation(min_std: float) -> None:
    with pytest.raises(ValueError, match="min_std must be finite"):
        gaussian_nlpd([0.0], [0.0], [1.0], min_std=min_std)
