from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.model_selection import GridSearchCV

from matgpr.sklearn_gpr import (
    build_sklearn_gpr_grid_search,
    build_sklearn_gpr_kernel,
    build_sklearn_gpr_model,
)


@pytest.mark.scientific
def test_fixed_rbf_gpr_matches_analytical_posterior() -> None:
    X_train = np.array([[0.0], [0.8], [1.7]])
    y_train = np.array([0.3, 1.1, -0.2])
    X_test = np.array([[0.4], [1.2]])
    constant_value = 1.7
    length_scale = 0.6
    noise_level = 0.04
    alpha = 0.01
    model = build_sklearn_gpr_model(
        kernel="rbf",
        alpha=alpha,
        normalize_y=False,
        n_restarts_optimizer=0,
        random_state=7,
        constant_value=constant_value,
        constant_value_bounds="fixed",
        length_scale=length_scale,
        length_scale_bounds="fixed",
        noise_level=noise_level,
        noise_level_bounds="fixed",
    )

    model.fit(X_train, y_train)
    predicted_mean, predicted_covariance = model.predict(X_test, return_cov=True)

    train_distance = X_train[:, None, 0] - X_train[None, :, 0]
    cross_distance = X_test[:, None, 0] - X_train[None, :, 0]
    test_distance = X_test[:, None, 0] - X_test[None, :, 0]
    train_covariance = constant_value * np.exp(-0.5 * train_distance**2 / length_scale**2)
    train_covariance += (noise_level + alpha) * np.eye(len(X_train))
    cross_covariance = constant_value * np.exp(-0.5 * cross_distance**2 / length_scale**2)
    test_covariance = constant_value * np.exp(-0.5 * test_distance**2 / length_scale**2)
    test_covariance += noise_level * np.eye(len(X_test))
    expected_mean = cross_covariance @ np.linalg.solve(train_covariance, y_train)
    expected_covariance = test_covariance - cross_covariance @ np.linalg.solve(
        train_covariance,
        cross_covariance.T,
    )

    np.testing.assert_allclose(predicted_mean, expected_mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        predicted_covariance,
        expected_covariance,
        rtol=1e-10,
        atol=1e-12,
    )
    assert model.n_features_in_ == 1
    assert model.X_train_.shape == X_train.shape
    assert model.alpha_.shape == y_train.shape
    assert model.L_.shape == (len(X_train), len(X_train))
    assert np.isfinite(model.log_marginal_likelihood_value_)


@pytest.mark.scientific
def test_normalized_gpr_handles_constant_target_with_finite_uncertainty() -> None:
    X = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    y = np.full(6, 5.0)
    model = build_sklearn_gpr_model(
        kernel="matern",
        normalize_y=True,
        n_restarts_optimizer=0,
        constant_value_bounds="fixed",
        length_scale_bounds="fixed",
        noise_level_bounds="fixed",
    )

    model.fit(X, y)
    mean, std = model.predict([[0.25], [0.75]], return_std=True)

    np.testing.assert_allclose(mean, 5.0, rtol=0.0, atol=1e-12)
    assert np.isfinite(std).all()
    assert (std >= 0.0).all()


@pytest.mark.unit
def test_model_preserves_per_observation_alpha_and_rejects_wrong_length() -> None:
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.arange(4, dtype=float)
    alpha = np.array([0.01, 0.02, 0.03, 0.04])
    model = build_sklearn_gpr_model(
        kernel="rbf",
        alpha=alpha,
        n_restarts_optimizer=0,
        constant_value_bounds="fixed",
        length_scale_bounds="fixed",
        noise_level_bounds="fixed",
    )

    model.fit(X, y)

    np.testing.assert_array_equal(model.alpha, alpha)
    wrong_alpha_model = build_sklearn_gpr_model(
        kernel="rbf",
        alpha=alpha[:-1],
        n_restarts_optimizer=0,
    )
    with pytest.raises(ValueError, match="alpha must be a scalar or an array with same number"):
        wrong_alpha_model.fit(X, y)


@pytest.mark.unit
def test_model_rejects_non_matrix_features() -> None:
    model = build_sklearn_gpr_model(kernel="rbf", n_restarts_optimizer=0)

    with pytest.raises(ValueError, match="Expected 2D array"):
        model.fit(np.arange(4, dtype=float), np.arange(4, dtype=float))


@pytest.mark.unit
def test_kernel_builders_cover_isotropic_ard_and_invalid_names() -> None:
    rbf = build_sklearn_gpr_kernel("RBF")
    matern = build_sklearn_gpr_kernel("matern-5-2")
    ard_matern = build_sklearn_gpr_kernel("ard_matern", n_features=3)

    assert isinstance(rbf.k1.k2, RBF)
    assert isinstance(matern.k1.k2, Matern)
    assert isinstance(ard_matern.k1.k2, Matern)
    np.testing.assert_array_equal(ard_matern.k1.k2.length_scale, np.ones(3))
    with pytest.raises(ValueError, match="name must be one of"):
        build_sklearn_gpr_kernel("periodic")


@pytest.mark.unit
def test_ard_kernel_rejects_boolean_feature_count() -> None:
    with pytest.raises(ValueError, match="n_features must be a positive integer"):
        build_sklearn_gpr_kernel("ard_rbf", n_features=True)


@pytest.mark.contract
def test_grid_search_builder_exposes_documented_search_space() -> None:
    search = build_sklearn_gpr_grid_search(
        n_features=2,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=13,
        n_jobs=2,
    )

    assert isinstance(search, GridSearchCV)
    assert isinstance(search.estimator, GaussianProcessRegressor)
    assert search.cv == 3
    assert search.scoring == "neg_mean_absolute_error"
    assert search.n_jobs == 2
    assert search.refit is True
    assert search.return_train_score is True
    assert len(search.param_grid["kernel"]) == 5
    assert search.param_grid["alpha"] == [1e-10, 1e-8, 1e-6]
    assert search.param_grid["n_restarts_optimizer"] == [0, 2]


@pytest.mark.unit
@pytest.mark.parametrize("n_features", [None, 0, -1, 1.5, True])
def test_grid_search_rejects_invalid_feature_count(n_features: object) -> None:
    with pytest.raises(ValueError):
        build_sklearn_gpr_grid_search(n_features=n_features)  # type: ignore[arg-type]
