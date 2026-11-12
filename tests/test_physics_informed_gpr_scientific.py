"""Scientific characterization for physics-informed exact GPR."""

from __future__ import annotations

import gpytorch
import numpy as np
import pytest
import torch

from matgpr import MatGPRRegressor, PhysicsInformedGPRRegressor, PhysicsInformedMean

pytestmark = [pytest.mark.scientific]


def _linear_equation(features, parameters):
    return parameters["slope"] * features["x"] + parameters["intercept"]


def _temperature_equation(features, parameters):
    return parameters["slope"] * features["temperature_k"] + parameters["intercept"]


def test_pi_gpr_recovers_learned_parameter_and_preserves_fixed_parameter() -> None:
    x = np.linspace(-2.0, 2.0, 16).reshape(-1, 1)
    y = 3.2 * x[:, 0] + 1.5
    estimator = PhysicsInformedGPRRegressor(
        equation=_linear_equation,
        feature_indices={"x": 0},
        learnable_parameters={"slope": 1.0},
        positive_parameters=("slope",),
        fixed_parameters={"intercept": 1.5},
        kernel="rbf",
        ard=False,
        lr=0.03,
        training_iter=150,
        initial_noise=0.01,
        random_state=17,
    )

    estimator.fit(x, y)

    parameters = estimator.learned_physics_parameters_
    assert parameters["slope"] == pytest.approx(3.2, abs=0.02)
    assert parameters["intercept"] == pytest.approx(1.5, abs=0.0)
    named_parameters = dict(estimator.mean_module_.named_parameters())
    assert "raw_slope" in named_parameters
    assert "raw_intercept" not in named_parameters


def test_pi_gpr_restores_physical_feature_units_and_target_units() -> None:
    temperature_k = np.linspace(300.0, 800.0, 16)
    feature_mean = float(np.mean(temperature_k))
    feature_std = float(np.std(temperature_k))
    x_scaled = ((temperature_k - feature_mean) / feature_std).reshape(-1, 1)
    y = 0.012 * temperature_k + 2.5
    estimator = PhysicsInformedGPRRegressor(
        equation=_temperature_equation,
        feature_indices={"temperature_k": 0},
        learnable_parameters={"slope": 0.005},
        positive_parameters=("slope",),
        fixed_parameters={"intercept": 2.5},
        feature_means={"temperature_k": feature_mean},
        feature_stds={"temperature_k": feature_std},
        kernel="rbf",
        ard=False,
        lr=0.03,
        training_iter=200,
        initial_noise=0.01,
        random_state=19,
    ).fit(x_scaled, y)

    assert estimator.learned_physics_parameters_["slope"] == pytest.approx(0.012, abs=2e-4)
    assert float(estimator.mean_module_.target_mean) == pytest.approx(float(np.mean(y)))
    assert float(estimator.mean_module_.target_std) == pytest.approx(float(np.std(y)))

    extrapolation_temperature = np.array([250.0, 900.0])
    extrapolation_x = ((extrapolation_temperature - feature_mean) / feature_std).reshape(-1, 1)
    expected = 0.012 * extrapolation_temperature + 2.5
    prediction = estimator.predict(extrapolation_x)
    np.testing.assert_allclose(prediction, expected, atol=0.01)

    standardized_physics_mean = estimator.mean_module_(
        torch.as_tensor(extrapolation_x, dtype=torch.float64)
    )
    expected_standardized = (expected - estimator.target_mean_) / estimator.target_std_
    np.testing.assert_allclose(
        standardized_physics_mean.detach().numpy(),
        expected_standardized,
        atol=2e-4,
    )


def test_correct_physics_mean_improves_low_data_extrapolation() -> None:
    x_train = np.array([[-0.5], [0.0], [0.5]])
    y_train = 2.4 * x_train[:, 0] + 0.7
    x_test = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y_test = 2.4 * x_test[:, 0] + 0.7
    common_options = {
        "kernel": "rbf",
        "ard": False,
        "lr": 0.03,
        "training_iter": 100,
        "initial_noise": 0.01,
        "standardize_y": True,
        "random_state": 23,
    }

    standard = MatGPRRegressor(**common_options).fit(x_train, y_train)
    physics_informed = PhysicsInformedGPRRegressor(
        equation=_linear_equation,
        feature_indices={"x": 0},
        learnable_parameters={"slope": 1.0, "intercept": 0.0},
        positive_parameters=("slope",),
        **common_options,
    ).fit(x_train, y_train)

    standard_rmse = float(np.sqrt(np.mean((standard.predict(x_test) - y_test) ** 2)))
    physics_rmse = float(np.sqrt(np.mean((physics_informed.predict(x_test) - y_test) ** 2)))

    # This bounded synthetic claim applies only because the supplied equation is
    # the true data-generating trend and every test point is extrapolative.
    assert standard_rmse > 1.0
    assert physics_rmse < 0.02
    assert physics_rmse < 0.01 * standard_rmse


@pytest.mark.parametrize(
    ("argument", "values"),
    [
        ("learnable_parameters", {"slope": np.nan}),
        ("fixed_parameters", {"intercept": np.inf}),
        ("feature_means", {"x": np.nan}),
        ("feature_stds", {"x": np.inf}),
    ],
)
def test_physics_mean_rejects_nonfinite_parameter_and_feature_metadata(
    argument: str,
    values: dict[str, float],
) -> None:
    options = {
        "equation": _linear_equation,
        "feature_indices": {"x": 0},
        "learnable_parameters": {"slope": 1.0},
        "fixed_parameters": {"intercept": 0.0},
        argument: values,
    }

    with pytest.raises(ValueError, match=f"{argument} values must be finite"):
        PhysicsInformedMean(**options)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("target_mean", np.nan, "target_mean must be finite"),
        ("target_std", np.nan, "target_std must be finite and positive"),
        ("target_std", np.inf, "target_std must be finite and positive"),
    ],
)
def test_physics_mean_rejects_nonfinite_target_scaling(
    argument: str,
    value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PhysicsInformedMean(
            equation=_linear_equation,
            feature_indices={"x": 0},
            learnable_parameters={"slope": 1.0, "intercept": 0.0},
            **{argument: value},
        )


@pytest.mark.parametrize(
    ("target_mean", "target_std", "message"),
    [
        (np.nan, 1.0, "mean must be finite"),
        (0.0, np.nan, "std must be finite and positive"),
        (0.0, np.inf, "std must be finite and positive"),
    ],
)
def test_target_standardization_setter_rejects_nonfinite_values(
    target_mean: float,
    target_std: float,
    message: str,
) -> None:
    mean = PhysicsInformedMean(
        equation=_linear_equation,
        feature_indices={"x": 0},
        learnable_parameters={"slope": 1.0, "intercept": 0.0},
    )

    with pytest.raises(ValueError, match=message):
        mean.set_target_standardization(target_mean, target_std)


@pytest.mark.parametrize("metadata_name", ["feature_means", "feature_stds"])
def test_physics_mean_rejects_unit_metadata_for_unknown_feature(metadata_name: str) -> None:
    with pytest.raises(ValueError, match=f"{metadata_name} contains unknown physics features"):
        PhysicsInformedMean(
            equation=_linear_equation,
            feature_indices={"x": 0},
            learnable_parameters={"slope": 1.0, "intercept": 0.0},
            **{metadata_name: {"temperature_typo": 1.0}},
        )


def test_physics_mean_accepts_numpy_integer_feature_index() -> None:
    mean = PhysicsInformedMean(
        equation=_linear_equation,
        feature_indices={"x": np.int64(0)},
        learnable_parameters={"slope": 1.0, "intercept": 0.0},
    )

    assert mean.feature_indices == {"x": 0}


def test_physics_mean_rejects_boolean_feature_index() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        PhysicsInformedMean(
            equation=_linear_equation,
            feature_indices={"x": True},
            learnable_parameters={"slope": 1.0, "intercept": 0.0},
        )


def test_physics_mean_rejects_nonfinite_equation_output() -> None:
    mean = PhysicsInformedMean(
        equation=lambda features, parameters: torch.full_like(features["x"], torch.nan),
        feature_indices={"x": 0},
    )

    with pytest.raises(ValueError, match="Physics equation must return only finite values"):
        mean(torch.zeros(3, 1, dtype=torch.float64))


def test_pi_gpr_uses_gaussian_likelihood_and_residual_kernel() -> None:
    x = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y = 1.2 * x[:, 0] + 0.4
    estimator = PhysicsInformedGPRRegressor(
        equation=_linear_equation,
        feature_indices={"x": 0},
        learnable_parameters={"slope": 1.0, "intercept": 0.0},
        kernel="rbf",
        training_iter=2,
        random_state=31,
    ).fit(x, y)

    assert isinstance(estimator.likelihood_, gpytorch.likelihoods.GaussianLikelihood)
    assert isinstance(estimator.model_.covar_module, gpytorch.kernels.ScaleKernel)
    assert estimator.model_.mean_module is estimator.mean_module_
