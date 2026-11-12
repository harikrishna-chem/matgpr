"""Run the manual physics-informed GPR scientific-convergence pilot.

The synthetic material follows a linear thermal-expansion relation,

    a(T) = a_ref + slope * ((T - 550 K) / 100 K),

where the slope is learned in angstrom per 100 K and the reference lattice
parameter is fixed in angstrom. The GP kernel models any residual left by the
physics mean. This script is intentionally manual and is not an ordinary
pull-request test.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from matgpr import PhysicsInformedGPRRegressor

TRAINING_BUDGETS = (10, 25, 50, 100, 200, 400, 800, 1_600)
REFERENCE_TEMPERATURE_K = 550.0
TEMPERATURE_SCALE_K = 100.0
TRUE_SLOPE_ANGSTROM_PER_100K = 0.04
REFERENCE_LATTICE_ANGSTROM = 3.6
RANDOM_SEED = 2026


@dataclass(frozen=True)
class ConvergenceRecord:
    """Scientific and optimization evidence for one training budget."""

    training_iterations: int
    loss_points: int
    initial_loss: float
    final_loss: float
    minimum_loss: float
    learned_slope_angstrom_per_100k: float
    slope_absolute_error_angstrom_per_100k: float
    fixed_reference_lattice_angstrom: float
    extrapolation_rmse_angstrom: float
    extrapolation_mae_angstrom: float
    mean_latent_std_angstrom: float
    coverage_95: float
    elapsed_seconds: float


def thermal_expansion_equation(features, parameters):
    """Return the lattice parameter from the linear thermal physics mean."""

    return (
        parameters["slope_angstrom_per_100k"] * features["temperature_offset_100k"]
        + parameters["reference_lattice_angstrom"]
    )


def run_convergence_study() -> tuple[ConvergenceRecord, ...]:
    """Fit fixed-seed PI-GPR models across increasing optimizer budgets."""

    train_temperature_k = np.linspace(475.0, 625.0, 5)
    test_temperature_k = np.linspace(250.0, 850.0, 121)
    train_x = _temperature_offset(train_temperature_k).reshape(-1, 1)
    test_x = _temperature_offset(test_temperature_k).reshape(-1, 1)
    train_y = _true_lattice_parameter(train_temperature_k)
    test_y = _true_lattice_parameter(test_temperature_k)

    records = []
    for training_iterations in TRAINING_BUDGETS:
        started = time.perf_counter()
        estimator = PhysicsInformedGPRRegressor(
            equation=thermal_expansion_equation,
            feature_indices={"temperature_offset_100k": 0},
            learnable_parameters={"slope_angstrom_per_100k": 0.02},
            positive_parameters=("slope_angstrom_per_100k",),
            fixed_parameters={
                "reference_lattice_angstrom": REFERENCE_LATTICE_ANGSTROM,
            },
            kernel="rbf",
            ard=False,
            lr=0.03,
            training_iter=training_iterations,
            initial_noise=0.01,
            standardize_y=True,
            include_observation_noise=False,
            random_state=RANDOM_SEED,
        ).fit(train_x, train_y)
        elapsed_seconds = time.perf_counter() - started

        prediction = estimator.predict_distribution(
            test_x,
            confidence_level=0.95,
            include_observation_noise=False,
        )
        if prediction.std is None or prediction.lower is None or prediction.upper is None:
            raise AssertionError("PI-GPR convergence prediction must include uncertainty")

        learned = estimator.learned_physics_parameters_
        learned_slope = float(learned["slope_angstrom_per_100k"])
        prediction_error = prediction.mean - test_y
        coverage = np.mean((test_y >= prediction.lower) & (test_y <= prediction.upper))
        records.append(
            ConvergenceRecord(
                training_iterations=training_iterations,
                loss_points=len(estimator.loss_history_),
                initial_loss=float(estimator.loss_history_[0]),
                final_loss=float(estimator.loss_history_[-1]),
                minimum_loss=float(min(estimator.loss_history_)),
                learned_slope_angstrom_per_100k=learned_slope,
                slope_absolute_error_angstrom_per_100k=abs(
                    learned_slope - TRUE_SLOPE_ANGSTROM_PER_100K
                ),
                fixed_reference_lattice_angstrom=float(learned["reference_lattice_angstrom"]),
                extrapolation_rmse_angstrom=float(np.sqrt(np.mean(prediction_error**2))),
                extrapolation_mae_angstrom=float(np.mean(np.abs(prediction_error))),
                mean_latent_std_angstrom=float(np.mean(prediction.std)),
                coverage_95=float(coverage),
                elapsed_seconds=elapsed_seconds,
            )
        )
    return tuple(records)


def validate_convergence(records: tuple[ConvergenceRecord, ...]) -> None:
    """Enforce stable endpoint claims without requiring monotonic RMSE."""

    budgets = tuple(record.training_iterations for record in records)
    if budgets != TRAINING_BUDGETS:
        raise AssertionError(f"unexpected convergence budgets: {budgets}")
    if any(record.loss_points != record.training_iterations for record in records):
        raise AssertionError("each training iteration must contribute one loss value")

    numeric_values = np.asarray(
        [
            [
                record.initial_loss,
                record.final_loss,
                record.minimum_loss,
                record.learned_slope_angstrom_per_100k,
                record.slope_absolute_error_angstrom_per_100k,
                record.fixed_reference_lattice_angstrom,
                record.extrapolation_rmse_angstrom,
                record.extrapolation_mae_angstrom,
                record.mean_latent_std_angstrom,
                record.coverage_95,
                record.elapsed_seconds,
            ]
            for record in records
        ]
    )
    if not np.all(np.isfinite(numeric_values)):
        raise AssertionError("convergence records must contain only finite values")
    if not np.allclose(
        [record.initial_loss for record in records],
        records[0].initial_loss,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("fixed-seed runs must start from the same loss")
    if any(record.mean_latent_std_angstrom <= 0.0 for record in records):
        raise AssertionError("latent predictive uncertainty must stay positive")
    if any(not 0.0 <= record.coverage_95 <= 1.0 for record in records):
        raise AssertionError("coverage must remain in [0, 1]")
    if any(
        record.fixed_reference_lattice_angstrom != REFERENCE_LATTICE_ANGSTROM for record in records
    ):
        raise AssertionError("the fixed reference lattice parameter changed during fitting")

    first, final = records[0], records[-1]
    if final.final_loss >= first.final_loss - 2.5:
        raise AssertionError("the long-budget marginal loss did not improve materially")
    if final.slope_absolute_error_angstrom_per_100k >= 1e-5:
        raise AssertionError("the learned thermal slope did not converge to the known value")
    if final.extrapolation_rmse_angstrom >= 1e-5:
        raise AssertionError("the long-budget extrapolation RMSE is unexpectedly large")
    if final.extrapolation_rmse_angstrom >= 0.01 * first.extrapolation_rmse_angstrom:
        raise AssertionError("the long-budget RMSE did not improve by at least two orders")
    if final.mean_latent_std_angstrom >= first.mean_latent_std_angstrom:
        raise AssertionError("latent uncertainty did not contract with the longer budget")
    if final.coverage_95 < 0.95:
        raise AssertionError("the long-budget 95% interval does not cover the synthetic truth")


def _temperature_offset(temperature_k: np.ndarray) -> np.ndarray:
    return (temperature_k - REFERENCE_TEMPERATURE_K) / TEMPERATURE_SCALE_K


def _true_lattice_parameter(temperature_k: np.ndarray) -> np.ndarray:
    return REFERENCE_LATTICE_ANGSTROM + TRUE_SLOPE_ANGSTROM_PER_100K * _temperature_offset(
        temperature_k
    )


def main() -> None:
    """Run, validate, and print the convergence evidence as JSON."""

    records = run_convergence_study()
    validate_convergence(records)
    print(
        json.dumps(
            {
                "study": "physics_informed_gpr_thermal_expansion_convergence",
                "status": "passed",
                "training_temperature_range_k": [475.0, 625.0],
                "extrapolation_temperature_range_k": [250.0, 850.0],
                "learned_parameter": "slope_angstrom_per_100k",
                "fixed_parameter": "reference_lattice_angstrom",
                "records": [asdict(record) for record in records],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
