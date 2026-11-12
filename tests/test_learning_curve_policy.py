from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from matgpr import validation

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("percent", "percent"),
        ("percentage", "percent"),
        ("%", "percent"),
        ("fraction", "fraction"),
        ("proportion", "fraction"),
        ("count", "count"),
        ("counts", "count"),
        ("n", "count"),
        (" Percentage ", "percent"),
    ],
)
def test_train_size_unit_aliases(value: str, expected: str) -> None:
    assert validation._normalize_train_size_unit(value) == expected


def test_train_size_unit_rejects_unknown_value() -> None:
    with pytest.raises(
        ValueError,
        match="train_size_unit must be one of: percent, fraction, count",
    ):
        validation._normalize_train_size_unit("samples")


@pytest.mark.parametrize(
    ("start", "stop", "step", "expected"),
    [
        (10, 30, 10, (10.0, 20.0, 30.0)),
        (0.1, 0.3, 0.1, (0.1, 0.2, 0.3)),
        (10, 15, 20, (10.0,)),
        (25, 25, 5, (25.0,)),
    ],
)
def test_inclusive_train_size_range(
    start: float,
    stop: float,
    step: float,
    expected: tuple[float, ...],
) -> None:
    assert validation._inclusive_train_size_range(start=start, stop=stop, step=step) == expected


@pytest.mark.parametrize("step", [0, -1])
def test_inclusive_train_size_range_requires_positive_step(step: float) -> None:
    with pytest.raises(ValueError, match="train_size_step must be positive"):
        validation._inclusive_train_size_range(start=10, stop=30, step=step)


def test_inclusive_train_size_range_requires_ordered_bounds() -> None:
    with pytest.raises(
        ValueError,
        match="train_size_stop must be greater than or equal to train_size_start",
    ):
        validation._inclusive_train_size_range(start=30, stop=10, step=10)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("start", np.nan),
        ("stop", np.inf),
        ("step", np.nan),
        ("step", np.inf),
    ],
)
def test_inclusive_train_size_range_requires_finite_values(
    argument: str,
    value: float,
) -> None:
    options = {"start": 10.0, "stop": 30.0, "step": 10.0, argument: value}

    with pytest.raises(ValueError, match=f"train_size_{argument} must be finite"):
        validation._inclusive_train_size_range(**options)


def test_percent_train_sizes_resolve_to_model_fractions() -> None:
    assert validation._resolve_learning_curve_train_sizes(
        (10, 100),
        start=0,
        stop=0,
        step=1,
        unit="percentage",
    ) == (
        {
            "requested_train_size": 10.0,
            "requested_train_size_unit": "percent",
            "model_train_size": 0.1,
        },
        {
            "requested_train_size": 100.0,
            "requested_train_size_unit": "percent",
            "model_train_size": 1.0,
        },
    )


def test_fraction_train_sizes_remain_fractions() -> None:
    assert validation._resolve_learning_curve_train_sizes(
        (0.25, 1),
        start=0,
        stop=0,
        step=1,
        unit="fraction",
    ) == (
        {
            "requested_train_size": 0.25,
            "requested_train_size_unit": "fraction",
            "model_train_size": 0.25,
        },
        {
            "requested_train_size": 1.0,
            "requested_train_size_unit": "fraction",
            "model_train_size": 1.0,
        },
    )


def test_count_train_sizes_are_normalized_to_integers() -> None:
    assert validation._resolve_learning_curve_train_sizes(
        (2, 3.0),
        start=0,
        stop=0,
        step=1,
        unit="count",
    ) == (
        {
            "requested_train_size": 2,
            "requested_train_size_unit": "count",
            "model_train_size": 2,
        },
        {
            "requested_train_size": 3,
            "requested_train_size_unit": "count",
            "model_train_size": 3,
        },
    )


def test_generated_train_sizes_use_inclusive_range() -> None:
    resolved = validation._resolve_learning_curve_train_sizes(
        None,
        start=10,
        stop=30,
        step=10,
        unit="percent",
    )
    assert tuple(item["requested_train_size"] for item in resolved) == (
        10.0,
        20.0,
        30.0,
    )
    assert tuple(item["model_train_size"] for item in resolved) == (0.1, 0.2, 0.3)


def test_explicit_train_sizes_preserve_order_and_duplicates() -> None:
    resolved = validation._resolve_learning_curve_train_sizes(
        (50, 25, 50),
        start=0,
        stop=0,
        step=1,
        unit="percent",
    )
    assert tuple(item["requested_train_size"] for item in resolved) == (
        50.0,
        25.0,
        50.0,
    )


def test_train_size_resolution_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="train_sizes must contain at least one value"):
        validation._resolve_learning_curve_train_sizes(
            (),
            start=10,
            stop=100,
            step=10,
            unit="percent",
        )


@pytest.mark.parametrize("value", [0, 101])
def test_percent_train_sizes_must_be_in_range(value: float) -> None:
    with pytest.raises(ValueError, match=r"Percent train sizes.*\(0, 100\]"):
        validation._resolve_learning_curve_train_sizes(
            (value,),
            start=0,
            stop=0,
            step=1,
            unit="percent",
        )


@pytest.mark.parametrize("value", [0, 1.1])
def test_fraction_train_sizes_must_be_in_range(value: float) -> None:
    with pytest.raises(ValueError, match=r"Fraction train sizes.*\(0, 1\]"):
        validation._resolve_learning_curve_train_sizes(
            (value,),
            start=0,
            stop=0,
            step=1,
            unit="fraction",
        )


def test_count_train_sizes_must_be_integral() -> None:
    with pytest.raises(ValueError, match="Count train sizes must be integers"):
        validation._resolve_learning_curve_train_sizes(
            (2.5,),
            start=0,
            stop=0,
            step=1,
            unit="count",
        )


def test_count_train_sizes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="Count train sizes must be positive"):
        validation._resolve_learning_curve_train_sizes(
            (0,),
            start=0,
            stop=0,
            step=1,
            unit="count",
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("rmse", "RMSE"),
        ("root_mean_squared_error", "RMSE"),
        ("r2", "R2"),
        ("r^2", "R2"),
        ("mae", "MAE"),
        ("mean_absolute_error", "MAE"),
        ("r", "r"),
        ("pearson", "r"),
        ("pearson_r", "r"),
        (" RMSE ", "RMSE"),
    ],
)
def test_metric_name_aliases(value: str, expected: str) -> None:
    assert validation._normalize_metric_name(value) == expected


def test_metric_name_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="metrics can include only: RMSE, R2, MAE, r"):
        validation._normalize_metric_name("mape")


def test_metric_names_preserve_first_canonical_order_and_remove_duplicates() -> None:
    assert validation._normalize_metric_names(
        ("mae", "RMSE", "mean_absolute_error", "pearson", "r")
    ) == ("MAE", "RMSE", "r")


def test_metric_names_accept_single_string() -> None:
    assert validation._normalize_metric_names("r2") == ("R2",)


def test_metric_names_reject_empty_sequence() -> None:
    with pytest.raises(ValueError, match="metrics must contain at least one metric name"):
        validation._normalize_metric_names(())


@pytest.mark.parametrize("value", ["both", "train-test", "train_test", "TRAIN_TEST"])
def test_metric_split_combined_aliases(value: str) -> None:
    assert validation._normalize_metric_splits(value) == ("train", "test")


def test_metric_split_combined_alias_ignores_outer_whitespace() -> None:
    assert validation._normalize_metric_splits(" BOTH ") == ("train", "test")


def test_metric_splits_preserve_first_order_and_remove_duplicates() -> None:
    assert validation._normalize_metric_splits((" Test ", "train", "test")) == (
        "test",
        "train",
    )


def test_metric_splits_reject_empty_sequence() -> None:
    with pytest.raises(
        ValueError,
        match="metric_splits must contain at least one split",
    ):
        validation._normalize_metric_splits(())


def test_metric_splits_reject_unknown_value() -> None:
    with pytest.raises(
        ValueError,
        match="metric_splits can include only 'train' and 'test'",
    ):
        validation._normalize_metric_splits("validation")


def test_metric_column_name_normalizes_split_and_metric() -> None:
    assert validation._metric_column_name(" Test ", "root_mean_squared_error") == ("test_RMSE")


def test_metric_columns_follow_split_then_metric_order() -> None:
    frame = pd.DataFrame(
        {
            "test_RMSE": [1.0],
            "test_R2": [0.5],
            "train_RMSE": [0.5],
            "train_R2": [0.8],
        }
    )
    assert validation._metric_columns(
        frame,
        metrics=("rmse", "r2"),
        splits=("test", "train"),
    ) == ("test_RMSE", "test_R2", "train_RMSE", "train_R2")


def test_metric_columns_report_all_missing_columns() -> None:
    frame = pd.DataFrame({"test_RMSE": [1.0]})
    with pytest.raises(
        KeyError,
        match=r"Metric columns are not present.*'test_R2'.*'train_RMSE'.*'train_R2'",
    ):
        validation._metric_columns(
            frame,
            metrics=("rmse", "r2"),
            splits=("test", "train"),
        )


def test_validation_uses_learning_curve_policy_helpers_directly() -> None:
    policy = importlib.import_module("matgpr._learning_curve_policy")
    helper_names = (
        "_resolve_learning_curve_train_sizes",
        "_inclusive_train_size_range",
        "_normalize_train_size_unit",
        "_normalize_metric_names",
        "_normalize_metric_name",
        "_normalize_metric_splits",
        "_metric_column_name",
        "_metric_columns",
    )
    for helper_name in helper_names:
        assert getattr(validation, helper_name) is getattr(policy, helper_name)
