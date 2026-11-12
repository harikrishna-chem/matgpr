from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from matgpr.validation import LearningCurveResult, learning_curve


def _nonlinear_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    x = np.linspace(-1.5, 1.5, 24)
    features = pd.DataFrame(
        {
            "x": x,
            "x_squared": x**2,
        },
        index=[f"sample-{index}" for index in range(x.size)],
    )
    target = pd.Series(
        0.4 + 1.2 * x - 0.7 * x**2 + 0.15 * np.sin(4.0 * x),
        index=features.index,
        name="target",
    )
    return features, target


def _run_learning_curve(seed: int) -> LearningCurveResult:
    features, target = _nonlinear_regression_data()
    return learning_curve(
        DecisionTreeRegressor(max_depth=3, splitter="random"),
        features,
        target,
        train_sizes=(50, 100),
        train_size_unit="percent",
        n_splits=2,
        test_size=6,
        random_state=seed,
        model_names="seed_aware_tree",
        metrics=("RMSE", "MAE"),
        metric_splits=("train", "test"),
        return_std=False,
        store_predictions=True,
    )


@pytest.mark.unit
def test_learning_curve_is_exactly_reproducible_for_same_seed() -> None:
    first = _run_learning_curve(17)
    second = _run_learning_curve(17)

    assert first.model_name == second.model_name
    assert first.metric_names == second.metric_names
    assert first.metric_splits == second.metric_splits
    pd.testing.assert_frame_equal(first.runs, second.runs, check_exact=True)
    pd.testing.assert_frame_equal(first.predictions, second.predictions, check_exact=True)
    pd.testing.assert_frame_equal(first.summary(), second.summary(), check_exact=True)


@pytest.mark.unit
def test_learning_curve_changes_sampled_rows_for_different_seeds() -> None:
    first = _run_learning_curve(17)
    second = _run_learning_curve(18)

    first_sampling = first.predictions[["repeat", "train_size_percent", "split", "sample_position"]]
    second_sampling = second.predictions[
        ["repeat", "train_size_percent", "split", "sample_position"]
    ]

    assert not first_sampling.equals(second_sampling)
    assert first.runs.columns.tolist() == second.runs.columns.tolist()
    assert first.predictions.columns.tolist() == second.predictions.columns.tolist()


@pytest.mark.unit
def test_learning_curve_contract_remains_finite_and_aligned_across_ten_seeds() -> None:
    features, target = _nonlinear_regression_data()
    expected_run_columns: list[str] | None = None
    expected_prediction_columns: list[str] | None = None
    sampling_signatures: set[tuple[int, ...]] = set()

    for seed in range(10):
        result = _run_learning_curve(seed)
        assert result.predictions is not None

        if expected_run_columns is None:
            expected_run_columns = result.runs.columns.tolist()
            expected_prediction_columns = result.predictions.columns.tolist()
        assert result.runs.columns.tolist() == expected_run_columns
        assert result.predictions.columns.tolist() == expected_prediction_columns

        numeric_runs = result.runs.select_dtypes(include=[np.number]).to_numpy()
        numeric_predictions = result.predictions.select_dtypes(include=[np.number]).to_numpy()
        assert np.isfinite(numeric_runs).all()
        assert np.isfinite(numeric_predictions).all()

        expected_true = target.iloc[result.predictions["sample_position"]].to_numpy()
        expected_labels = features.index[result.predictions["sample_position"]].to_numpy()
        np.testing.assert_allclose(result.predictions["y_true"], expected_true)
        np.testing.assert_array_equal(result.predictions["sample_label"], expected_labels)

        for row in result.runs.itertuples(index=False):
            matching = result.predictions[
                (result.predictions["repeat"] == row.repeat)
                & (result.predictions["train_size_percent"] == row.train_size_percent)
            ]
            split_counts = matching["split"].value_counts().to_dict()
            assert split_counts == {"train": row.n_train, "test": row.n_test}

        held_out = result.predictions[result.predictions["split"] == "test"]
        sampling_signatures.add(tuple(held_out["sample_position"].astype(int)))

    assert len(sampling_signatures) > 1
