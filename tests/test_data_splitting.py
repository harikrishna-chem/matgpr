from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from matgpr.data_splitting import separate_features_target, split_train_test


@pytest.mark.unit
class TestSeparateFeaturesTarget:
    def test_separates_target_without_mutating_input(self) -> None:
        frame = pd.DataFrame(
            {
                "temperature": [300.0, 325.0],
                "phase": ["alpha", "beta"],
                "conductivity": [1.2, 1.8],
            },
            index=pd.Index(["sample-a", "sample-b"], name="sample_id"),
        )
        original = frame.copy(deep=True)

        features, target = separate_features_target(frame, "conductivity")

        pd.testing.assert_frame_equal(features, frame[["temperature", "phase"]])
        pd.testing.assert_series_equal(target, frame["conductivity"])
        pd.testing.assert_frame_equal(frame, original)

    def test_rejects_missing_target(self) -> None:
        frame = pd.DataFrame({"temperature": [300.0]})

        with pytest.raises(
            KeyError,
            match="target_column 'conductivity' not found in dataframe",
        ):
            separate_features_target(frame, "conductivity")

    def test_requires_dataframe_input(self) -> None:
        with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
            separate_features_target({"target": [1.0]}, "target")  # type: ignore[arg-type]

    def test_rejects_duplicate_column_names(self) -> None:
        frame = pd.DataFrame(
            [[300.0, 1.2, 1.3]],
            columns=["temperature", "conductivity", "conductivity"],
        )

        with pytest.raises(ValueError, match="df columns must be unique"):
            separate_features_target(frame, "conductivity")


@pytest.mark.unit
class TestSplitTrainTest:
    def test_is_deterministic_and_preserves_row_correspondence(self) -> None:
        features = pd.DataFrame(
            {"feature": np.arange(20, dtype=float)},
            index=[f"sample-{index}" for index in range(20)],
        )
        target = pd.Series(
            features["feature"].to_numpy() * 10.0,
            index=features.index,
            name="target",
        )

        first = split_train_test(features, target, test_size=0.25, random_state=7)
        second = split_train_test(features, target, test_size=0.25, random_state=7)

        assert isinstance(first, list)
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])
        pd.testing.assert_series_equal(first[2], second[2])
        pd.testing.assert_series_equal(first[3], second[3])

        X_train, X_test, y_train, y_test = first
        assert len(X_train) == len(y_train) == 15
        assert len(X_test) == len(y_test) == 5
        assert set(X_train.index).isdisjoint(X_test.index)
        assert set(X_train.index) | set(X_test.index) == set(features.index)
        np.testing.assert_allclose(y_train, X_train["feature"] * 10.0)
        np.testing.assert_allclose(y_test, X_test["feature"] * 10.0)

    def test_supports_ordered_split_without_shuffling(self) -> None:
        features = pd.DataFrame({"feature": np.arange(10)})
        target = pd.Series(np.arange(10), name="target")

        X_train, X_test, y_train, y_test = split_train_test(
            features,
            target,
            test_size=0.2,
            shuffle=False,
        )

        assert X_train.index.tolist() == list(range(8))
        assert X_test.index.tolist() == [8, 9]
        assert y_train.index.tolist() == list(range(8))
        assert y_test.index.tolist() == [8, 9]

    def test_supports_stratified_classification_split(self) -> None:
        features = pd.DataFrame({"feature": np.arange(20)})
        target = pd.Series(["alpha"] * 10 + ["beta"] * 10, name="phase")

        _, _, y_train, y_test = split_train_test(
            features,
            target,
            test_size=0.2,
            random_state=11,
            stratify=target,
        )

        assert y_train.value_counts().to_dict() == {"alpha": 8, "beta": 8}
        assert y_test.value_counts().to_dict() == {"alpha": 2, "beta": 2}

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            split_train_test(np.ones((3, 1)), np.ones(2))

    @pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 3])
    def test_rejects_invalid_test_size(self, test_size: int | float) -> None:
        with pytest.raises(ValueError):
            split_train_test(np.ones((3, 1)), np.ones(3), test_size=test_size)
