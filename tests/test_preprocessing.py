from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from matgpr.preprocessing import build_preprocessor, build_scaler, identify_feature_types


@pytest.mark.unit
class TestIdentifyFeatureTypes:
    def test_classifies_numeric_and_non_numeric_extension_dtypes(self) -> None:
        frame = pd.DataFrame(
            {
                "temperature": pd.Series([300.0, 325.0], dtype="Float64"),
                "cycles": pd.Series([1, 2], dtype="Int64"),
                "is_crystalline": pd.Series([True, False], dtype="boolean"),
                "phase": pd.Series(["alpha", "beta"], dtype="category"),
                "measured_at": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            }
        )

        numeric, categorical = identify_feature_types(frame)

        assert numeric == ["temperature", "cycles"]
        assert categorical == ["is_crystalline", "phase", "measured_at"]

    def test_requires_dataframe_input(self) -> None:
        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            identify_feature_types([[1.0, "alpha"]])  # type: ignore[arg-type]

    def test_rejects_duplicate_column_names(self) -> None:
        frame = pd.DataFrame([[1.0, 2.0]], columns=["feature", "feature"])

        with pytest.raises(ValueError, match="X columns must be unique"):
            identify_feature_types(frame)

    def test_requires_string_column_names(self) -> None:
        frame = pd.DataFrame([[1.0]], columns=[0])

        with pytest.raises(TypeError, match="X column names must be strings"):
            identify_feature_types(frame)


@pytest.mark.unit
class TestBuildScaler:
    @pytest.mark.parametrize(
        ("name", "expected_type"),
        [
            ("standard", StandardScaler),
            ("STANDARD", StandardScaler),
            ("minmax", MinMaxScaler),
            ("robust", RobustScaler),
        ],
    )
    def test_builds_named_scaler(self, name: str, expected_type: type) -> None:
        assert isinstance(build_scaler(name), expected_type)

    @pytest.mark.parametrize("name", ["none", "passthrough"])
    def test_builds_passthrough_scaler(self, name: str) -> None:
        assert build_scaler(name) == "passthrough"

    def test_rejects_unknown_scaler(self) -> None:
        with pytest.raises(
            ValueError,
            match="name must be one of: standard, minmax, robust, none, passthrough",
        ):
            build_scaler("unit")

    def test_requires_text_name(self) -> None:
        with pytest.raises(TypeError, match="name must be a string"):
            build_scaler(None)  # type: ignore[arg-type]


@pytest.mark.unit
class TestBuildPreprocessor:
    def test_accepts_index_generator_and_scalar_feature_specs(self) -> None:
        frame = pd.DataFrame(
            {
                "temperature": [300.0, np.nan, 350.0],
                "pressure": [1.0, 2.0, 3.0],
                "phase": ["alpha", "beta", np.nan],
            }
        )
        transformer = build_preprocessor(
            numeric_features=pd.Index(["temperature", "pressure"]),
            categorical_features=(name for name in ["phase"]),
        )

        transformed = transformer.fit_transform(frame)

        assert transformed.shape == (3, 4)
        assert np.isfinite(transformed).all()
        assert transformer.get_feature_names_out().tolist() == [
            "numeric__temperature",
            "numeric__pressure",
            "categorical__phase_alpha",
            "categorical__phase_beta",
        ]

        scalar_transformer = build_preprocessor(numeric_features="temperature")
        assert scalar_transformer.fit_transform(frame).shape == (3, 1)

    def test_rejects_overlapping_feature_groups(self) -> None:
        with pytest.raises(
            ValueError,
            match="numeric_features and categorical_features must not overlap: phase",
        ):
            build_preprocessor(
                numeric_features=["temperature", "phase"],
                categorical_features=["phase"],
            )

    def test_rejects_duplicate_feature_names_within_group(self) -> None:
        with pytest.raises(ValueError, match="numeric_features must contain unique names"):
            build_preprocessor(numeric_features=["temperature", "temperature"])

    @pytest.mark.parametrize(
        ("features", "message"),
        [
            (3, "must be a column name or iterable of column names"),
            (["temperature", 3], "must contain only string column names"),
        ],
    )
    def test_rejects_malformed_feature_specs(self, features: object, message: str) -> None:
        with pytest.raises(TypeError, match=message):
            build_preprocessor(numeric_features=features)  # type: ignore[arg-type]

    def test_requires_at_least_one_feature(self) -> None:
        with pytest.raises(ValueError, match="at least one numeric or categorical feature"):
            build_preprocessor()

    def test_ignores_unseen_categories_without_changing_output_shape(self) -> None:
        train = pd.DataFrame(
            {
                "temperature": [300.0, 325.0, 350.0],
                "phase": ["alpha", "beta", "alpha"],
            }
        )
        test = pd.DataFrame({"temperature": [375.0], "phase": ["gamma"]})
        transformer = build_preprocessor(
            numeric_features=["temperature"],
            categorical_features=["phase"],
        )

        train_transformed = transformer.fit_transform(train)
        test_transformed = transformer.transform(test)

        assert train_transformed.shape == (3, 3)
        assert test_transformed.shape == (1, 3)
        np.testing.assert_allclose(test_transformed[0, 1:], [0.0, 0.0])


@pytest.mark.component
def test_preprocessor_uses_training_statistics_without_test_leakage() -> None:
    train = pd.DataFrame({"temperature": [0.0, 2.0]})
    test = pd.DataFrame({"temperature": [100.0]})
    transformer = build_preprocessor(numeric_features=["temperature"])

    train_transformed = transformer.fit_transform(train)
    test_transformed = transformer.transform(test)

    np.testing.assert_allclose(train_transformed[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(test_transformed[:, 0], [99.0])
