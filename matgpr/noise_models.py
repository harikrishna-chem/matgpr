from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "FeatureNoiseModel",
    "ObservationNoiseProfile",
    "ReplicateNoiseModel",
    "SourceNoiseModel",
    "combine_noise_profiles",
    "constant_noise_profile",
]


@dataclass(frozen=True)
class ObservationNoiseProfile:
    """Per-observation noise standard deviations for GPR workflows.

    Gaussian Process Regression libraries commonly accept observation variance
    through an ``alpha`` vector. This profile stores standard deviations because
    they are easier to reason about in physical target units, then exposes the
    corresponding variances through :attr:`alpha`.
    """

    noise_std: Sequence[float] | np.ndarray
    labels: Sequence[str] | np.ndarray | None = None
    component_names: Sequence[str] | np.ndarray | None = None

    def __post_init__(self) -> None:
        noise = _to_1d_finite(self.noise_std, "noise_std")
        if np.any(noise < 0):
            raise ValueError("noise_std must be non-negative")
        labels = _label_array(self.labels, noise.shape[0], default="observation_noise")
        components = _label_array(
            self.component_names,
            noise.shape[0],
            default="noise",
        )

        object.__setattr__(self, "noise_std", noise)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "component_names", components)

    @property
    def n_observations(self) -> int:
        """Number of observations in the profile."""
        return int(self.noise_std.shape[0])

    @property
    def alpha(self) -> np.ndarray:
        """Observation variances suitable for scikit-learn GPR ``alpha``."""
        return self.noise_std**2

    @property
    def variance(self) -> np.ndarray:
        """Alias for observation variance."""
        return self.alpha

    def to_frame(self) -> pd.DataFrame:
        """Return a compact dataframe for reports and debugging."""
        return pd.DataFrame(
            {
                "noise_std": self.noise_std,
                "variance": self.variance,
                "label": self.labels,
                "component": self.component_names,
            }
        )

    def summary(self) -> pd.DataFrame:
        """Summarize noise levels by label."""
        frame = self.to_frame()
        return (
            frame.groupby("label", dropna=False)["noise_std"]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
        )


@dataclass(frozen=True)
class SourceNoiseModel:
    """Assign observation noise by data source, instrument, method, or paper.

    Use this when rows come from sources with different expected reliability,
    such as experimental versus simulated data, different characterization
    techniques, or multiple literature papers.
    """

    source_noise_std: Mapping[object, float]
    default_noise_std: float | None = None
    unknown: str = "error"
    label_prefix: str = "source"

    def profile(self, sources) -> ObservationNoiseProfile:
        """Return per-row noise for source labels."""
        source_values = _to_1d_object(sources, "sources")
        normalized_unknown = self.unknown.lower()
        if normalized_unknown not in {"error", "default"}:
            raise ValueError("unknown must be 'error' or 'default'")
        if normalized_unknown == "default" and self.default_noise_std is None:
            raise ValueError("default_noise_std is required when unknown='default'")

        source_noise = {
            key: _nonnegative_scalar(value, f"source_noise_std[{key!r}]")
            for key, value in self.source_noise_std.items()
        }
        default_noise = None
        if self.default_noise_std is not None:
            default_noise = _nonnegative_scalar(self.default_noise_std, "default_noise_std")

        noise_values = []
        labels = []
        for source in source_values:
            if source in source_noise:
                noise = source_noise[source]
            elif normalized_unknown == "default":
                noise = default_noise
            else:
                raise ValueError(f"No noise_std was provided for source {source!r}")
            noise_values.append(noise)
            labels.append(f"{self.label_prefix}:{source}")

        return ObservationNoiseProfile(
            noise_std=np.asarray(noise_values, dtype=float),
            labels=np.asarray(labels, dtype=object),
            component_names=np.full(source_values.shape[0], "source", dtype=object),
        )


@dataclass
class ReplicateNoiseModel:
    """Estimate observation noise from replicate target measurements.

    Rows sharing the same group label are treated as replicate measurements.
    Groups with at least two observations receive their sample standard
    deviation. Singleton groups receive ``fallback_noise_std`` when supplied,
    otherwise the pooled replicate standard deviation, and finally
    ``min_noise_std`` if no replicate information is available.
    """

    min_noise_std: float = 1e-8
    fallback_noise_std: float | None = None
    ddof: int = 1
    label_prefix: str = "replicate"

    def __post_init__(self) -> None:
        self.min_noise_std = _nonnegative_scalar(self.min_noise_std, "min_noise_std")
        if self.fallback_noise_std is not None:
            self.fallback_noise_std = _nonnegative_scalar(
                self.fallback_noise_std,
                "fallback_noise_std",
            )
        if self.ddof < 0:
            raise ValueError("ddof must be non-negative")

    def fit(self, y, groups):
        """Estimate group-level noise from targets and replicate labels."""
        y_values = _to_1d_finite(y, "y")
        group_values = _to_1d_object(groups, "groups")
        _validate_same_length(y_values, group_values, "y", "groups")

        frame = pd.DataFrame({"y": y_values, "group": group_values})
        group_noise: dict[object, float] = {}
        replicate_variances = []

        for group, group_frame in frame.groupby("group", sort=False, dropna=False):
            values = group_frame["y"].to_numpy(dtype=float)
            if values.shape[0] > self.ddof:
                noise = float(np.std(values, ddof=self.ddof))
                replicate_variances.append(noise**2)
            else:
                noise = np.nan
            group_noise[group] = noise

        if self.fallback_noise_std is not None:
            fallback = self.fallback_noise_std
        elif replicate_variances:
            fallback = float(np.sqrt(np.mean(replicate_variances)))
        else:
            fallback = self.min_noise_std

        self.group_noise_std_ = {
            group: max(noise if np.isfinite(noise) else fallback, self.min_noise_std)
            for group, noise in group_noise.items()
        }
        self.fallback_noise_std_ = max(fallback, self.min_noise_std)
        return self

    def profile(self, groups) -> ObservationNoiseProfile:
        """Return per-row noise for fitted replicate labels."""
        self._require_fitted()
        group_values = _to_1d_object(groups, "groups")
        noise_values = []
        labels = []
        for group in group_values:
            noise_values.append(self.group_noise_std_.get(group, self.fallback_noise_std_))
            labels.append(f"{self.label_prefix}:{group}")
        return ObservationNoiseProfile(
            noise_std=np.asarray(noise_values, dtype=float),
            labels=np.asarray(labels, dtype=object),
            component_names=np.full(group_values.shape[0], "replicate", dtype=object),
        )

    def fit_profile(self, y, groups) -> ObservationNoiseProfile:
        """Fit the model and return the training-row noise profile."""
        self.fit(y, groups)
        return self.profile(groups)

    def _require_fitted(self) -> None:
        if not hasattr(self, "group_noise_std_"):
            raise ValueError("ReplicateNoiseModel must be fitted before calling profile")


@dataclass(frozen=True)
class FeatureNoiseModel:
    """Compute heteroscedastic noise from feature values.

    The supplied callable receives a numeric feature matrix and returns one
    noise standard deviation per row. This lets users encode physics-aware
    assumptions, such as increasing measurement uncertainty at high
    temperature, low signal intensity, extreme compositions, or high load.
    """

    noise_std_function: Callable[[np.ndarray], Sequence[float] | np.ndarray]
    label: str = "feature_noise"

    def profile(self, X) -> ObservationNoiseProfile:
        """Return per-row noise from a feature-dependent equation."""
        x_values = _to_2d_finite(_as_numeric_matrix(X), "X")
        noise = _to_1d_finite(self.noise_std_function(x_values), "noise_std_function(X)")
        if noise.shape[0] != x_values.shape[0]:
            raise ValueError("noise_std_function must return one value per feature row")
        if np.any(noise < 0):
            raise ValueError("noise_std_function must return non-negative values")
        return ObservationNoiseProfile(
            noise_std=noise,
            labels=np.full(x_values.shape[0], self.label, dtype=object),
            component_names=np.full(x_values.shape[0], "feature", dtype=object),
        )


def constant_noise_profile(
    n_observations: int,
    noise_std: float,
    *,
    label: str = "constant",
) -> ObservationNoiseProfile:
    """Return a constant noise profile."""
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    noise = np.full(int(n_observations), _nonnegative_scalar(noise_std, "noise_std"), dtype=float)
    labels = np.full(int(n_observations), label, dtype=object)
    return ObservationNoiseProfile(
        noise_std=noise,
        labels=labels,
        component_names=np.full(int(n_observations), "constant", dtype=object),
    )


def combine_noise_profiles(
    *profiles: ObservationNoiseProfile,
    mode: str = "quadrature",
    label: str = "combined",
) -> ObservationNoiseProfile:
    """Combine independent noise profiles.

    ``mode="quadrature"`` combines independent variance components as
    ``sqrt(sum(sigma_i^2))``. ``mode="max"`` keeps the largest standard
    deviation per row, and ``mode="sum"`` adds standard deviations directly.
    """
    if len(profiles) == 1 and isinstance(profiles[0], (list, tuple)):
        profiles = tuple(profiles[0])
    if not profiles:
        raise ValueError("At least one ObservationNoiseProfile is required")
    if not all(isinstance(profile, ObservationNoiseProfile) for profile in profiles):
        raise TypeError("profiles must contain only ObservationNoiseProfile objects")

    n_observations = profiles[0].n_observations
    for profile in profiles:
        if profile.n_observations != n_observations:
            raise ValueError("All noise profiles must have the same number of observations")

    noise_matrix = np.vstack([profile.noise_std for profile in profiles])
    normalized_mode = mode.lower()
    if normalized_mode == "quadrature":
        noise = np.sqrt(np.sum(noise_matrix * noise_matrix, axis=0))
    elif normalized_mode == "max":
        noise = np.max(noise_matrix, axis=0)
    elif normalized_mode == "sum":
        noise = np.sum(noise_matrix, axis=0)
    else:
        raise ValueError("mode must be one of: quadrature, max, sum")

    component_labels = np.asarray(
        ["+".join(str(profile.component_names[row]) for profile in profiles) for row in range(n_observations)],
        dtype=object,
    )
    return ObservationNoiseProfile(
        noise_std=noise,
        labels=np.full(n_observations, label, dtype=object),
        component_names=component_labels,
    )


def _as_numeric_matrix(values) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def _to_2d_finite(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one feature")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _to_1d_finite(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _to_1d_object(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return array


def _label_array(labels, n_observations: int, *, default: str) -> np.ndarray:
    if labels is None:
        return np.full(n_observations, default, dtype=object)
    label_array = np.asarray(labels, dtype=object).ravel()
    if label_array.shape[0] != n_observations:
        raise ValueError("labels must have one value per observation")
    return label_array


def _validate_same_length(first: np.ndarray, second: np.ndarray, first_name: str, second_name: str) -> None:
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"{first_name} and {second_name} must have the same number of rows")


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result
