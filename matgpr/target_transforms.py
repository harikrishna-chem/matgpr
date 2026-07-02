from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .gpytorch_gpr import GPyTorchPrediction

__all__ = [
    "BoundedTargetTransform",
    "IdentityTargetTransform",
    "LogTargetTransform",
    "PhysicsResidualTransform",
    "StandardizedTargetTransform",
    "TargetTransformSpec",
    "available_target_transform_specs",
    "describe_target_transform_spec",
    "get_target_transform_spec",
    "list_target_transform_specs",
    "make_materials_target_transform",
    "make_target_transform",
    "search_target_transform_specs",
    "summarize_target_transform_specs",
]


@dataclass
class IdentityTargetTransform:
    """No-op target transform with the same interface as other transforms."""

    def fit(self, y, **kwargs):
        """Validate target values and return the transform."""
        _to_1d_finite(y, "y")
        return self

    def transform(self, y, **kwargs) -> np.ndarray:
        """Return target values unchanged."""
        return _to_1d_finite(y, "y")

    def fit_transform(self, y, **kwargs) -> np.ndarray:
        """Fit the transform and return transformed target values."""
        self.fit(y, **kwargs)
        return self.transform(y, **kwargs)

    def inverse_transform(self, y_transformed, **kwargs) -> np.ndarray:
        """Return transformed target values unchanged."""
        return _to_1d_finite(y_transformed, "y_transformed")

    def inverse_std(self, mean_transformed, std_transformed, **kwargs) -> np.ndarray:
        """Return predictive standard deviations unchanged."""
        _to_1d_finite(mean_transformed, "mean_transformed")
        return _to_nonnegative_std(std_transformed)

    def inverse_prediction(self, prediction: GPyTorchPrediction, **kwargs) -> GPyTorchPrediction:
        """Return a prediction object in the original target scale."""
        return _inverse_prediction_with_arrays(self, prediction, **kwargs)


@dataclass
class StandardizedTargetTransform:
    """Standardize target values using training-set mean and standard deviation."""

    mean_: float | None = field(default=None, init=False)
    scale_: float | None = field(default=None, init=False)

    def fit(self, y, **kwargs):
        """Estimate the mean and standard deviation from training targets."""
        values = _to_1d_finite(y, "y")
        scale = float(np.std(values, ddof=0))
        if scale <= 0:
            raise ValueError("Cannot standardize a constant target")
        self.mean_ = float(np.mean(values))
        self.scale_ = scale
        return self

    def transform(self, y, **kwargs) -> np.ndarray:
        """Return standardized target values."""
        self._require_fitted()
        values = _to_1d_finite(y, "y")
        return (values - self.mean_) / self.scale_

    def fit_transform(self, y, **kwargs) -> np.ndarray:
        """Fit the transform and return standardized target values."""
        self.fit(y, **kwargs)
        return self.transform(y, **kwargs)

    def inverse_transform(self, y_transformed, **kwargs) -> np.ndarray:
        """Return standardized values in the original target scale."""
        self._require_fitted()
        values = _to_1d_finite(y_transformed, "y_transformed")
        return values * self.scale_ + self.mean_

    def inverse_std(self, mean_transformed, std_transformed, **kwargs) -> np.ndarray:
        """Return predictive standard deviations in the original target scale."""
        self._require_fitted()
        _to_1d_finite(mean_transformed, "mean_transformed")
        return _to_nonnegative_std(std_transformed) * self.scale_

    def inverse_prediction(self, prediction: GPyTorchPrediction, **kwargs) -> GPyTorchPrediction:
        """Return a prediction object in the original target scale."""
        return _inverse_prediction_with_arrays(self, prediction, **kwargs)

    def _require_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardizedTargetTransform must be fitted before use")


@dataclass
class LogTargetTransform:
    """Log-transform positive targets and invert Gaussian predictions.

    The transform is ``z = log(y + offset)``. Predictive means and standard
    deviations are inverted with log-normal moments, which is more appropriate
    than simply exponentiating the transformed-space mean.
    """

    offset: float = 0.0

    def fit(self, y, **kwargs):
        """Validate that all shifted target values are positive."""
        self._validate_shifted_target(y, "y")
        return self

    def transform(self, y, **kwargs) -> np.ndarray:
        """Return log-transformed target values."""
        values = self._validate_shifted_target(y, "y")
        return np.log(values + self.offset)

    def fit_transform(self, y, **kwargs) -> np.ndarray:
        """Fit the transform and return log-transformed target values."""
        self.fit(y, **kwargs)
        return self.transform(y, **kwargs)

    def inverse_transform(self, y_transformed, **kwargs) -> np.ndarray:
        """Return log-scale values in the original target scale."""
        values = _to_1d_finite(y_transformed, "y_transformed")
        return np.exp(values) - self.offset

    def inverse_std(self, mean_transformed, std_transformed, **kwargs) -> np.ndarray:
        """Return original-scale standard deviations from log-normal moments."""
        mean = _to_1d_finite(mean_transformed, "mean_transformed")
        std = _to_nonnegative_std(std_transformed)
        _validate_same_length(mean, std, "mean_transformed", "std_transformed")
        variance = std**2
        return np.sqrt(np.expm1(variance) * np.exp(2.0 * mean + variance))

    def inverse_prediction(self, prediction: GPyTorchPrediction, **kwargs) -> GPyTorchPrediction:
        """Return a log-space prediction object in the original target scale."""
        mean = _to_1d_finite(prediction.mean, "prediction.mean")
        std = None
        if prediction.std is not None:
            std = self.inverse_std(mean, prediction.std)

        lower = None
        if prediction.lower is not None:
            lower = self.inverse_transform(prediction.lower)

        upper = None
        if prediction.upper is not None:
            upper = self.inverse_transform(prediction.upper)

        if prediction.std is not None:
            original_mean = np.exp(mean + 0.5 * _to_nonnegative_std(prediction.std) ** 2) - self.offset
        else:
            original_mean = self.inverse_transform(mean)

        return GPyTorchPrediction(mean=original_mean, std=std, lower=lower, upper=upper)

    def _validate_shifted_target(self, y, name: str) -> np.ndarray:
        values = _to_1d_finite(y, name)
        if np.any(values + self.offset <= 0):
            raise ValueError(f"{name} + offset must contain only positive values")
        return values


@dataclass
class BoundedTargetTransform:
    """Logit-transform targets constrained to a finite physical interval.

    The transform is:

    ``z = logit((y - lower_bound) / (upper_bound - lower_bound))``.

    Predictive means and standard deviations are inverted with Gauss-Hermite
    quadrature over the implied logistic-normal distribution. This is useful
    for properties with known physical bounds, such as efficiencies,
    fractions, probabilities, normalized phase fractions, or bounded scores.
    """

    lower_bound: float
    upper_bound: float
    n_quadrature_points: int = 20

    def __post_init__(self) -> None:
        self.lower_bound = float(self.lower_bound)
        self.upper_bound = float(self.upper_bound)
        if not np.isfinite(self.lower_bound) or not np.isfinite(self.upper_bound):
            raise ValueError("Bounds must be finite")
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound")
        if not isinstance(self.n_quadrature_points, int) or self.n_quadrature_points < 3:
            raise ValueError("n_quadrature_points must be an integer greater than or equal to 3")

    def fit(self, y, **kwargs):
        """Validate that targets are strictly inside the physical interval."""
        self._validate_bounded_target(y, "y")
        return self

    def transform(self, y, **kwargs) -> np.ndarray:
        """Return logit-transformed target values."""
        values = self._validate_bounded_target(y, "y")
        scaled = (values - self.lower_bound) / self._width
        return np.log(scaled / (1.0 - scaled))

    def fit_transform(self, y, **kwargs) -> np.ndarray:
        """Fit the transform and return logit-transformed target values."""
        self.fit(y, **kwargs)
        return self.transform(y, **kwargs)

    def inverse_transform(self, y_transformed, **kwargs) -> np.ndarray:
        """Return logit-space values in the original bounded target scale."""
        values = _to_1d_finite(y_transformed, "y_transformed")
        return self.lower_bound + self._width * _sigmoid(values)

    def inverse_std(self, mean_transformed, std_transformed, **kwargs) -> np.ndarray:
        """Return original-scale standard deviations from logistic-normal moments."""
        mean = _to_1d_finite(mean_transformed, "mean_transformed")
        std = _to_nonnegative_std(std_transformed)
        _validate_same_length(mean, std, "mean_transformed", "std_transformed")
        _, original_std = self._logistic_normal_moments(mean, std)
        return original_std

    def inverse_prediction(self, prediction: GPyTorchPrediction, **kwargs) -> GPyTorchPrediction:
        """Return a bounded prediction object in the original target scale."""
        mean = _to_1d_finite(prediction.mean, "prediction.mean")

        std = None
        if prediction.std is not None:
            original_mean, std = self._logistic_normal_moments(mean, prediction.std)
        else:
            original_mean = self.inverse_transform(mean)

        lower = None
        if prediction.lower is not None:
            lower = self.inverse_transform(prediction.lower)

        upper = None
        if prediction.upper is not None:
            upper = self.inverse_transform(prediction.upper)

        return GPyTorchPrediction(mean=original_mean, std=std, lower=lower, upper=upper)

    @property
    def _width(self) -> float:
        return self.upper_bound - self.lower_bound

    def _validate_bounded_target(self, y, name: str) -> np.ndarray:
        values = _to_1d_finite(y, name)
        if np.any(values <= self.lower_bound) or np.any(values >= self.upper_bound):
            raise ValueError(f"{name} must be strictly between lower_bound and upper_bound")
        return values

    def _logistic_normal_moments(
        self,
        mean_transformed: np.ndarray,
        std_transformed,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = _to_1d_finite(mean_transformed, "mean_transformed")
        std = _to_nonnegative_std(std_transformed)
        _validate_same_length(mean, std, "mean_transformed", "std_transformed")

        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quadrature_points)
        samples = mean[:, None] + np.sqrt(2.0) * std[:, None] * nodes[None, :]
        original_samples = self.lower_bound + self._width * _sigmoid(samples)
        normalized_weights = weights / np.sqrt(np.pi)
        original_mean = original_samples @ normalized_weights
        second_moment = (original_samples * original_samples) @ normalized_weights
        variance = np.maximum(second_moment - original_mean * original_mean, 0.0)
        return original_mean, np.sqrt(variance)


@dataclass
class PhysicsResidualTransform:
    """Model residuals relative to a physics baseline.

    This transform supports workflows where a simple physics model provides a
    baseline prediction and GPR learns the residual:
    ``residual = observed_target - physics_baseline``.
    """

    baseline_name: str = "physics_baseline"

    def fit(self, y, *, baseline):
        """Validate target and baseline arrays."""
        values = _to_1d_finite(y, "y")
        baseline = _to_1d_finite(baseline, self.baseline_name)
        _validate_same_length(values, baseline, "y", self.baseline_name)
        return self

    def transform(self, y, *, baseline) -> np.ndarray:
        """Return residual target values."""
        values = _to_1d_finite(y, "y")
        baseline = _to_1d_finite(baseline, self.baseline_name)
        _validate_same_length(values, baseline, "y", self.baseline_name)
        return values - baseline

    def fit_transform(self, y, *, baseline) -> np.ndarray:
        """Fit the transform and return residual target values."""
        self.fit(y, baseline=baseline)
        return self.transform(y, baseline=baseline)

    def inverse_transform(self, y_transformed, *, baseline) -> np.ndarray:
        """Return residual predictions in the original target scale."""
        residual = _to_1d_finite(y_transformed, "y_transformed")
        baseline = _to_1d_finite(baseline, self.baseline_name)
        _validate_same_length(residual, baseline, "y_transformed", self.baseline_name)
        return residual + baseline

    def inverse_std(self, mean_transformed, std_transformed, **kwargs) -> np.ndarray:
        """Return residual predictive standard deviations unchanged."""
        _to_1d_finite(mean_transformed, "mean_transformed")
        return _to_nonnegative_std(std_transformed)

    def inverse_prediction(self, prediction: GPyTorchPrediction, *, baseline) -> GPyTorchPrediction:
        """Add a physics baseline back to a residual prediction object."""
        return _inverse_prediction_with_arrays(self, prediction, baseline=baseline)


@dataclass(frozen=True)
class TargetTransformSpec:
    """Metadata for a reusable materials-property target-transform preset.

    The spec documents which transform is usually appropriate for a property
    type, stores default transform keyword arguments, and builds the concrete
    transform when requested.
    """

    name: str
    transform_name: str
    transform_kwargs: Mapping[str, object] = field(default_factory=dict)
    aliases: Sequence[str] = ()
    category: str = ""
    description: str = ""
    target_units: str | None = None
    examples: Sequence[str] = ()
    assumptions: Sequence[str] = ()
    notes: Sequence[str] = ()
    tags: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("name must be non-empty")
        if not str(self.transform_name).strip():
            raise ValueError("transform_name must be non-empty")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "transform_name", str(self.transform_name))
        object.__setattr__(self, "transform_kwargs", dict(self.transform_kwargs))
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "examples", tuple(self.examples))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "tags", tuple(self.tags))

    def build_transform(self, **overrides):
        """Build the concrete transform for this property preset."""
        transform_kwargs = dict(self.transform_kwargs)
        transform_kwargs.update(overrides)
        return make_target_transform(self.transform_name, **transform_kwargs)

    def summary(self) -> dict[str, object]:
        """Return a compact serializable summary for reports."""
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "category": self.category,
            "transform_name": self.transform_name,
            "transform_kwargs": dict(self.transform_kwargs),
            "description": self.description,
            "target_units": self.target_units,
            "examples": list(self.examples),
            "assumptions": list(self.assumptions),
            "tags": list(self.tags),
        }

    def discovery_record(self) -> dict[str, object]:
        """Return complete metadata for one transform preset."""
        return {
            **self.summary(),
            "notes": list(self.notes),
        }


def make_target_transform(name: str, **kwargs):
    """Create a target transform by name.

    Parameters
    ----------
    name
        One of ``"identity"``, ``"standard"``, ``"standardized"``, ``"log"``,
        ``"positive"``, ``"bounded"``, ``"logit"``, ``"residual"``, or
        ``"physics_residual"``. Registered materials-property presets such as
        ``"diffusivity"``, ``"efficiency_percent"``, ``"band_gap_ev"``, and
        ``"formation_energy"`` are also accepted.
    """
    normalized = _normalize_spec_name(name)
    if normalized in {"identity", "none", "passthrough"}:
        return IdentityTargetTransform(**kwargs)
    if normalized in {"standard", "standardized", "zscore", "z_score"}:
        return StandardizedTargetTransform(**kwargs)
    if normalized in {"log", "logarithmic", "positive", "positivity"}:
        return LogTargetTransform(**kwargs)
    if normalized in {"bounded", "bound", "logit"}:
        return BoundedTargetTransform(**kwargs)
    if normalized in {"residual", "physics_residual"}:
        return PhysicsResidualTransform(**kwargs)

    lookup = _target_transform_spec_lookup()
    if normalized in lookup:
        return lookup[normalized].build_transform(**kwargs)

    available = ", ".join(
        [
            "identity",
            "standard",
            "log",
            "bounded",
            "residual",
            *available_target_transform_specs(include_aliases=True),
        ]
    )
    raise ValueError(f"Unknown target transform '{name}'. Available transforms: {available}")


def make_materials_target_transform(property_name: str, **overrides):
    """Build a target transform from a materials-property preset."""
    return get_target_transform_spec(property_name).build_transform(**overrides)


def list_target_transform_specs() -> tuple[TargetTransformSpec, ...]:
    """Return registered materials-property target-transform presets."""
    return _TARGET_TRANSFORM_SPECS


def available_target_transform_specs(*, include_aliases: bool = False) -> tuple[str, ...]:
    """Return available target-transform preset names, optionally with aliases."""
    names = [spec.name for spec in _TARGET_TRANSFORM_SPECS]
    if include_aliases:
        names.extend(alias for spec in _TARGET_TRANSFORM_SPECS for alias in spec.aliases)
    return tuple(names)


def get_target_transform_spec(name: str) -> TargetTransformSpec:
    """Return a target-transform preset by name or alias."""
    key = _normalize_spec_name(name)
    lookup = _target_transform_spec_lookup()
    if key not in lookup:
        available = ", ".join(available_target_transform_specs(include_aliases=True))
        raise ValueError(f"Unknown target-transform preset '{name}'. Available presets: {available}")
    return lookup[key]


def describe_target_transform_spec(name: str) -> dict[str, object]:
    """Return complete registry metadata for one target-transform preset."""
    return get_target_transform_spec(name).discovery_record()


def search_target_transform_specs(
    query: str | None = None,
    *,
    category: str | None = None,
    tag: str | None = None,
    transform_name: str | None = None,
) -> tuple[TargetTransformSpec, ...]:
    """Search registered target-transform presets by text and metadata."""
    normalized_transform = (
        _normalize_spec_name(transform_name) if transform_name is not None else None
    )
    matches: list[TargetTransformSpec] = []
    for spec in _TARGET_TRANSFORM_SPECS:
        if query is not None and _normalize_spec_name(query) not in _target_spec_search_text(spec):
            continue
        if category is not None and _normalize_spec_name(category) != _normalize_spec_name(
            spec.category
        ):
            continue
        if tag is not None and _normalize_spec_name(tag) not in {
            _normalize_spec_name(item) for item in spec.tags
        }:
            continue
        if normalized_transform is not None and normalized_transform != _normalize_spec_name(
            spec.transform_name
        ):
            continue
        matches.append(spec)
    return tuple(matches)


def summarize_target_transform_specs(*, include_aliases: bool = True) -> pd.DataFrame:
    """Return a dataframe summary of registered target-transform presets."""
    rows = []
    for spec in _TARGET_TRANSFORM_SPECS:
        rows.append(
            {
                "name": spec.name,
                "aliases": ", ".join(spec.aliases) if include_aliases else "",
                "category": spec.category,
                "transform_name": spec.transform_name,
                "transform_kwargs": dict(spec.transform_kwargs),
                "description": spec.description,
                "target_units": spec.target_units,
                "examples": ", ".join(spec.examples),
                "assumptions": " ".join(spec.assumptions),
                "tags": ", ".join(spec.tags),
            }
        )
    return pd.DataFrame(rows)


_TARGET_TRANSFORM_SPECS = (
    TargetTransformSpec(
        name="efficiency_percent",
        aliases=("efficiency", "pce", "power_conversion_efficiency", "percent_efficiency"),
        transform_name="bounded",
        transform_kwargs={"lower_bound": 0.0, "upper_bound": 100.0},
        category="bounded_fractional_response",
        description="Efficiency reported as a percentage with finite 0 to 100 bounds.",
        target_units="%",
        examples=("solar-cell PCE", "catalyst conversion percent", "yield percent"),
        assumptions=("Values are strictly between 0 and 100 before the logit transform.",),
        notes=("Use the fraction preset when the target is stored from 0 to 1.",),
        tags=("bounded", "efficiency", "percentage", "opv", "photovoltaics"),
    ),
    TargetTransformSpec(
        name="fraction",
        aliases=("probability", "phase_fraction", "volume_fraction", "mole_fraction"),
        transform_name="bounded",
        transform_kwargs={"lower_bound": 0.0, "upper_bound": 1.0},
        category="bounded_fractional_response",
        description="Fraction, probability, or normalized phase amount bounded by 0 and 1.",
        target_units="dimensionless",
        examples=("phase fraction", "volume fraction", "classification probability"),
        assumptions=("Values are strictly between 0 and 1 before the logit transform.",),
        tags=("bounded", "fraction", "probability", "mixture"),
    ),
    TargetTransformSpec(
        name="percentage",
        aliases=("percent", "percent_yield", "yield_percent", "selectivity_percent"),
        transform_name="bounded",
        transform_kwargs={"lower_bound": 0.0, "upper_bound": 100.0},
        category="bounded_fractional_response",
        description="Generic percentage-valued target bounded by 0 and 100.",
        target_units="%",
        examples=("reaction yield percent", "selectivity percent", "conversion percent"),
        assumptions=("Values are strictly between 0 and 100 before the logit transform.",),
        tags=("bounded", "percentage", "yield", "selectivity"),
    ),
    TargetTransformSpec(
        name="diffusivity",
        aliases=("diffusion_coefficient", "diffusion_constant"),
        transform_name="log",
        transform_kwargs={"offset": 0.0},
        category="positive_transport",
        description="Positive transport coefficient often spanning orders of magnitude.",
        target_units="dataset units",
        examples=("solvent diffusivity", "ion diffusivity", "gas diffusion coefficient"),
        assumptions=("Values are positive and measured on a ratio scale.",),
        tags=("positive", "log", "transport", "diffusion"),
    ),
    TargetTransformSpec(
        name="permeability",
        aliases=("gas_permeability", "membrane_permeability"),
        transform_name="log",
        transform_kwargs={"offset": 0.0},
        category="positive_transport",
        description="Positive permeability-like property often modeled on a log scale.",
        target_units="dataset units",
        examples=("polymer gas permeability", "membrane permeability"),
        assumptions=("Values are positive and measured on a ratio scale.",),
        tags=("positive", "log", "transport", "polymer"),
    ),
    TargetTransformSpec(
        name="conductivity",
        aliases=("electrical_conductivity", "ionic_conductivity", "thermal_conductivity"),
        transform_name="log",
        transform_kwargs={"offset": 0.0},
        category="positive_transport",
        description="Positive conductivity-like property often spanning orders of magnitude.",
        target_units="dataset units",
        examples=("ionic conductivity", "electrical conductivity", "thermal conductivity"),
        assumptions=("Values are positive and measured on a ratio scale.",),
        tags=("positive", "log", "transport", "conductivity"),
    ),
    TargetTransformSpec(
        name="rate_constant",
        aliases=("rate", "reaction_rate", "kinetic_rate"),
        transform_name="log",
        transform_kwargs={"offset": 0.0},
        category="positive_kinetics",
        description="Positive kinetic rate or rate constant.",
        target_units="dataset units",
        examples=("reaction rate constant", "degradation rate", "crystallization rate"),
        assumptions=("Values are positive and measured on a ratio scale.",),
        tags=("positive", "log", "kinetics", "rate"),
    ),
    TargetTransformSpec(
        name="band_gap_ev",
        aliases=("band_gap", "bandgap", "eg"),
        transform_name="log",
        transform_kwargs={"offset": 1e-8},
        category="nonnegative_electronic_property",
        description="Nonnegative electronic band gap in electronvolts.",
        target_units="eV",
        examples=("DFT band gap", "experimental optical band gap"),
        assumptions=("Values are nonnegative; metals may have zero band gap.",),
        notes=("Use standardization if the dataset contains signed gap corrections.",),
        tags=("nonnegative", "log", "electronic", "band_gap"),
    ),
    TargetTransformSpec(
        name="energy_above_hull",
        aliases=("e_above_hull", "hull_energy", "decomposition_energy"),
        transform_name="log",
        transform_kwargs={"offset": 1e-8},
        category="nonnegative_stability",
        description="Nonnegative thermodynamic instability above the convex hull.",
        target_units="eV/atom",
        examples=("energy above hull", "decomposition energy"),
        assumptions=("Values are nonnegative; stable phases may be exactly zero.",),
        tags=("nonnegative", "log", "stability", "energy"),
    ),
    TargetTransformSpec(
        name="modulus",
        aliases=("elastic_modulus", "bulk_modulus", "shear_modulus", "youngs_modulus"),
        transform_name="log",
        transform_kwargs={"offset": 0.0},
        category="positive_mechanical_property",
        description="Positive elastic modulus or stiffness property.",
        target_units="dataset units",
        examples=("bulk modulus", "shear modulus", "Young's modulus"),
        assumptions=("Values are positive and measured on a ratio scale.",),
        tags=("positive", "log", "mechanical", "elasticity"),
    ),
    TargetTransformSpec(
        name="strength",
        aliases=("yield_strength", "ultimate_strength", "tensile_strength", "spall_strength"),
        transform_name="log",
        transform_kwargs={"offset": 0.0},
        category="positive_mechanical_property",
        description="Positive strength-like mechanical property.",
        target_units="dataset units",
        examples=("yield strength", "tensile strength", "spall strength"),
        assumptions=("Values are positive and measured on a ratio scale.",),
        tags=("positive", "log", "mechanical", "strength"),
    ),
    TargetTransformSpec(
        name="hardness",
        aliases=("vickers_hardness", "nanoindentation_hardness"),
        transform_name="log",
        transform_kwargs={"offset": 0.0},
        category="positive_mechanical_property",
        description="Positive indentation hardness property.",
        target_units="dataset units",
        examples=("Vickers hardness", "nanoindentation hardness"),
        assumptions=("Values are positive and measured on a ratio scale.",),
        tags=("positive", "log", "mechanical", "hardness"),
    ),
    TargetTransformSpec(
        name="transition_temperature_k",
        aliases=("transition_temperature", "melting_temperature_k", "glass_transition_temperature_k"),
        transform_name="log",
        transform_kwargs={"offset": 0.0},
        category="positive_temperature",
        description="Positive absolute transition temperature in Kelvin.",
        target_units="K",
        examples=("melting temperature", "glass-transition temperature", "Curie temperature"),
        assumptions=("Temperature is absolute Kelvin and positive.",),
        notes=("Do not use this preset for Celsius targets that can be negative."),
        tags=("positive", "log", "temperature"),
    ),
    TargetTransformSpec(
        name="formation_energy",
        aliases=("formation_energy_ev_atom", "formation_enthalpy"),
        transform_name="standard",
        category="signed_energy",
        description="Signed formation energy or enthalpy that may be negative or positive.",
        target_units="dataset units",
        examples=("formation energy per atom", "formation enthalpy"),
        assumptions=("The target is signed and does not have a strict positive domain.",),
        tags=("signed", "standard", "energy", "stability"),
    ),
    TargetTransformSpec(
        name="binding_energy",
        aliases=("adsorption_energy", "interaction_energy", "cohesive_energy"),
        transform_name="standard",
        category="signed_energy",
        description="Signed binding, adsorption, cohesive, or interaction energy.",
        target_units="dataset units",
        examples=("adsorption energy", "binding energy", "cohesive energy"),
        assumptions=("The sign convention is dataset-specific and must be reported.",),
        tags=("signed", "standard", "energy", "adsorption"),
    ),
)


def _inverse_prediction_with_arrays(transform, prediction: GPyTorchPrediction, **kwargs) -> GPyTorchPrediction:
    mean = transform.inverse_transform(prediction.mean, **kwargs)

    std = None
    if prediction.std is not None:
        std = transform.inverse_std(prediction.mean, prediction.std, **kwargs)

    lower = None
    if prediction.lower is not None:
        lower = transform.inverse_transform(prediction.lower, **kwargs)

    upper = None
    if prediction.upper is not None:
        upper = transform.inverse_transform(prediction.upper, **kwargs)

    return GPyTorchPrediction(mean=mean, std=std, lower=lower, upper=upper)


def _to_1d_finite(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _to_nonnegative_std(values) -> np.ndarray:
    array = _to_1d_finite(values, "std_transformed")
    if np.any(array < 0):
        raise ValueError("std_transformed must contain only non-negative values")
    return array


def _validate_same_length(first: np.ndarray, second: np.ndarray, first_name: str, second_name: str) -> None:
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"{first_name} and {second_name} must have the same length")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _normalize_spec_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def _target_transform_spec_lookup() -> dict[str, TargetTransformSpec]:
    lookup: dict[str, TargetTransformSpec] = {}
    for spec in _TARGET_TRANSFORM_SPECS:
        keys = (spec.name, *spec.aliases)
        for key in keys:
            normalized = _normalize_spec_name(key)
            if normalized in lookup:
                raise ValueError(f"Duplicate target-transform preset name or alias: {key}")
            lookup[normalized] = spec
    return lookup


def _target_spec_search_text(spec: TargetTransformSpec) -> str:
    parts = [
        spec.name,
        spec.transform_name,
        spec.category,
        spec.description,
        spec.target_units or "",
        *spec.aliases,
        *spec.examples,
        *spec.assumptions,
        *spec.notes,
        *spec.tags,
    ]
    return " ".join(_normalize_spec_name(part) for part in parts)
