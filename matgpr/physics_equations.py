from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from .gpytorch_gpr import PhysicsEquation, PhysicsInformedMean

R_GAS_CONSTANT_J_MOL_K = 8.31446261815324

__all__ = [
    "R_GAS_CONSTANT_J_MOL_K",
    "PhysicsEquationTemplate",
    "arrhenius_rate_equation",
    "arrhenius_sqrt_time_equation",
    "available_physics_equation_templates",
    "free_volume_exponential_equation",
    "get_physics_equation_template",
    "hall_petch_equation",
    "list_physics_equation_templates",
    "power_law_equation",
    "rule_of_mixtures_equation",
]


def _validate_numeric_mapping(values: Mapping[str, float], name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if not key:
            raise ValueError(f"{name} keys must be non-empty")
        numeric_value = float(value)
        if not torch.isfinite(torch.tensor(numeric_value)):
            raise ValueError(f"{name}[{key!r}] must be finite")
        result[str(key)] = numeric_value
    return result


@dataclass(frozen=True)
class PhysicsEquationTemplate:
    """Reusable physics-equation metadata for physics-informed GPR.

    A template stores the equation callable plus the features, default
    learnable-parameter initial values, fixed constants, sign constraints, and
    documentation needed to use the equation as a
    :class:`matgpr.PhysicsInformedMean`.
    """

    name: str
    equation: PhysicsEquation
    feature_names: Sequence[str]
    default_learnable_parameters: Mapping[str, float] = field(default_factory=dict)
    positive_parameters: Sequence[str] = ()
    default_fixed_parameters: Mapping[str, float] = field(default_factory=dict)
    aliases: Sequence[str] = ()
    description: str = ""
    equation_latex: str = ""
    feature_descriptions: Mapping[str, str] = field(default_factory=dict)
    parameter_descriptions: Mapping[str, str] = field(default_factory=dict)
    applications: Sequence[str] = ()
    notes: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.feature_names:
            raise ValueError("feature_names must contain at least one feature")

        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "positive_parameters", tuple(self.positive_parameters))
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "applications", tuple(self.applications))
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(
            self,
            "default_learnable_parameters",
            _validate_numeric_mapping(
                self.default_learnable_parameters,
                "default_learnable_parameters",
            ),
        )
        object.__setattr__(
            self,
            "default_fixed_parameters",
            _validate_numeric_mapping(self.default_fixed_parameters, "default_fixed_parameters"),
        )
        object.__setattr__(
            self,
            "feature_descriptions",
            dict(self.feature_descriptions),
        )
        object.__setattr__(
            self,
            "parameter_descriptions",
            dict(self.parameter_descriptions),
        )

        missing_positive = set(self.positive_parameters).difference(
            self.default_learnable_parameters
        )
        if missing_positive:
            raise ValueError(
                "positive_parameters must refer to default learnable parameters; "
                f"unknown names: {sorted(missing_positive)}"
            )

    @property
    def required_features(self) -> tuple[str, ...]:
        """Feature names expected by the equation callable."""
        return tuple(self.feature_names)

    @property
    def learnable_parameter_names(self) -> tuple[str, ...]:
        """Default parameter names optimized jointly with the GP."""
        return tuple(self.default_learnable_parameters)

    @property
    def fixed_parameter_names(self) -> tuple[str, ...]:
        """Default fixed-constant names passed to the equation."""
        return tuple(self.default_fixed_parameters)

    def build_mean_function(
        self,
        feature_indices: Mapping[str, int],
        *,
        learnable_parameter_overrides: Mapping[str, float] | None = None,
        fixed_parameter_overrides: Mapping[str, float] | None = None,
        positive_parameters: Sequence[str] | None = None,
        feature_means: Mapping[str, float] | None = None,
        feature_stds: Mapping[str, float] | None = None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        strict_features: bool = True,
    ) -> PhysicsInformedMean:
        """Build a :class:`matgpr.PhysicsInformedMean` from this template.

        Parameters
        ----------
        feature_indices
            Mapping from template feature names to columns in the model feature
            matrix. Keys use the template's canonical feature names even if the
            original dataframe columns have different names.
        learnable_parameter_overrides
            Optional initial values that replace selected template defaults.
        fixed_parameter_overrides
            Optional fixed constants that replace selected template defaults.
        positive_parameters
            Optional replacement for the template positive-parameter list.
        feature_means, feature_stds
            Optional scaling metadata so the physics equation receives original
            physical units when the GPR feature matrix is standardized.
        target_mean, target_std
            Target standardization metadata passed to ``PhysicsInformedMean``.
        strict_features
            If ``True``, require ``feature_indices`` to include every template
            feature.
        """
        resolved_feature_indices = dict(feature_indices)
        if strict_features:
            missing = [
                feature for feature in self.feature_names if feature not in resolved_feature_indices
            ]
            if missing:
                raise ValueError(f"feature_indices is missing template features: {missing}")

        learnable_parameters = dict(self.default_learnable_parameters)
        if learnable_parameter_overrides is not None:
            learnable_parameters.update(
                _validate_numeric_mapping(
                    learnable_parameter_overrides,
                    "learnable_parameter_overrides",
                )
            )

        fixed_parameters = dict(self.default_fixed_parameters)
        if fixed_parameter_overrides is not None:
            fixed_parameters.update(
                _validate_numeric_mapping(
                    fixed_parameter_overrides,
                    "fixed_parameter_overrides",
                )
            )

        positive = (
            tuple(self.positive_parameters)
            if positive_parameters is None
            else tuple(positive_parameters)
        )

        return PhysicsInformedMean(
            equation=self.equation,
            feature_indices=resolved_feature_indices,
            learnable_parameters=learnable_parameters,
            positive_parameters=positive,
            fixed_parameters=fixed_parameters,
            feature_means=feature_means,
            feature_stds=feature_stds,
            target_mean=target_mean,
            target_std=target_std,
        )

    def summary(self) -> dict[str, object]:
        """Return a compact serializable summary for reports."""
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "features": list(self.feature_names),
            "learnable_parameters": dict(self.default_learnable_parameters),
            "positive_parameters": list(self.positive_parameters),
            "fixed_parameters": dict(self.default_fixed_parameters),
            "description": self.description,
            "equation_latex": self.equation_latex,
            "applications": list(self.applications),
        }


def arrhenius_rate_equation(
    features: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Temperature-activated Arrhenius response.

    The equation is ``A exp(-E_a / (R T))``. It is useful for transport,
    reaction, diffusion, conductivity, and rate-like material properties when
    temperature is an important control variable.
    """
    temperature = _positive_feature(features, "temperature_k")
    prefactor = _parameter(parameters, "prefactor", reference=temperature)
    activation_energy = _parameter(parameters, "activation_energy", reference=temperature)
    gas_constant = _parameter(
        parameters,
        "gas_constant",
        reference=temperature,
        default=R_GAS_CONSTANT_J_MOL_K,
    )
    denominator = torch.clamp(gas_constant * temperature, min=_tiny_like(temperature))
    return prefactor * torch.exp(-activation_energy / denominator)


def arrhenius_sqrt_time_equation(
    features: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Parabolic-growth mean from Arrhenius rate and exposure time.

    The equation is ``offset + sqrt(A exp(-E_a / (R T)) time)``. It is useful
    for oxidation, diffusion-depth, aging, and other square-root-time trends.
    """
    time = _positive_feature(features, "time")
    offset = _parameter(parameters, "offset", reference=time, default=0.0)
    rate = arrhenius_rate_equation(features, parameters)
    return offset + torch.sqrt(torch.clamp(rate * time, min=0.0))


def power_law_equation(
    features: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Power-law scaling with a single driving variable.

    The equation is ``offset + coefficient * driving_variable ** exponent``.
    It is a compact template for load effects, rate effects, and empirical
    scaling laws where a log-log trend is expected.
    """
    driving_variable = _positive_feature(features, "driving_variable")
    coefficient = _parameter(parameters, "coefficient", reference=driving_variable)
    exponent = _parameter(parameters, "exponent", reference=driving_variable)
    offset = _parameter(parameters, "offset", reference=driving_variable, default=0.0)
    return offset + coefficient * torch.pow(driving_variable, exponent)


def hall_petch_equation(
    features: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Hall-Petch-style inverse-square-root grain-size trend.

    The equation is ``intrinsic_property + coefficient / sqrt(grain_size)``.
    It is useful for strength, hardness, yield stress, or related properties
    when grain-size strengthening is physically plausible.
    """
    grain_size = _positive_feature(features, "grain_size")
    intrinsic_property = _parameter(parameters, "intrinsic_property", reference=grain_size)
    coefficient = _parameter(parameters, "coefficient", reference=grain_size)
    return intrinsic_property + coefficient / torch.sqrt(grain_size)


def free_volume_exponential_equation(
    features: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Free-volume exponential trend for polymer transport properties.

    The equation is ``offset + prefactor * exp(-barrier / free_volume_fraction)``.
    It is useful for diffusion, permeability, and segmental-mobility trends in
    polymers when fractional free volume is available.
    """
    free_volume_fraction = _positive_feature(features, "free_volume_fraction")
    prefactor = _parameter(parameters, "prefactor", reference=free_volume_fraction)
    barrier = _parameter(parameters, "barrier", reference=free_volume_fraction)
    offset = _parameter(parameters, "offset", reference=free_volume_fraction, default=0.0)
    return offset + prefactor * torch.exp(-barrier / free_volume_fraction)


def rule_of_mixtures_equation(
    features: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Linear rule of mixtures with an optional interaction term.

    The equation is ``(1 - phi) matrix_property + phi inclusion_property +
    interaction phi (1 - phi)``. It is useful for composite, alloy, blend, or
    mixture properties when phase fraction is known.
    """
    volume_fraction = torch.clamp(_feature(features, "volume_fraction"), min=0.0, max=1.0)
    matrix_property = _parameter(parameters, "matrix_property", reference=volume_fraction)
    inclusion_property = _parameter(parameters, "inclusion_property", reference=volume_fraction)
    interaction = _parameter(parameters, "interaction", reference=volume_fraction, default=0.0)
    return (
        (1.0 - volume_fraction) * matrix_property
        + volume_fraction * inclusion_property
        + interaction * volume_fraction * (1.0 - volume_fraction)
    )


def list_physics_equation_templates() -> tuple[PhysicsEquationTemplate, ...]:
    """Return the registered reusable physics equation templates."""
    return _PHYSICS_EQUATION_TEMPLATES


def available_physics_equation_templates(*, include_aliases: bool = False) -> tuple[str, ...]:
    """Return available template names, optionally including aliases."""
    names = [template.name for template in _PHYSICS_EQUATION_TEMPLATES]
    if include_aliases:
        names.extend(alias for template in _PHYSICS_EQUATION_TEMPLATES for alias in template.aliases)
    return tuple(names)


def get_physics_equation_template(name: str) -> PhysicsEquationTemplate:
    """Return a physics-equation template by name or alias."""
    key = _normalize_template_name(name)
    lookup = _template_lookup()
    if key not in lookup:
        available = ", ".join(available_physics_equation_templates(include_aliases=True))
        raise ValueError(f"Unknown physics equation template '{name}'. Available templates: {available}")
    return lookup[key]


_PHYSICS_EQUATION_TEMPLATES = (
    PhysicsEquationTemplate(
        name="arrhenius_rate",
        aliases=("arrhenius", "temperature_activated"),
        equation=arrhenius_rate_equation,
        feature_names=("temperature_k",),
        default_learnable_parameters={
            "prefactor": 1.0,
            "activation_energy": 10_000.0,
        },
        positive_parameters=("prefactor", "activation_energy"),
        default_fixed_parameters={"gas_constant": R_GAS_CONSTANT_J_MOL_K},
        description="Temperature-activated Arrhenius rate or property trend.",
        equation_latex=r"m(T)=A\exp\left(-E_a/(RT)\right)",
        feature_descriptions={"temperature_k": "Absolute temperature in Kelvin."},
        parameter_descriptions={
            "prefactor": "Pre-exponential response scale in target units.",
            "activation_energy": "Activation energy in J/mol.",
            "gas_constant": "Gas constant in J/(mol K).",
        },
        applications=("diffusion", "conductivity", "reaction rate", "thermal aging"),
        notes=("Use Kelvin for temperature.",),
    ),
    PhysicsEquationTemplate(
        name="arrhenius_sqrt_time",
        aliases=("parabolic_growth", "oxidation_growth"),
        equation=arrhenius_sqrt_time_equation,
        feature_names=("temperature_k", "time"),
        default_learnable_parameters={
            "prefactor": 1.0,
            "activation_energy": 10_000.0,
            "offset": 0.0,
        },
        positive_parameters=("prefactor", "activation_energy"),
        default_fixed_parameters={"gas_constant": R_GAS_CONSTANT_J_MOL_K},
        description="Square-root-time response controlled by an Arrhenius rate.",
        equation_latex=r"m(T,t)=b+\sqrt{A\exp\left(-E_a/(RT)\right)t}",
        feature_descriptions={
            "temperature_k": "Absolute temperature in Kelvin.",
            "time": "Exposure, aging, diffusion, or reaction time.",
        },
        parameter_descriptions={
            "prefactor": "Rate prefactor.",
            "activation_energy": "Activation energy in J/mol.",
            "offset": "Baseline response in target units.",
            "gas_constant": "Gas constant in J/(mol K).",
        },
        applications=("oxidation thickness", "diffusion depth", "thermal aging"),
        notes=("Keep time units consistent between training and prediction.",),
    ),
    PhysicsEquationTemplate(
        name="power_law",
        aliases=("scaling_law", "load_power_law"),
        equation=power_law_equation,
        feature_names=("driving_variable",),
        default_learnable_parameters={
            "coefficient": 1.0,
            "exponent": 1.0,
            "offset": 0.0,
        },
        positive_parameters=("coefficient",),
        description="Empirical power-law scaling with one positive driving variable.",
        equation_latex=r"m(x)=b+c x^n",
        feature_descriptions={
            "driving_variable": "Positive load, rate, concentration, field, or other driver."
        },
        parameter_descriptions={
            "coefficient": "Power-law amplitude.",
            "exponent": "Power-law exponent.",
            "offset": "Baseline response in target units.",
        },
        applications=("indentation load effects", "strain-rate effects", "scaling laws"),
        notes=("Map any positive driver, such as load, to the canonical feature name.",),
    ),
    PhysicsEquationTemplate(
        name="hall_petch",
        aliases=("grain_size_strengthening", "inverse_sqrt_grain_size"),
        equation=hall_petch_equation,
        feature_names=("grain_size",),
        default_learnable_parameters={
            "intrinsic_property": 1.0,
            "coefficient": 1.0,
        },
        positive_parameters=("intrinsic_property", "coefficient"),
        description="Inverse-square-root grain-size strengthening trend.",
        equation_latex=r"m(d)=\sigma_0+k d^{-1/2}",
        feature_descriptions={"grain_size": "Positive grain size in a consistent length unit."},
        parameter_descriptions={
            "intrinsic_property": "Large-grain limiting property.",
            "coefficient": "Grain-size strengthening coefficient.",
        },
        applications=("hardness", "yield stress", "strength"),
        notes=("Use only when grain-size strengthening is meaningful for the dataset.",),
    ),
    PhysicsEquationTemplate(
        name="free_volume_exponential",
        aliases=("free_volume", "polymer_transport_free_volume"),
        equation=free_volume_exponential_equation,
        feature_names=("free_volume_fraction",),
        default_learnable_parameters={
            "prefactor": 1.0,
            "barrier": 0.1,
            "offset": 0.0,
        },
        positive_parameters=("prefactor", "barrier"),
        description="Exponential free-volume trend for polymer transport or mobility.",
        equation_latex=r"m(f_v)=b+A\exp\left(-B/f_v\right)",
        feature_descriptions={"free_volume_fraction": "Positive fractional free volume."},
        parameter_descriptions={
            "prefactor": "Transport or mobility scale.",
            "barrier": "Free-volume sensitivity parameter.",
            "offset": "Baseline response in target units.",
        },
        applications=("polymer diffusion", "gas permeability", "segmental mobility"),
        notes=("Fractional free volume should be positive and consistently computed.",),
    ),
    PhysicsEquationTemplate(
        name="rule_of_mixtures",
        aliases=("mixture_rule", "composite_mixture"),
        equation=rule_of_mixtures_equation,
        feature_names=("volume_fraction",),
        default_learnable_parameters={
            "matrix_property": 1.0,
            "inclusion_property": 2.0,
            "interaction": 0.0,
        },
        description="Rule-of-mixtures baseline with a quadratic interaction correction.",
        equation_latex=r"m(\phi)=(1-\phi)y_m+\phi y_i+\gamma\phi(1-\phi)",
        feature_descriptions={"volume_fraction": "Inclusion or second-phase volume fraction."},
        parameter_descriptions={
            "matrix_property": "Property of the matrix or first phase.",
            "inclusion_property": "Property of the inclusion or second phase.",
            "interaction": "Nonlinear interaction correction.",
        },
        applications=("composites", "alloys", "polymer blends", "mixtures"),
        notes=("The feature is clipped to [0, 1] inside the equation.",),
    ),
)


def _template_lookup() -> dict[str, PhysicsEquationTemplate]:
    lookup: dict[str, PhysicsEquationTemplate] = {}
    for template in _PHYSICS_EQUATION_TEMPLATES:
        keys = (template.name, *template.aliases)
        for key in keys:
            normalized = _normalize_template_name(key)
            if normalized in lookup:
                raise ValueError(f"Duplicate physics equation template name or alias: {key}")
            lookup[normalized] = template
    return lookup


def _normalize_template_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _feature(features: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in features:
        raise KeyError(f"Missing required physics feature '{name}'")
    value = features[name]
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _positive_feature(features: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    value = _feature(features, name)
    return torch.clamp(value, min=_tiny_like(value))


def _parameter(
    parameters: Mapping[str, torch.Tensor],
    name: str,
    *,
    reference: torch.Tensor,
    default: float | None = None,
) -> torch.Tensor:
    if name in parameters:
        value = parameters[name]
    elif default is not None:
        value = default
    else:
        raise KeyError(f"Missing required physics parameter '{name}'")

    if isinstance(value, torch.Tensor):
        return value.to(dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _tiny_like(value: torch.Tensor) -> float:
    if value.dtype in {torch.float16, torch.float32, torch.bfloat16}:
        return 1e-12
    return 1e-15
