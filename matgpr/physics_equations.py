from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd
import torch

from .gpytorch_gpr import PhysicsEquation, PhysicsInformedMean

R_GAS_CONSTANT_J_MOL_K = 8.31446261815324

__all__ = [
    "R_GAS_CONSTANT_J_MOL_K",
    "PhysicsFeatureSpec",
    "PhysicsEquationTemplate",
    "PhysicsParameterSpec",
    "arrhenius_linear_growth_equation",
    "arrhenius_rate_equation",
    "arrhenius_sqrt_time_equation",
    "available_physics_equation_templates",
    "describe_physics_equation_template",
    "free_volume_exponential_equation",
    "get_physics_equation_template",
    "hall_petch_equation",
    "linear_parabolic_growth_equation",
    "linear_growth_equation",
    "list_physics_equation_templates",
    "power_law_equation",
    "rule_of_mixtures_equation",
    "search_physics_equation_templates",
    "summarize_physics_equation_templates",
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
class PhysicsFeatureSpec:
    """Metadata for one canonical physics-equation input feature."""

    name: str
    description: str = ""
    units: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("feature spec name must be non-empty")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "description", str(self.description))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable feature metadata record."""
        return {
            "name": self.name,
            "description": self.description,
            "units": self.units,
            "required": self.required,
        }


@dataclass(frozen=True)
class PhysicsParameterSpec:
    """Metadata for one physics-equation parameter."""

    name: str
    kind: str
    default_value: float | None = None
    description: str = ""
    units: str | None = None
    positive: bool = False
    learned_on_fit: bool = False

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("parameter spec name must be non-empty")
        if self.kind not in {"learnable", "fixed"}:
            raise ValueError("parameter spec kind must be 'learnable' or 'fixed'")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "description", str(self.description))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable parameter metadata record."""
        return {
            "name": self.name,
            "kind": self.kind,
            "default_value": self.default_value,
            "description": self.description,
            "units": self.units,
            "positive": self.positive,
            "learned_on_fit": self.learned_on_fit,
        }


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
    category: str = ""
    description: str = ""
    equation_latex: str = ""
    target_description: str = ""
    target_units: str | None = None
    feature_descriptions: Mapping[str, str] = field(default_factory=dict)
    feature_units: Mapping[str, str] = field(default_factory=dict)
    parameter_descriptions: Mapping[str, str] = field(default_factory=dict)
    parameter_units: Mapping[str, str] = field(default_factory=dict)
    applications: Sequence[str] = ()
    assumptions: Sequence[str] = ()
    notes: Sequence[str] = ()
    references: Sequence[str] = ()
    tags: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.feature_names:
            raise ValueError("feature_names must contain at least one feature")

        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "positive_parameters", tuple(self.positive_parameters))
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "applications", tuple(self.applications))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "tags", tuple(self.tags))
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
        object.__setattr__(self, "feature_units", dict(self.feature_units))
        object.__setattr__(
            self,
            "parameter_descriptions",
            dict(self.parameter_descriptions),
        )
        object.__setattr__(self, "parameter_units", dict(self.parameter_units))

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

    def feature_specs(self) -> tuple[PhysicsFeatureSpec, ...]:
        """Return report-ready metadata for the required input features."""
        return tuple(
            PhysicsFeatureSpec(
                name=feature,
                description=self.feature_descriptions.get(feature, ""),
                units=self.feature_units.get(feature),
                required=True,
            )
            for feature in self.feature_names
        )

    def parameter_specs(self) -> tuple[PhysicsParameterSpec, ...]:
        """Return report-ready metadata for learned and fixed parameters."""
        learnable = tuple(
            PhysicsParameterSpec(
                name=name,
                kind="learnable",
                default_value=value,
                description=self.parameter_descriptions.get(name, ""),
                units=self.parameter_units.get(name),
                positive=name in self.positive_parameters,
                learned_on_fit=True,
            )
            for name, value in self.default_learnable_parameters.items()
        )
        fixed = tuple(
            PhysicsParameterSpec(
                name=name,
                kind="fixed",
                default_value=value,
                description=self.parameter_descriptions.get(name, ""),
                units=self.parameter_units.get(name),
                positive=False,
                learned_on_fit=False,
            )
            for name, value in self.default_fixed_parameters.items()
        )
        return (*learnable, *fixed)

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
            "category": self.category,
            "features": list(self.feature_names),
            "learnable_parameters": dict(self.default_learnable_parameters),
            "positive_parameters": list(self.positive_parameters),
            "fixed_parameters": dict(self.default_fixed_parameters),
            "description": self.description,
            "equation_latex": self.equation_latex,
            "applications": list(self.applications),
            "assumptions": list(self.assumptions),
            "tags": list(self.tags),
        }

    def discovery_record(self) -> dict[str, object]:
        """Return complete metadata for registry/discovery workflows."""
        return {
            **self.summary(),
            "target_description": self.target_description,
            "target_units": self.target_units,
            "feature_metadata": [spec.to_dict() for spec in self.feature_specs()],
            "parameter_metadata": [spec.to_dict() for spec in self.parameter_specs()],
            "notes": list(self.notes),
            "references": list(self.references),
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


def arrhenius_linear_growth_equation(
    features: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Linear-growth mean controlled by an Arrhenius rate.

    The equation is ``offset + A exp(-E_a / (R T)) time``. It is useful for
    reaction-controlled oxidation, aging, or damage accumulation when the
    response is approximately linear in time at fixed temperature.
    """
    time = _positive_feature(features, "time")
    offset = _parameter(parameters, "offset", reference=time, default=0.0)
    rate = arrhenius_rate_equation(features, parameters)
    return offset + rate * time


def linear_growth_equation(
    features: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Linear growth with exposure time or another nonnegative driving variable.

    The equation is ``offset + rate * time``. It is useful for oxidation,
    aging, damage accumulation, or thickness/mass growth when a first-order
    time trend is an appropriate physics prior.
    """
    time = _positive_feature(features, "time")
    rate = _parameter(parameters, "rate", reference=time)
    offset = _parameter(parameters, "offset", reference=time, default=0.0)
    return offset + rate * time


def linear_parabolic_growth_equation(
    features: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Linear-parabolic growth from mixed interface and diffusion control.

    The equation solves ``m(t)^2 + linear_rate * m(t) = parabolic_rate * time``
    for the nonnegative root and adds an optional ``offset``. It is useful for
    oxidation or other growth processes that transition smoothly between
    interface-controlled and diffusion-controlled kinetics.
    """
    time = _positive_feature(features, "time")
    linear_rate = _parameter(parameters, "linear_rate", reference=time)
    parabolic_rate = _parameter(parameters, "parabolic_rate", reference=time)
    offset = _parameter(parameters, "offset", reference=time, default=0.0)
    discriminant = linear_rate**2 + 4.0 * parabolic_rate * time
    growth = 0.5 * (torch.sqrt(torch.clamp(discriminant, min=0.0)) - linear_rate)
    return offset + growth


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


def describe_physics_equation_template(name: str) -> dict[str, object]:
    """Return complete registry metadata for one physics-equation template."""
    return get_physics_equation_template(name).discovery_record()


def search_physics_equation_templates(
    query: str | None = None,
    *,
    application: str | None = None,
    tag: str | None = None,
    required_features: Sequence[str] | None = None,
) -> tuple[PhysicsEquationTemplate, ...]:
    """Search registered templates by text, application, tag, or features.

    This searches the built-in, human-authored template registry. It does not
    automatically discover new equations from data.
    """
    required_feature_set = (
        {_normalize_template_name(feature) for feature in required_features}
        if required_features is not None
        else None
    )
    matches: list[PhysicsEquationTemplate] = []
    for template in _PHYSICS_EQUATION_TEMPLATES:
        if query is not None and _normalize_template_name(query) not in _template_search_text(template):
            continue
        if application is not None and not _contains_normalized(
            template.applications,
            application,
        ):
            continue
        if tag is not None and _normalize_template_name(tag) not in {
            _normalize_template_name(item) for item in template.tags
        }:
            continue
        if required_feature_set is not None:
            template_feature_set = {
                _normalize_template_name(feature) for feature in template.feature_names
            }
            if not required_feature_set.issubset(template_feature_set):
                continue
        matches.append(template)
    return tuple(matches)


def summarize_physics_equation_templates(
    *,
    include_aliases: bool = True,
) -> pd.DataFrame:
    """Return a dataframe summary of registered physics-equation templates."""
    rows = []
    for template in _PHYSICS_EQUATION_TEMPLATES:
        rows.append(
            {
                "name": template.name,
                "aliases": ", ".join(template.aliases) if include_aliases else "",
                "category": template.category,
                "description": template.description,
                "equation_latex": template.equation_latex,
                "required_features": ", ".join(template.feature_names),
                "learnable_parameters": ", ".join(template.learnable_parameter_names),
                "fixed_parameters": ", ".join(template.fixed_parameter_names),
                "positive_parameters": ", ".join(template.positive_parameters),
                "applications": ", ".join(template.applications),
                "assumptions": " ".join(template.assumptions),
                "tags": ", ".join(template.tags),
            }
        )
    return pd.DataFrame(rows)


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
        category="temperature_activated",
        description="Temperature-activated Arrhenius rate or property trend.",
        equation_latex=r"m(T)=A\exp\left(-E_a/(RT)\right)",
        target_description="Rate-like or temperature-activated material response.",
        target_units="target property units",
        feature_descriptions={"temperature_k": "Absolute temperature in Kelvin."},
        feature_units={"temperature_k": "K"},
        parameter_descriptions={
            "prefactor": "Pre-exponential response scale in target units.",
            "activation_energy": "Activation energy in J/mol.",
            "gas_constant": "Gas constant in J/(mol K).",
        },
        parameter_units={
            "prefactor": "target property units",
            "activation_energy": "J/mol",
            "gas_constant": "J/(mol K)",
        },
        applications=("diffusion", "conductivity", "reaction rate", "thermal aging"),
        assumptions=(
            "The response follows a single dominant thermally activated process.",
            "Temperature is absolute temperature in Kelvin.",
        ),
        notes=("Use Kelvin for temperature.",),
        references=("Arrhenius-type activated-process model.",),
        tags=("temperature", "activation_energy", "transport", "kinetics"),
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
        category="temperature_time_growth",
        description="Square-root-time response controlled by an Arrhenius rate.",
        equation_latex=r"m(T,t)=b+\sqrt{A\exp\left(-E_a/(RT)\right)t}",
        target_description="Growth depth, damage, aging, or transport response with parabolic time scaling.",
        target_units="target property units",
        feature_descriptions={
            "temperature_k": "Absolute temperature in Kelvin.",
            "time": "Exposure, aging, diffusion, or reaction time.",
        },
        feature_units={
            "temperature_k": "K",
            "time": "consistent time unit",
        },
        parameter_descriptions={
            "prefactor": "Rate prefactor.",
            "activation_energy": "Activation energy in J/mol.",
            "offset": "Baseline response in target units.",
            "gas_constant": "Gas constant in J/(mol K).",
        },
        parameter_units={
            "prefactor": "target^2/time units when target is a growth length",
            "activation_energy": "J/mol",
            "offset": "target property units",
            "gas_constant": "J/(mol K)",
        },
        applications=("oxidation thickness", "diffusion depth", "thermal aging"),
        assumptions=(
            "The rate is controlled by a single Arrhenius process.",
            "The response grows approximately with the square root of time.",
        ),
        notes=("Keep time units consistent between training and prediction.",),
        references=("Arrhenius rate law with parabolic growth kinetics.",),
        tags=("temperature", "time", "activation_energy", "parabolic_growth"),
    ),
    PhysicsEquationTemplate(
        name="arrhenius_linear_growth",
        aliases=(
            "temperature_linear_growth",
            "arrhenius_linear_oxidation",
            "temperature_linear_oxidation",
        ),
        equation=arrhenius_linear_growth_equation,
        feature_names=("temperature_k", "time"),
        default_learnable_parameters={
            "prefactor": 1.0,
            "activation_energy": 10_000.0,
            "offset": 0.0,
        },
        positive_parameters=("prefactor", "activation_energy"),
        default_fixed_parameters={"gas_constant": R_GAS_CONSTANT_J_MOL_K},
        category="temperature_time_growth",
        description="Linear time-growth response controlled by an Arrhenius rate.",
        equation_latex=r"m(T,t)=b+A\exp\left(-E_a/(RT)\right)t",
        target_description="Growth, mass gain, damage, or aging response with linear time scaling.",
        target_units="target property units",
        feature_descriptions={
            "temperature_k": "Absolute temperature in Kelvin.",
            "time": "Exposure, aging, diffusion, reaction, or process time.",
        },
        feature_units={
            "temperature_k": "K",
            "time": "consistent time unit",
        },
        parameter_descriptions={
            "prefactor": "Linear-rate prefactor.",
            "activation_energy": "Activation energy in J/mol.",
            "offset": "Baseline response in target units.",
            "gas_constant": "Gas constant in J/(mol K).",
        },
        parameter_units={
            "prefactor": "target property units per time",
            "activation_energy": "J/mol",
            "offset": "target property units",
            "gas_constant": "J/(mol K)",
        },
        applications=("linear oxidation", "reaction-controlled growth", "thermal aging"),
        assumptions=(
            "Growth is approximately linear with time at fixed temperature.",
            "The linear rate follows a single dominant Arrhenius process.",
            "Temperature is absolute temperature in Kelvin.",
        ),
        notes=(
            "Use this when temperature changes the growth rate but the observed "
            "response does not follow square-root-time kinetics.",
        ),
        references=("Arrhenius rate law with linear growth kinetics.",),
        tags=("temperature", "time", "activation_energy", "linear_growth", "oxidation"),
    ),
    PhysicsEquationTemplate(
        name="linear_growth",
        aliases=("linear_time_growth", "linear_oxidation"),
        equation=linear_growth_equation,
        feature_names=("time",),
        default_learnable_parameters={
            "rate": 1.0,
            "offset": 0.0,
        },
        positive_parameters=("rate",),
        category="time_growth",
        description="Linear response with a nonnegative time or exposure variable.",
        equation_latex=r"m(t)=b+k t",
        target_description="Thickness, mass gain, damage, or other linearly growing response.",
        target_units="target property units",
        feature_descriptions={
            "time": "Exposure, aging, diffusion, reaction, or process time.",
        },
        feature_units={
            "time": "consistent time unit",
        },
        parameter_descriptions={
            "rate": "Linear growth rate in target units per time.",
            "offset": "Baseline response in target units.",
        },
        parameter_units={
            "rate": "target property units per time",
            "offset": "target property units",
        },
        applications=("linear oxidation", "aging", "damage accumulation", "growth"),
        assumptions=(
            "The response is approximately linear over the modeled time range.",
            "Time is nonnegative and reported in a consistent unit.",
        ),
        notes=(
            "Use this as a low-order mean function when the residual GP should "
            "capture deviations from simple linear kinetics.",
        ),
        references=("Linear time-growth model.",),
        tags=("time", "linear", "growth", "oxidation"),
    ),
    PhysicsEquationTemplate(
        name="linear_parabolic_growth",
        aliases=("linear_parabolic", "mixed_control_growth", "linear_parabolic_oxidation"),
        equation=linear_parabolic_growth_equation,
        feature_names=("time",),
        default_learnable_parameters={
            "linear_rate": 1.0,
            "parabolic_rate": 1.0,
            "offset": 0.0,
        },
        positive_parameters=("linear_rate", "parabolic_rate"),
        category="time_growth",
        description="Mixed-control growth with linear and parabolic kinetic terms.",
        equation_latex=r"m(t)=b+\frac{\sqrt{k_l^2+4k_p t}-k_l}{2}",
        target_description="Thickness, mass gain, damage, or growth response with mixed time scaling.",
        target_units="target property units",
        feature_descriptions={
            "time": "Exposure, aging, diffusion, reaction, or process time.",
        },
        feature_units={
            "time": "consistent time unit",
        },
        parameter_descriptions={
            "linear_rate": "Linear or interface-control term in target units.",
            "parabolic_rate": "Parabolic growth-rate contribution.",
            "offset": "Baseline response in target units.",
        },
        parameter_units={
            "linear_rate": "target property units",
            "parabolic_rate": "target property units squared per time",
            "offset": "target property units",
        },
        applications=("linear-parabolic oxidation", "mixed-control growth", "thermal aging"),
        assumptions=(
            "Both interface and diffusion control can contribute over the modeled range.",
            "The transition between regimes is smooth and can be represented by one effective pair of rates.",
            "Time is nonnegative and reported in a consistent unit.",
        ),
        notes=(
            "Use this when a dataset spans early-time linear behavior and "
            "later-time parabolic behavior.",
        ),
        references=("Linear-parabolic mixed-control growth model.",),
        tags=("time", "linear_parabolic", "mixed_control", "oxidation"),
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
        category="empirical_scaling",
        description="Empirical power-law scaling with one positive driving variable.",
        equation_latex=r"m(x)=b+c x^n",
        target_description="Material property with empirical monotonic or nonmonotonic scaling in one driver.",
        target_units="target property units",
        feature_descriptions={
            "driving_variable": "Positive load, rate, concentration, field, or other driver."
        },
        feature_units={"driving_variable": "driver units"},
        parameter_descriptions={
            "coefficient": "Power-law amplitude.",
            "exponent": "Power-law exponent.",
            "offset": "Baseline response in target units.",
        },
        parameter_units={
            "coefficient": "target units divided by driver units^exponent",
            "exponent": "dimensionless",
            "offset": "target property units",
        },
        applications=("indentation load effects", "strain-rate effects", "scaling laws"),
        assumptions=(
            "The selected driver is positive.",
            "A power-law trend is a plausible low-order inductive bias.",
        ),
        notes=("Map any positive driver, such as load, to the canonical feature name.",),
        references=("Empirical power-law scaling model.",),
        tags=("scaling", "empirical", "load", "rate"),
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
        category="microstructure_strengthening",
        description="Inverse-square-root grain-size strengthening trend.",
        equation_latex=r"m(d)=\sigma_0+k d^{-1/2}",
        target_description="Strength, hardness, yield stress, or related microstructure-sensitive property.",
        target_units="target property units",
        feature_descriptions={"grain_size": "Positive grain size in a consistent length unit."},
        feature_units={"grain_size": "length unit"},
        parameter_descriptions={
            "intrinsic_property": "Large-grain limiting property.",
            "coefficient": "Grain-size strengthening coefficient.",
        },
        parameter_units={
            "intrinsic_property": "target property units",
            "coefficient": "target property units * sqrt(length unit)",
        },
        applications=("hardness", "yield stress", "strength"),
        assumptions=(
            "Grain-boundary strengthening is relevant for the material class.",
            "A single effective grain-size descriptor is meaningful.",
        ),
        notes=("Use only when grain-size strengthening is meaningful for the dataset.",),
        references=("Hall-Petch inverse-square-root grain-size relation.",),
        tags=("microstructure", "grain_size", "strength", "hardness"),
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
        category="polymer_transport",
        description="Exponential free-volume trend for polymer transport or mobility.",
        equation_latex=r"m(f_v)=b+A\exp\left(-B/f_v\right)",
        target_description="Polymer transport, permeability, diffusion, or mobility response.",
        target_units="target property units",
        feature_descriptions={"free_volume_fraction": "Positive fractional free volume."},
        feature_units={"free_volume_fraction": "dimensionless fraction"},
        parameter_descriptions={
            "prefactor": "Transport or mobility scale.",
            "barrier": "Free-volume sensitivity parameter.",
            "offset": "Baseline response in target units.",
        },
        parameter_units={
            "prefactor": "target property units",
            "barrier": "dimensionless fraction",
            "offset": "target property units",
        },
        applications=("polymer diffusion", "gas permeability", "segmental mobility"),
        assumptions=(
            "Fractional free volume is available and physically meaningful.",
            "Transport increases with available free volume after residual GP correction.",
        ),
        notes=("Fractional free volume should be positive and consistently computed.",),
        references=("Free-volume exponential transport model.",),
        tags=("polymer", "free_volume", "diffusion", "permeability"),
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
        category="mixture_model",
        description="Rule-of-mixtures baseline with a quadratic interaction correction.",
        equation_latex=r"m(\phi)=(1-\phi)y_m+\phi y_i+\gamma\phi(1-\phi)",
        target_description="Composite, blend, alloy, or mixture property.",
        target_units="target property units",
        feature_descriptions={"volume_fraction": "Inclusion or second-phase volume fraction."},
        feature_units={"volume_fraction": "dimensionless fraction"},
        parameter_descriptions={
            "matrix_property": "Property of the matrix or first phase.",
            "inclusion_property": "Property of the inclusion or second phase.",
            "interaction": "Nonlinear interaction correction.",
        },
        parameter_units={
            "matrix_property": "target property units",
            "inclusion_property": "target property units",
            "interaction": "target property units",
        },
        applications=("composites", "alloys", "polymer blends", "mixtures"),
        assumptions=(
            "The system can be approximated by a two-phase composition variable.",
            "The volume fraction is bounded between 0 and 1.",
        ),
        notes=("The feature is clipped to [0, 1] inside the equation.",),
        references=("Rule-of-mixtures baseline with an interaction correction.",),
        tags=("mixture", "composite", "alloy", "volume_fraction"),
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


def _template_search_text(template: PhysicsEquationTemplate) -> str:
    parts: list[str] = [
        template.name,
        *template.aliases,
        template.category,
        template.description,
        template.equation_latex,
        template.target_description,
        *template.feature_names,
        *template.default_learnable_parameters,
        *template.default_fixed_parameters,
        *template.applications,
        *template.assumptions,
        *template.notes,
        *template.tags,
    ]
    parts.extend(template.feature_descriptions.values())
    parts.extend(template.parameter_descriptions.values())
    return " ".join(_normalize_template_name(part) for part in parts)


def _contains_normalized(values: Sequence[str], query: str) -> bool:
    normalized_query = _normalize_template_name(query)
    return any(normalized_query in _normalize_template_name(value) for value in values)


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
