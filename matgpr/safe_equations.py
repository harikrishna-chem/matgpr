from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
import torch

from .gpytorch_gpr import PhysicsInformedMean

SAFE_EQUATION_SCHEMA_VERSION = "0.1"

ALLOWED_SAFE_EQUATION_OPERATORS = (
    "+",
    "-",
    "*",
    "/",
    "**",
    "unary+",
    "unary-",
)

ALLOWED_SAFE_EQUATION_FUNCTIONS = (
    "abs",
    "exp",
    "log",
    "max",
    "min",
    "pow",
    "sqrt",
)

__all__ = [
    "ALLOWED_SAFE_EQUATION_FUNCTIONS",
    "ALLOWED_SAFE_EQUATION_OPERATORS",
    "SAFE_EQUATION_SCHEMA_VERSION",
    "SafeEquationConstant",
    "SafeEquationEvaluationResult",
    "SafeEquationMeanFunction",
    "SafeEquationMeanPreview",
    "SafeEquationParameter",
    "SafeEquationSpec",
    "SafeEquationValidationIssue",
    "SafeEquationValidationResult",
    "SafeEquationVariable",
    "build_safe_equation_mean_function",
    "evaluate_safe_equation_numpy",
    "evaluate_safe_equation_torch",
    "initialize_safe_equation_torch_parameters",
    "preview_safe_equation_mean",
    "safe_equation_schema_snapshot",
    "validate_safe_equation_spec",
]

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_NAMES = frozenset(ALLOWED_SAFE_EQUATION_FUNCTIONS)
_ALLOWED_BINARY_OPERATORS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
}
_ALLOWED_UNARY_OPERATORS: dict[type[ast.unaryop], str] = {
    ast.UAdd: "unary+",
    ast.USub: "unary-",
}
_ALLOWED_FUNCTION_ARITY: dict[str, tuple[int, int | None]] = {
    "abs": (1, 1),
    "exp": (1, 1),
    "log": (1, 1),
    "sqrt": (1, 1),
    "pow": (2, 2),
    "min": (2, None),
    "max": (2, None),
}


@dataclass(frozen=True)
class SafeEquationVariable:
    """One declared input variable for a safe custom equation."""

    name: str
    description: str = ""
    units: str | None = None
    required: bool = True
    bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_symbol_name(self.name, "variable name"))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "units", _optional_string(self.units, "variable units"))
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "bounds", _validate_bounds(self.bounds, "variable bounds"))

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "SafeEquationVariable":
        """Build a variable record from a JSON-like mapping."""
        return cls(
            name=_required(record, "name", "SafeEquationVariable"),
            description=record.get("description", ""),
            units=record.get("units"),
            required=record.get("required", True),
            bounds=record.get("bounds"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe variable record."""
        return {
            "name": self.name,
            "description": self.description,
            "units": self.units,
            "required": self.required,
            "bounds": _bounds_to_list(self.bounds),
        }


@dataclass(frozen=True)
class SafeEquationParameter:
    """One declared equation parameter.

    ``initial_value`` is the starting value for learnable parameters and the
    default value for fixed parameters when downstream tools choose to keep a
    parameter fixed.
    """

    name: str
    initial_value: float
    bounds: tuple[float, float] | None = None
    description: str = ""
    units: str | None = None
    learnable: bool = True

    def __post_init__(self) -> None:
        initial_value = _finite_float(self.initial_value, "parameter initial_value")
        bounds = _validate_bounds(self.bounds, "parameter bounds")
        if bounds is not None and not bounds[0] <= initial_value <= bounds[1]:
            raise ValueError("parameter initial_value must lie within bounds")

        object.__setattr__(self, "name", _validate_symbol_name(self.name, "parameter name"))
        object.__setattr__(self, "initial_value", initial_value)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "units", _optional_string(self.units, "parameter units"))
        object.__setattr__(self, "learnable", bool(self.learnable))

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "SafeEquationParameter":
        """Build a parameter record from a JSON-like mapping."""
        return cls(
            name=_required(record, "name", "SafeEquationParameter"),
            initial_value=_required(record, "initial_value", "SafeEquationParameter"),
            bounds=record.get("bounds"),
            description=record.get("description", ""),
            units=record.get("units"),
            learnable=record.get("learnable", True),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe parameter record."""
        return {
            "name": self.name,
            "initial_value": self.initial_value,
            "bounds": _bounds_to_list(self.bounds),
            "description": self.description,
            "units": self.units,
            "learnable": self.learnable,
        }


@dataclass(frozen=True)
class SafeEquationConstant:
    """One declared finite constant used by a safe custom equation."""

    name: str
    value: float
    description: str = ""
    units: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_symbol_name(self.name, "constant name"))
        object.__setattr__(self, "value", _finite_float(self.value, "constant value"))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "units", _optional_string(self.units, "constant units"))

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "SafeEquationConstant":
        """Build a constant record from a JSON-like mapping."""
        return cls(
            name=_required(record, "name", "SafeEquationConstant"),
            value=_required(record, "value", "SafeEquationConstant"),
            description=record.get("description", ""),
            units=record.get("units"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe constant record."""
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "units": self.units,
        }


@dataclass(frozen=True)
class SafeEquationSpec:
    """JSON-safe custom equation specification for future PI-GPR mean functions."""

    name: str
    expression: str
    variables: Sequence[SafeEquationVariable | Mapping[str, Any]]
    parameters: Sequence[SafeEquationParameter | Mapping[str, Any]] = ()
    constants: Sequence[SafeEquationConstant | Mapping[str, Any]] = ()
    display_name: str = ""
    description: str = ""
    target_units: str | None = None
    assumptions: Sequence[str] = ()
    validity_limits: Mapping[str, Any] = field(default_factory=dict)
    references: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SAFE_EQUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = _validate_symbol_name(self.name, "spec name")
        expression = str(self.expression).strip()
        if not expression:
            raise ValueError("expression must be non-empty")

        variables = tuple(_coerce_variable(variable) for variable in self.variables)
        parameters = tuple(_coerce_parameter(parameter) for parameter in self.parameters)
        constants = tuple(_coerce_constant(constant) for constant in self.constants)
        if not variables:
            raise ValueError("variables must contain at least one variable")
        _validate_unique_symbols(variables, parameters, constants)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "constants", constants)
        object.__setattr__(self, "display_name", str(self.display_name))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(
            self, "target_units", _optional_string(self.target_units, "target_units")
        )
        object.__setattr__(self, "assumptions", _string_tuple(self.assumptions, "assumptions"))
        object.__setattr__(
            self,
            "validity_limits",
            _json_safe_mapping(self.validity_limits, "validity_limits"),
        )
        object.__setattr__(self, "references", _string_tuple(self.references, "references"))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "schema_version", str(self.schema_version))

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "SafeEquationSpec":
        """Build a safe equation spec from a JSON-like mapping."""
        return cls(
            name=_required(record, "name", "SafeEquationSpec"),
            expression=_required(record, "expression", "SafeEquationSpec"),
            variables=record.get("variables", ()),
            parameters=record.get("parameters", ()),
            constants=record.get("constants", ()),
            display_name=record.get("display_name", ""),
            description=record.get("description", ""),
            target_units=record.get("target_units"),
            assumptions=record.get("assumptions", ()),
            validity_limits=record.get("validity_limits", {}),
            references=record.get("references", ()),
            metadata=record.get("metadata", {}),
            schema_version=record.get("schema_version", SAFE_EQUATION_SCHEMA_VERSION),
        )

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Declared input-variable names."""
        return tuple(variable.name for variable in self.variables)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Declared parameter names."""
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def constant_names(self) -> tuple[str, ...]:
        """Declared constant names."""
        return tuple(constant.name for constant in self.constants)

    @property
    def declared_symbols(self) -> tuple[str, ...]:
        """All declared variables, parameters, and constants."""
        return (*self.variable_names, *self.parameter_names, *self.constant_names)

    def to_dict(self) -> dict[str, object]:
        """Return a strict-JSON-safe equation specification."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "display_name": self.display_name,
            "expression": self.expression,
            "variables": [variable.to_dict() for variable in self.variables],
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "constants": [constant.to_dict() for constant in self.constants],
            "description": self.description,
            "target_units": self.target_units,
            "assumptions": list(self.assumptions),
            "validity_limits": dict(self.validity_limits),
            "references": list(self.references),
            "metadata": dict(self.metadata),
            "declared_symbols": list(self.declared_symbols),
            "allowed_functions": list(ALLOWED_SAFE_EQUATION_FUNCTIONS),
        }


@dataclass(frozen=True)
class SafeEquationValidationIssue:
    """Structured validation issue for a safe equation specification."""

    code: str
    message: str
    field: str | None = None
    severity: str = "error"

    def __post_init__(self) -> None:
        if not str(self.code).strip():
            raise ValueError("issue code must be non-empty")
        if self.severity not in {"error", "warning"}:
            raise ValueError("issue severity must be 'error' or 'warning'")
        object.__setattr__(self, "code", str(self.code))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "field", _optional_string(self.field, "issue field"))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe validation issue."""
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class SafeEquationValidationResult:
    """Validation result for a safe equation specification."""

    issues: Sequence[SafeEquationValidationIssue | Mapping[str, Any]] = ()
    spec_name: str | None = None
    declared_symbols: Sequence[str] = ()
    expression_symbols: Sequence[str] = ()
    expression_functions: Sequence[str] = ()
    allowed_functions: Sequence[str] = ALLOWED_SAFE_EQUATION_FUNCTIONS
    allowed_operators: Sequence[str] = ALLOWED_SAFE_EQUATION_OPERATORS
    schema_version: str = SAFE_EQUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        issues = tuple(_coerce_issue(issue) for issue in self.issues)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "spec_name", _optional_string(self.spec_name, "spec_name"))
        object.__setattr__(
            self,
            "declared_symbols",
            _string_tuple(self.declared_symbols, "declared_symbols"),
        )
        object.__setattr__(
            self,
            "expression_symbols",
            _string_tuple(self.expression_symbols, "expression_symbols"),
        )
        object.__setattr__(
            self,
            "expression_functions",
            _string_tuple(self.expression_functions, "expression_functions"),
        )
        object.__setattr__(
            self,
            "allowed_functions",
            _string_tuple(self.allowed_functions, "allowed_functions"),
        )
        object.__setattr__(
            self,
            "allowed_operators",
            _string_tuple(self.allowed_operators, "allowed_operators"),
        )
        object.__setattr__(self, "schema_version", str(self.schema_version))

    @property
    def is_valid(self) -> bool:
        """Whether the result contains no error-severity issues."""
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[SafeEquationValidationIssue, ...]:
        """Error-severity issues."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[SafeEquationValidationIssue, ...]:
        """Warning-severity issues."""
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe validation result."""
        return {
            "schema_version": self.schema_version,
            "is_valid": self.is_valid,
            "spec_name": self.spec_name,
            "declared_symbols": list(self.declared_symbols),
            "expression_symbols": list(self.expression_symbols),
            "expression_functions": list(self.expression_functions),
            "allowed_functions": list(self.allowed_functions),
            "allowed_operators": list(self.allowed_operators),
            "issues": [issue.to_dict() for issue in self.issues],
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class SafeEquationEvaluationResult:
    """JSON-safe shell for future safe-equation evaluation results."""

    spec_name: str
    values: Sequence[float | None] = ()
    finite_count: int = 0
    nonfinite_count: int = 0
    warnings: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SAFE_EQUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        values = tuple(_optional_finite_float(value, "evaluation value") for value in self.values)
        object.__setattr__(self, "spec_name", _validate_symbol_name(self.spec_name, "spec_name"))
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self, "finite_count", _nonnegative_int(self.finite_count, "finite_count")
        )
        object.__setattr__(
            self,
            "nonfinite_count",
            _nonnegative_int(self.nonfinite_count, "nonfinite_count"),
        )
        object.__setattr__(self, "warnings", _string_tuple(self.warnings, "warnings"))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "schema_version", str(self.schema_version))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe evaluation result."""
        return {
            "schema_version": self.schema_version,
            "spec_name": self.spec_name,
            "values": list(self.values),
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SafeEquationMeanPreview:
    """JSON-safe shell for future physics-mean preview payloads."""

    spec_name: str
    preview_ready: bool
    valid_input_row_count: int = 0
    invalid_input_row_count: int = 0
    finite_mean_count: int = 0
    nonfinite_mean_count: int = 0
    physics_mean_summary: Mapping[str, Any] = field(default_factory=dict)
    target_residual_summary: Mapping[str, Any] | None = None
    warnings: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SAFE_EQUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec_name", _validate_symbol_name(self.spec_name, "spec_name"))
        object.__setattr__(self, "preview_ready", bool(self.preview_ready))
        object.__setattr__(
            self,
            "valid_input_row_count",
            _nonnegative_int(self.valid_input_row_count, "valid_input_row_count"),
        )
        object.__setattr__(
            self,
            "invalid_input_row_count",
            _nonnegative_int(self.invalid_input_row_count, "invalid_input_row_count"),
        )
        object.__setattr__(
            self,
            "finite_mean_count",
            _nonnegative_int(self.finite_mean_count, "finite_mean_count"),
        )
        object.__setattr__(
            self,
            "nonfinite_mean_count",
            _nonnegative_int(self.nonfinite_mean_count, "nonfinite_mean_count"),
        )
        object.__setattr__(
            self,
            "physics_mean_summary",
            _json_safe_mapping(self.physics_mean_summary, "physics_mean_summary"),
        )
        residual_summary = (
            None
            if self.target_residual_summary is None
            else _json_safe_mapping(self.target_residual_summary, "target_residual_summary")
        )
        object.__setattr__(self, "target_residual_summary", residual_summary)
        object.__setattr__(self, "warnings", _string_tuple(self.warnings, "warnings"))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "schema_version", str(self.schema_version))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe physics-mean preview shell."""
        return {
            "schema_version": self.schema_version,
            "spec_name": self.spec_name,
            "preview_ready": self.preview_ready,
            "valid_input_row_count": self.valid_input_row_count,
            "invalid_input_row_count": self.invalid_input_row_count,
            "finite_mean_count": self.finite_mean_count,
            "nonfinite_mean_count": self.nonfinite_mean_count,
            "physics_mean_summary": dict(self.physics_mean_summary),
            "target_residual_summary": (
                None if self.target_residual_summary is None else dict(self.target_residual_summary)
            ),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


class _SafeEquationTorchEquation:
    """Callable equation adapter used by ``SafeEquationMeanFunction``."""

    def __init__(self, spec: SafeEquationSpec):
        self.spec = spec

    def __call__(
        self,
        features: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return evaluate_safe_equation_torch(
            self.spec,
            features,
            parameter_values=parameters,
        )


class SafeEquationMeanFunction(PhysicsInformedMean):
    """Physics-informed mean function backed by a safe custom equation spec."""

    def __init__(
        self,
        spec: SafeEquationSpec | Mapping[str, Any],
        *,
        feature_indices: Mapping[str, int] | None = None,
        column_indices: Mapping[str, int] | None = None,
        parameter_values: Mapping[str, Any] | None = None,
        positive_parameters: Sequence[str] | None = None,
        infer_positive_parameters: bool = True,
        feature_means: Mapping[str, float] | None = None,
        feature_stds: Mapping[str, float] | None = None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        resolved = _require_valid_spec(spec)
        resolved_feature_indices = _resolve_safe_equation_feature_indices(
            resolved,
            feature_indices=feature_indices,
            column_indices=column_indices,
        )
        learnable_parameters, fixed_parameters = _safe_equation_parameter_maps(
            resolved,
            parameter_values=parameter_values,
        )
        resolved_positive_parameters = _resolve_safe_equation_positive_parameters(
            resolved,
            learnable_parameters=learnable_parameters,
            positive_parameters=positive_parameters,
            infer_positive_parameters=infer_positive_parameters,
        )

        super().__init__(
            equation=_SafeEquationTorchEquation(resolved),
            feature_indices=resolved_feature_indices,
            learnable_parameters=learnable_parameters,
            positive_parameters=resolved_positive_parameters,
            fixed_parameters=fixed_parameters,
            feature_means=feature_means,
            feature_stds=feature_stds,
            target_mean=target_mean,
            target_std=target_std,
        )
        self.safe_equation_spec = resolved
        self.safe_equation_feature_names = tuple(resolved_feature_indices)
        self.safe_equation_parameter_bounds = {
            parameter.name: _bounds_to_list(parameter.bounds) for parameter in resolved.parameters
        }

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe metadata describing the mean-function adapter."""
        return {
            "schema_version": self.safe_equation_spec.schema_version,
            "spec": self.safe_equation_spec.to_dict(),
            "feature_indices": dict(self.feature_indices),
            "feature_names": list(self.safe_equation_feature_names),
            "learnable_parameters": list(self._learnable_parameter_names),
            "fixed_parameters": list(self.fixed_parameters),
            "positive_parameters": sorted(self.positive_parameters),
            "parameter_bounds": dict(self.safe_equation_parameter_bounds),
            "current_parameter_values": self.current_parameter_values(),
        }


def validate_safe_equation_spec(
    spec: SafeEquationSpec | Mapping[str, Any],
) -> SafeEquationValidationResult:
    """Validate the schema-level safety of a custom equation spec.

    This validation layer checks names, duplicate symbols, finite numeric
    values, parameter bounds, JSON-safe metadata, and expression AST safety.
    """
    spec_name = _spec_name_from_unknown(spec)
    try:
        resolved = spec if isinstance(spec, SafeEquationSpec) else SafeEquationSpec.from_dict(spec)
    except (TypeError, ValueError, KeyError) as exc:
        issue = SafeEquationValidationIssue(
            code="invalid_spec",
            message=str(exc),
            field=None,
            severity="error",
        )
        return SafeEquationValidationResult(issues=(issue,), spec_name=spec_name)

    issues, expression_symbols, expression_functions = _validate_expression(resolved)
    return SafeEquationValidationResult(
        issues=tuple(issues),
        spec_name=resolved.name,
        declared_symbols=resolved.declared_symbols,
        expression_symbols=tuple(sorted(expression_symbols)),
        expression_functions=tuple(sorted(expression_functions)),
        allowed_functions=ALLOWED_SAFE_EQUATION_FUNCTIONS,
        allowed_operators=ALLOWED_SAFE_EQUATION_OPERATORS,
    )


def build_safe_equation_mean_function(
    spec: SafeEquationSpec | Mapping[str, Any],
    *,
    feature_indices: Mapping[str, int] | None = None,
    column_indices: Mapping[str, int] | None = None,
    parameter_values: Mapping[str, Any] | None = None,
    positive_parameters: Sequence[str] | None = None,
    infer_positive_parameters: bool = True,
    feature_means: Mapping[str, float] | None = None,
    feature_stds: Mapping[str, float] | None = None,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> SafeEquationMeanFunction:
    """Build a trainable PI-GPR mean function from a safe equation spec."""
    return SafeEquationMeanFunction(
        spec,
        feature_indices=feature_indices,
        column_indices=column_indices,
        parameter_values=parameter_values,
        positive_parameters=positive_parameters,
        infer_positive_parameters=infer_positive_parameters,
        feature_means=feature_means,
        feature_stds=feature_stds,
        target_mean=target_mean,
        target_std=target_std,
    )


def evaluate_safe_equation_numpy(
    spec: SafeEquationSpec | Mapping[str, Any],
    variable_values: Mapping[str, Any] | Any,
    *,
    parameter_values: Mapping[str, Any] | None = None,
    constant_values: Mapping[str, Any] | None = None,
) -> SafeEquationEvaluationResult:
    """Evaluate a validated safe equation with NumPy.

    The expression is never passed to ``eval``. The function validates the
    equation spec, parses the already allowlisted AST, and interprets only the
    small supported math grammar. Non-finite outputs are returned as ``None`` so
    ``to_dict()`` remains strict-JSON-safe.
    """
    resolved = _resolve_spec_or_failure(spec)
    if isinstance(resolved, SafeEquationEvaluationResult):
        return resolved
    validation = validate_safe_equation_spec(resolved)
    if not validation.is_valid:
        return _evaluation_failure(
            spec_name=resolved.name,
            warnings=_validation_warnings(validation),
            metadata={
                "evaluation_ready": False,
                "validation": validation.to_dict(),
            },
        )

    context_result = _build_numpy_context(
        resolved,
        variable_values,
        parameter_values=parameter_values,
        constant_values=constant_values,
    )
    if context_result["warnings"]:
        return _evaluation_failure(
            spec_name=resolved.name,
            warnings=context_result["warnings"],
            metadata={
                "evaluation_ready": False,
                "validation": validation.to_dict(),
                "input_row_count": context_result["input_row_count"],
            },
        )

    try:
        tree = ast.parse(resolved.expression, mode="eval")
        with np.errstate(all="ignore"):
            raw_values = _evaluate_numpy_node(tree, context_result["context"])
    except (FloatingPointError, OverflowError, ValueError, TypeError) as exc:
        return _evaluation_failure(
            spec_name=resolved.name,
            warnings=(f"evaluation failed: {exc}",),
            metadata={
                "evaluation_ready": False,
                "validation": validation.to_dict(),
                "input_row_count": context_result["input_row_count"],
            },
        )

    values_array = np.asarray(raw_values, dtype=float)
    if values_array.ndim == 0:
        row_count = int(context_result["input_row_count"])
        values_array = np.full(row_count, float(values_array), dtype=float)
    values_array = np.ravel(values_array)
    finite_mask = np.isfinite(values_array)
    finite_count = int(np.count_nonzero(finite_mask))
    nonfinite_count = int(values_array.size - finite_count)
    warnings: list[str] = []
    if nonfinite_count:
        warnings.append(f"equation produced {nonfinite_count} non-finite output value(s)")

    return SafeEquationEvaluationResult(
        spec_name=resolved.name,
        values=_json_safe_array_values(values_array),
        finite_count=finite_count,
        nonfinite_count=nonfinite_count,
        warnings=warnings,
        metadata={
            "evaluation_ready": nonfinite_count == 0,
            "input_row_count": int(values_array.size),
            "expression_symbols": list(validation.expression_symbols),
            "expression_functions": list(validation.expression_functions),
            "parameter_values": context_result["parameter_values"],
            "constant_values": context_result["constant_values"],
        },
    )


def evaluate_safe_equation_torch(
    spec: SafeEquationSpec | Mapping[str, Any],
    variable_values: Mapping[str, Any] | Any,
    *,
    parameter_values: Mapping[str, Any] | None = None,
    constant_values: Mapping[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Evaluate a validated safe equation with torch operations.

    This evaluator uses the same allowlisted AST as the NumPy preview path, but
    returns a torch tensor directly so gradients can flow through tensor
    variables or parameters. It is intended for future PI-GPR mean-function
    conversion and lower-level research workflows, not JSON API payloads.
    """
    resolved = _require_valid_spec(spec)
    dtype, device = _infer_torch_options(
        variable_values,
        parameter_values=parameter_values,
        constant_values=constant_values,
        dtype=dtype,
        device=device,
    )
    context, row_count = _build_torch_context(
        resolved,
        variable_values,
        parameter_values=parameter_values,
        constant_values=constant_values,
        dtype=dtype,
        device=device,
    )
    tree = ast.parse(resolved.expression, mode="eval")
    result = _evaluate_torch_node(tree, context)
    if not isinstance(result, torch.Tensor):
        result = torch.as_tensor(result, dtype=dtype, device=device)
    else:
        result = result.to(dtype=dtype, device=device)
    if result.ndim == 0:
        result = result.reshape(()).expand(row_count)
    return result.reshape(-1)


def initialize_safe_equation_torch_parameters(
    spec: SafeEquationSpec | Mapping[str, Any],
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
    requires_grad: bool = True,
    include_fixed: bool = True,
) -> dict[str, torch.Tensor]:
    """Initialize declared equation parameters as torch tensors.

    Learnable parameters receive ``requires_grad=True`` by default. Parameters
    marked ``learnable=False`` are included as fixed tensors when
    ``include_fixed=True`` and never require gradients. The returned tensors can
    be passed directly to ``evaluate_safe_equation_torch(...)`` or wrapped by a
    downstream ``torch.nn.Module``/``ParameterDict``.
    """
    resolved = _require_valid_spec(spec)
    tensors: dict[str, torch.Tensor] = {}
    for parameter in resolved.parameters:
        if not parameter.learnable and not include_fixed:
            continue
        tensor = torch.tensor(
            parameter.initial_value,
            dtype=dtype,
            device=device,
            requires_grad=bool(requires_grad and parameter.learnable),
        )
        tensors[parameter.name] = tensor
    return tensors


def preview_safe_equation_mean(
    spec: SafeEquationSpec | Mapping[str, Any],
    variable_values: Mapping[str, Any] | Any,
    *,
    target_values: Any | None = None,
    parameter_values: Mapping[str, Any] | None = None,
    constant_values: Mapping[str, Any] | None = None,
) -> SafeEquationMeanPreview:
    """Preview physics-mean outputs and optional target residuals.

    This helper is designed for UI/API validation before model fitting. It
    reports input-row validity, finite physics-mean counts, compact summary
    statistics, and optional ``target - physics_mean`` residual summaries.
    """
    resolved = _resolve_spec_or_failure(spec)
    if isinstance(resolved, SafeEquationEvaluationResult):
        return SafeEquationMeanPreview(
            spec_name=resolved.spec_name,
            preview_ready=False,
            warnings=resolved.warnings,
            metadata=resolved.metadata,
        )

    input_quality = _input_row_quality(resolved, variable_values)
    evaluation = evaluate_safe_equation_numpy(
        resolved,
        variable_values,
        parameter_values=parameter_values,
        constant_values=constant_values,
    )
    mean_values = _values_to_numpy(evaluation.values)
    finite_mean_mask = np.isfinite(mean_values)
    finite_mean_count = int(np.count_nonzero(finite_mean_mask))
    nonfinite_mean_count = int(mean_values.size - finite_mean_count)
    warnings = [*input_quality["warnings"], *evaluation.warnings]

    target_residual_summary = None
    if target_values is not None:
        target_summary = _target_residual_summary(
            target_values=target_values,
            mean_values=mean_values,
        )
        warnings.extend(target_summary["warnings"])
        target_residual_summary = target_summary["summary"]

    preview_ready = (
        bool(evaluation.metadata.get("evaluation_ready", False))
        and input_quality["invalid_input_row_count"] == 0
        and nonfinite_mean_count == 0
    )
    if nonfinite_mean_count and not evaluation.warnings:
        warnings.append(f"physics mean contains {nonfinite_mean_count} non-finite value(s)")

    return SafeEquationMeanPreview(
        spec_name=resolved.name,
        preview_ready=preview_ready,
        valid_input_row_count=input_quality["valid_input_row_count"],
        invalid_input_row_count=input_quality["invalid_input_row_count"],
        finite_mean_count=finite_mean_count,
        nonfinite_mean_count=nonfinite_mean_count,
        physics_mean_summary=_summary_stats(mean_values),
        target_residual_summary=target_residual_summary,
        warnings=tuple(dict.fromkeys(warnings)),
        metadata={
            "evaluation": evaluation.to_dict(),
            "input_row_count": int(mean_values.size),
        },
    )


def safe_equation_schema_snapshot() -> dict[str, object]:
    """Return a JSON-safe snapshot of the current safe-equation schema surface."""
    return {
        "schema_version": SAFE_EQUATION_SCHEMA_VERSION,
        "allowed_functions": list(ALLOWED_SAFE_EQUATION_FUNCTIONS),
        "allowed_operators": list(ALLOWED_SAFE_EQUATION_OPERATORS),
        "identifier_rule": _IDENTIFIER_PATTERN.pattern,
        "records": {
            "SafeEquationSpec": {
                "required_fields": ["name", "expression", "variables"],
                "optional_fields": [
                    "parameters",
                    "constants",
                    "display_name",
                    "description",
                    "target_units",
                    "assumptions",
                    "validity_limits",
                    "references",
                    "metadata",
                ],
            },
            "SafeEquationVariable": {
                "required_fields": ["name"],
                "optional_fields": ["description", "units", "required", "bounds"],
            },
            "SafeEquationParameter": {
                "required_fields": ["name", "initial_value"],
                "optional_fields": ["bounds", "description", "units", "learnable"],
            },
            "SafeEquationConstant": {
                "required_fields": ["name", "value"],
                "optional_fields": ["description", "units"],
            },
        },
        "safety_rules": [
            "No arbitrary Python execution.",
            "Only declared variables, parameters, constants, and allowlisted functions are valid.",
            "Expression parsing is performed with ast.parse(..., mode='eval') and a manual allowlist.",
            "All numeric defaults and bounds must be finite.",
            "All payload metadata must be strict-JSON-safe.",
            "NumPy evaluation uses a manual AST interpreter, never eval().",
            "Non-finite evaluation outputs are represented as null values in JSON payloads.",
            "Torch evaluation uses a manual AST interpreter, never eval().",
            "Torch evaluation preserves gradients through tensor variables and parameters.",
            "PI-GPR mean conversion is handled by build_safe_equation_mean_function(...).",
        ],
    }


def _require_valid_spec(spec: SafeEquationSpec | Mapping[str, Any]) -> SafeEquationSpec:
    try:
        resolved = spec if isinstance(spec, SafeEquationSpec) else SafeEquationSpec.from_dict(spec)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid safe equation spec: {exc}") from exc
    validation = validate_safe_equation_spec(resolved)
    if not validation.is_valid:
        messages = "; ".join(_validation_warnings(validation))
        raise ValueError(f"safe equation spec is invalid: {messages}")
    return resolved


def _resolve_spec_or_failure(
    spec: SafeEquationSpec | Mapping[str, Any],
) -> SafeEquationSpec | SafeEquationEvaluationResult:
    try:
        return spec if isinstance(spec, SafeEquationSpec) else SafeEquationSpec.from_dict(spec)
    except (TypeError, ValueError, KeyError) as exc:
        return _evaluation_failure(
            spec_name=_safe_result_spec_name(spec),
            warnings=(f"invalid equation spec: {exc}",),
            metadata={"evaluation_ready": False},
        )


def _evaluation_failure(
    *,
    spec_name: str,
    warnings: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> SafeEquationEvaluationResult:
    return SafeEquationEvaluationResult(
        spec_name=_safe_result_spec_name(spec_name),
        values=(),
        finite_count=0,
        nonfinite_count=0,
        warnings=tuple(warnings),
        metadata={} if metadata is None else metadata,
    )


def _safe_result_spec_name(spec: SafeEquationSpec | Mapping[str, Any] | Any) -> str:
    raw_name = _spec_name_from_unknown(spec)
    if raw_name is None:
        raw_name = str(spec) if isinstance(spec, str) else "invalid_safe_equation"
    candidate = re.sub(r"\W+", "_", str(raw_name).strip())
    candidate = candidate.strip("_")
    if not candidate or candidate[0].isdigit() or candidate.startswith("__"):
        return "invalid_safe_equation"
    try:
        return _validate_symbol_name(candidate, "spec_name")
    except ValueError:
        return "invalid_safe_equation"


def _validation_warnings(validation: SafeEquationValidationResult) -> tuple[str, ...]:
    return tuple(
        f"{issue.code}: {issue.message}" for issue in validation.issues if issue.severity == "error"
    )


def _resolve_safe_equation_feature_indices(
    spec: SafeEquationSpec,
    *,
    feature_indices: Mapping[str, int] | None,
    column_indices: Mapping[str, int] | None,
) -> dict[str, int]:
    if feature_indices is not None and column_indices is not None:
        raise ValueError("Use either feature_indices or column_indices, not both")
    required_feature_names = _safe_equation_mean_feature_names(spec)
    declared = set(spec.variable_names)

    if feature_indices is None and column_indices is None:
        return {name: index for index, name in enumerate(required_feature_names)}

    resolved = dict(feature_indices if feature_indices is not None else column_indices or {})
    unknown = sorted(set(resolved).difference(declared))
    if unknown:
        raise ValueError(f"feature indices reference undeclared variables: {unknown}")
    missing = sorted(set(required_feature_names).difference(resolved))
    if missing:
        raise ValueError(f"feature indices are missing variables used by the equation: {missing}")
    for name, index in resolved.items():
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"feature index for {name!r} must be a non-negative integer")
    return resolved


def _safe_equation_mean_feature_names(spec: SafeEquationSpec) -> tuple[str, ...]:
    used = set(_used_safe_equation_variable_names(spec))
    if not used:
        return (spec.variable_names[0],)
    return tuple(name for name in spec.variable_names if name in used)


def _used_safe_equation_variable_names(spec: SafeEquationSpec) -> tuple[str, ...]:
    validation = validate_safe_equation_spec(spec)
    variable_names = set(spec.variable_names)
    return tuple(
        name
        for name in spec.variable_names
        if name in variable_names.intersection(validation.expression_symbols)
    )


def _safe_equation_parameter_maps(
    spec: SafeEquationSpec,
    *,
    parameter_values: Mapping[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    overrides = {} if parameter_values is None else dict(parameter_values)
    unknown = sorted(set(str(name) for name in overrides).difference(spec.parameter_names))
    if unknown:
        raise ValueError(f"parameter overrides reference undeclared parameters: {unknown}")

    learnable: dict[str, float] = {}
    fixed: dict[str, float] = {}
    for parameter in spec.parameters:
        value = overrides.get(parameter.name, parameter.initial_value)
        value = _finite_float(value, f"parameter override {parameter.name!r}")
        if parameter.bounds is not None and not parameter.bounds[0] <= value <= parameter.bounds[1]:
            raise ValueError(f"parameter override {parameter.name!r} must lie within bounds")
        if parameter.learnable:
            learnable[parameter.name] = value
        else:
            fixed[parameter.name] = value
    return learnable, fixed


def _resolve_safe_equation_positive_parameters(
    spec: SafeEquationSpec,
    *,
    learnable_parameters: Mapping[str, float],
    positive_parameters: Sequence[str] | None,
    infer_positive_parameters: bool,
) -> tuple[str, ...]:
    if positive_parameters is not None:
        requested = tuple(str(name) for name in positive_parameters)
    elif infer_positive_parameters:
        requested = tuple(
            parameter.name
            for parameter in spec.parameters
            if (
                parameter.learnable
                and parameter.bounds is not None
                and parameter.bounds[0] >= 0.0
                and learnable_parameters.get(parameter.name, 0.0) > 0.0
            )
        )
    else:
        requested = ()

    unknown = sorted(set(requested).difference(learnable_parameters))
    if unknown:
        raise ValueError(f"positive_parameters must refer to learnable parameters: {unknown}")
    non_positive = sorted(name for name in requested if learnable_parameters[name] <= 0.0)
    if non_positive:
        raise ValueError(f"positive parameter initial values must be > 0: {non_positive}")
    return requested


def _build_numpy_context(
    spec: SafeEquationSpec,
    variable_values: Mapping[str, Any] | Any,
    *,
    parameter_values: Mapping[str, Any] | None,
    constant_values: Mapping[str, Any] | None,
) -> dict[str, object]:
    warnings: list[str] = []
    variable_arrays, variable_warnings = _collect_variable_arrays(spec, variable_values)
    warnings.extend(variable_warnings)
    context: dict[str, object] = {}
    input_row_count = 0

    if not warnings:
        try:
            broadcasted = np.broadcast_arrays(*variable_arrays.values())
        except ValueError as exc:
            warnings.append(f"variable inputs cannot be broadcast together: {exc}")
            broadcasted = ()
        else:
            for name, array in zip(variable_arrays, broadcasted, strict=True):
                flat_array = np.asarray(array, dtype=float).reshape(-1)
                context[name] = flat_array
                input_row_count = int(flat_array.size)

    parameter_context, parameter_warnings = _numeric_context_values(
        defaults={parameter.name: parameter.initial_value for parameter in spec.parameters},
        overrides=parameter_values,
        allowed_names=spec.parameter_names,
        label="parameter",
    )
    constant_context, constant_warnings = _numeric_context_values(
        defaults={constant.name: constant.value for constant in spec.constants},
        overrides=constant_values,
        allowed_names=spec.constant_names,
        label="constant",
    )
    warnings.extend(parameter_warnings)
    warnings.extend(constant_warnings)
    context.update(parameter_context)
    context.update(constant_context)

    return {
        "context": context,
        "warnings": tuple(warnings),
        "input_row_count": input_row_count,
        "parameter_values": parameter_context,
        "constant_values": constant_context,
    }


def _numeric_context_values(
    *,
    defaults: Mapping[str, float],
    overrides: Mapping[str, Any] | None,
    allowed_names: Sequence[str],
    label: str,
) -> tuple[dict[str, float], tuple[str, ...]]:
    values = {name: float(value) for name, value in defaults.items()}
    warnings: list[str] = []
    if overrides is None:
        return values, ()
    if not isinstance(overrides, Mapping):
        return values, (f"{label}_values must be a mapping",)

    allowed = set(allowed_names)
    for name, value in overrides.items():
        name = str(name)
        if name not in allowed:
            warnings.append(f"unknown {label} override {name!r}")
            continue
        try:
            values[name] = _finite_float(value, f"{label} override {name!r}")
        except ValueError as exc:
            warnings.append(str(exc))
    return values, tuple(warnings)


def _infer_torch_options(
    variable_values: Mapping[str, Any] | Any,
    *,
    parameter_values: Mapping[str, Any] | None,
    constant_values: Mapping[str, Any] | None,
    dtype: torch.dtype | None,
    device: str | torch.device | None,
) -> tuple[torch.dtype, torch.device]:
    inferred_dtype = dtype
    inferred_device = torch.device("cpu") if device is None else torch.device(device)

    for value in _iter_possible_torch_values(
        variable_values,
        parameter_values,
        constant_values,
    ):
        if not isinstance(value, torch.Tensor):
            continue
        if dtype is None and value.dtype.is_floating_point:
            inferred_dtype = value.dtype
        if device is None:
            inferred_device = value.device
        if inferred_dtype is not None and device is not None:
            break

    return inferred_dtype or torch.float64, inferred_device


def _iter_possible_torch_values(*containers: Any):
    for container in containers:
        if container is None:
            continue
        if isinstance(container, torch.Tensor):
            yield container
            continue
        if isinstance(container, Mapping):
            yield from container.values()


def _build_torch_context(
    spec: SafeEquationSpec,
    variable_values: Mapping[str, Any] | Any,
    *,
    parameter_values: Mapping[str, Any] | None,
    constant_values: Mapping[str, Any] | None,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    variable_tensors = _collect_variable_tensors(
        spec,
        variable_values,
        dtype=dtype,
        device=device,
    )
    row_count = 1
    if variable_tensors:
        try:
            broadcasted = torch.broadcast_tensors(*variable_tensors.values())
        except RuntimeError as exc:
            raise ValueError(f"variable inputs cannot be broadcast together: {exc}") from exc
        variable_tensors = {
            name: tensor.reshape(-1)
            for name, tensor in zip(variable_tensors, broadcasted, strict=True)
        }
        row_count = len(next(iter(variable_tensors.values())))

    parameter_tensors = _torch_scalar_context_values(
        defaults={parameter.name: parameter.initial_value for parameter in spec.parameters},
        overrides=parameter_values,
        allowed_names=spec.parameter_names,
        label="parameter",
        dtype=dtype,
        device=device,
    )
    constant_tensors = _torch_scalar_context_values(
        defaults={constant.name: constant.value for constant in spec.constants},
        overrides=constant_values,
        allowed_names=spec.constant_names,
        label="constant",
        dtype=dtype,
        device=device,
    )
    return {**variable_tensors, **parameter_tensors, **constant_tensors}, row_count


def _collect_variable_tensors(
    spec: SafeEquationSpec,
    variable_values: Mapping[str, Any] | Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for variable in spec.variables:
        raw_value = _lookup_variable_value(variable_values, variable.name)
        if raw_value is None:
            if variable.required:
                raise ValueError(f"missing required variable input {variable.name!r}")
            continue
        tensor = _to_torch_tensor(
            raw_value,
            dtype=dtype,
            device=device,
            field_name=f"variable {variable.name!r}",
        )
        if tensor.ndim > 1:
            raise ValueError(f"variable {variable.name!r} must be scalar or one-dimensional")
        tensors[variable.name] = tensor.reshape(-1)
    return tensors


def _torch_scalar_context_values(
    *,
    defaults: Mapping[str, float],
    overrides: Mapping[str, Any] | None,
    allowed_names: Sequence[str],
    label: str,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    values = {
        name: torch.as_tensor(float(value), dtype=dtype, device=device)
        for name, value in defaults.items()
    }
    if overrides is None:
        return values
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{label}_values must be a mapping")

    allowed = set(allowed_names)
    for name, value in overrides.items():
        name = str(name)
        if name not in allowed:
            raise ValueError(f"unknown {label} override {name!r}")
        values[name] = _to_torch_scalar(
            value,
            dtype=dtype,
            device=device,
            field_name=f"{label} override {name!r}",
        )
    return values


def _to_torch_tensor(
    value: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
    field_name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    try:
        return torch.as_tensor(value, dtype=dtype, device=device)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} cannot be converted to a torch tensor: {exc}") from exc


def _to_torch_scalar(
    value: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
    field_name: str,
) -> torch.Tensor:
    tensor = _to_torch_tensor(value, dtype=dtype, device=device, field_name=field_name)
    if tensor.numel() != 1:
        raise ValueError(f"{field_name} must be a scalar tensor or finite numeric value")
    if not bool(torch.isfinite(tensor.detach()).all()):
        raise ValueError(f"{field_name} must be finite")
    return tensor.reshape(())


def _collect_variable_arrays(
    spec: SafeEquationSpec,
    variable_values: Mapping[str, Any] | Any,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    arrays: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    for variable in spec.variables:
        raw_value = _lookup_variable_value(variable_values, variable.name)
        if raw_value is None:
            if variable.required:
                warnings.append(f"missing required variable input {variable.name!r}")
            continue
        try:
            arrays[variable.name] = np.atleast_1d(np.asarray(raw_value, dtype=float))
        except (TypeError, ValueError) as exc:
            warnings.append(f"variable {variable.name!r} cannot be converted to float array: {exc}")
    return arrays, tuple(warnings)


def _lookup_variable_value(variable_values: Mapping[str, Any] | Any, name: str) -> Any | None:
    if isinstance(variable_values, Mapping):
        return variable_values.get(name)
    columns = getattr(variable_values, "columns", None)
    if columns is not None and name in columns:
        return variable_values[name]
    return None


def _input_row_quality(
    spec: SafeEquationSpec,
    variable_values: Mapping[str, Any] | Any,
) -> dict[str, object]:
    variable_arrays, warnings = _collect_variable_arrays(spec, variable_values)
    if warnings:
        return {
            "valid_input_row_count": 0,
            "invalid_input_row_count": 0,
            "warnings": warnings,
        }

    try:
        broadcasted = np.broadcast_arrays(*variable_arrays.values())
    except ValueError as exc:
        return {
            "valid_input_row_count": 0,
            "invalid_input_row_count": 0,
            "warnings": (f"variable inputs cannot be broadcast together: {exc}",),
        }

    finite_masks = [
        np.isfinite(np.asarray(array, dtype=float).reshape(-1)) for array in broadcasted
    ]
    row_finite = np.logical_and.reduce(finite_masks)
    invalid_count = int(np.count_nonzero(~row_finite))
    warnings_list = []
    if invalid_count:
        warnings_list.append(f"{invalid_count} input row(s) contain non-finite variable values")
    return {
        "valid_input_row_count": int(np.count_nonzero(row_finite)),
        "invalid_input_row_count": invalid_count,
        "warnings": tuple(warnings_list),
    }


def _evaluate_numpy_node(node: ast.AST, context: Mapping[str, object]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate_numpy_node(node.body, context)
    if isinstance(node, ast.BinOp):
        left = _evaluate_numpy_node(node.left, context)
        right = _evaluate_numpy_node(node.right, context)
        if isinstance(node.op, ast.Add):
            return np.add(left, right)
        if isinstance(node.op, ast.Sub):
            return np.subtract(left, right)
        if isinstance(node.op, ast.Mult):
            return np.multiply(left, right)
        if isinstance(node.op, ast.Div):
            return np.divide(left, right)
        if isinstance(node.op, ast.Pow):
            return np.power(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_numpy_node(node.operand, context)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return np.negative(operand)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("function calls must use direct allowlisted function names")
        arguments = [_evaluate_numpy_node(argument, context) for argument in node.args]
        return _evaluate_numpy_function(node.func.id, arguments)
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"symbol {node.id!r} is missing from evaluation context")
        return context[node.id]
    if isinstance(node, ast.Constant):
        return _finite_float(node.value, "expression constant")
    raise ValueError(f"expression syntax {type(node).__name__} is not supported")


def _evaluate_numpy_function(function_name: str, arguments: Sequence[Any]) -> Any:
    if function_name == "abs":
        return np.abs(arguments[0])
    if function_name == "exp":
        return np.exp(arguments[0])
    if function_name == "log":
        return np.log(arguments[0])
    if function_name == "sqrt":
        return np.sqrt(arguments[0])
    if function_name == "pow":
        return np.power(arguments[0], arguments[1])
    if function_name == "min":
        return _numpy_reduce(arguments, np.minimum)
    if function_name == "max":
        return _numpy_reduce(arguments, np.maximum)
    raise ValueError(f"function {function_name!r} is not supported")


def _evaluate_torch_node(node: ast.AST, context: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(node, ast.Expression):
        return _evaluate_torch_node(node.body, context)
    if isinstance(node, ast.BinOp):
        left = _evaluate_torch_node(node.left, context)
        right = _evaluate_torch_node(node.right, context)
        if isinstance(node.op, ast.Add):
            return torch.add(left, right)
        if isinstance(node.op, ast.Sub):
            return torch.sub(left, right)
        if isinstance(node.op, ast.Mult):
            return torch.mul(left, right)
        if isinstance(node.op, ast.Div):
            return torch.div(left, right)
        if isinstance(node.op, ast.Pow):
            return torch.pow(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_torch_node(node.operand, context)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return torch.neg(operand)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("function calls must use direct allowlisted function names")
        arguments = [_evaluate_torch_node(argument, context) for argument in node.args]
        return _evaluate_torch_function(node.func.id, arguments)
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"symbol {node.id!r} is missing from evaluation context")
        return context[node.id]
    if isinstance(node, ast.Constant):
        reference = next(iter(context.values()), None)
        dtype = torch.float64 if reference is None else reference.dtype
        device = torch.device("cpu") if reference is None else reference.device
        return torch.as_tensor(
            _finite_float(node.value, "expression constant"), dtype=dtype, device=device
        )
    raise ValueError(f"expression syntax {type(node).__name__} is not supported")


def _evaluate_torch_function(
    function_name: str,
    arguments: Sequence[torch.Tensor],
) -> torch.Tensor:
    if function_name == "abs":
        return torch.abs(arguments[0])
    if function_name == "exp":
        return torch.exp(arguments[0])
    if function_name == "log":
        return torch.log(arguments[0])
    if function_name == "sqrt":
        return torch.sqrt(arguments[0])
    if function_name == "pow":
        return torch.pow(arguments[0], arguments[1])
    if function_name == "min":
        return _torch_reduce(arguments, torch.minimum)
    if function_name == "max":
        return _torch_reduce(arguments, torch.maximum)
    raise ValueError(f"function {function_name!r} is not supported")


def _torch_reduce(
    arguments: Sequence[torch.Tensor],
    function: Any,
) -> torch.Tensor:
    result = arguments[0]
    for argument in arguments[1:]:
        result = function(result, argument)
    return result


def _numpy_reduce(arguments: Sequence[Any], function: Any) -> Any:
    result = arguments[0]
    for argument in arguments[1:]:
        result = function(result, argument)
    return result


def _json_safe_array_values(values: np.ndarray) -> tuple[float | None, ...]:
    flat_values = np.ravel(np.asarray(values, dtype=float))
    return tuple(float(value) if math.isfinite(float(value)) else None for value in flat_values)


def _values_to_numpy(values: Sequence[float | None]) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)


def _summary_stats(values: np.ndarray) -> dict[str, object]:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "max": None,
        }
    return {
        "count": int(finite_values.size),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "min": float(np.min(finite_values)),
        "median": float(np.median(finite_values)),
        "max": float(np.max(finite_values)),
    }


def _target_residual_summary(
    *,
    target_values: Any,
    mean_values: np.ndarray,
) -> dict[str, object]:
    warnings: list[str] = []
    try:
        target_array = np.asarray(target_values, dtype=float)
    except (TypeError, ValueError) as exc:
        return {
            "summary": None,
            "warnings": (f"target values cannot be converted to float array: {exc}",),
        }

    if target_array.ndim == 0:
        target_array = np.full(mean_values.size, float(target_array), dtype=float)
    target_array = target_array.reshape(-1)
    if target_array.size != mean_values.size:
        return {
            "summary": None,
            "warnings": ("target value count does not match evaluated physics-mean count",),
        }

    finite_mask = np.isfinite(target_array) & np.isfinite(mean_values)
    if not np.all(finite_mask):
        invalid_count = int(target_array.size - np.count_nonzero(finite_mask))
        warnings.append(f"{invalid_count} row(s) were excluded from target residual summary")
    residuals = target_array[finite_mask] - mean_values[finite_mask]
    summary = _summary_stats(residuals)
    if residuals.size:
        summary["mean_absolute_residual"] = float(np.mean(np.abs(residuals)))
    else:
        summary["mean_absolute_residual"] = None
    return {"summary": summary, "warnings": tuple(warnings)}


def _validate_expression(
    spec: SafeEquationSpec,
) -> tuple[list[SafeEquationValidationIssue], set[str], set[str]]:
    issues: list[SafeEquationValidationIssue] = []
    expression_symbols: set[str] = set()
    expression_functions: set[str] = set()
    try:
        tree = ast.parse(spec.expression, mode="eval")
    except SyntaxError as exc:
        issues.append(
            _expression_issue(
                code="syntax_error",
                message=f"expression syntax error: {exc.msg}",
            )
        )
        return issues, expression_symbols, expression_functions

    _validate_expression_node(
        tree,
        declared_symbols=set(spec.declared_symbols),
        issues=issues,
        expression_symbols=expression_symbols,
        expression_functions=expression_functions,
    )
    return issues, expression_symbols, expression_functions


def _validate_expression_node(
    node: ast.AST,
    *,
    declared_symbols: set[str],
    issues: list[SafeEquationValidationIssue],
    expression_symbols: set[str],
    expression_functions: set[str],
) -> None:
    if isinstance(node, ast.Expression):
        _validate_expression_node(
            node.body,
            declared_symbols=declared_symbols,
            issues=issues,
            expression_symbols=expression_symbols,
            expression_functions=expression_functions,
        )
        return

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINARY_OPERATORS:
            issues.append(
                _expression_issue(
                    code="unsupported_operator",
                    message=f"operator {type(node.op).__name__} is not allowed",
                )
            )
        _validate_expression_node(
            node.left,
            declared_symbols=declared_symbols,
            issues=issues,
            expression_symbols=expression_symbols,
            expression_functions=expression_functions,
        )
        _validate_expression_node(
            node.right,
            declared_symbols=declared_symbols,
            issues=issues,
            expression_symbols=expression_symbols,
            expression_functions=expression_functions,
        )
        return

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARY_OPERATORS:
            issues.append(
                _expression_issue(
                    code="unsupported_operator",
                    message=f"unary operator {type(node.op).__name__} is not allowed",
                )
            )
        _validate_expression_node(
            node.operand,
            declared_symbols=declared_symbols,
            issues=issues,
            expression_symbols=expression_symbols,
            expression_functions=expression_functions,
        )
        return

    if isinstance(node, ast.Call):
        _validate_function_call(
            node,
            issues=issues,
            expression_functions=expression_functions,
        )
        for argument in node.args:
            _validate_expression_node(
                argument,
                declared_symbols=declared_symbols,
                issues=issues,
                expression_symbols=expression_symbols,
                expression_functions=expression_functions,
            )
        return

    if isinstance(node, ast.Name):
        if node.id not in declared_symbols:
            issues.append(
                _expression_issue(
                    code="unknown_symbol",
                    message=f"symbol {node.id!r} is not declared in variables, parameters, or constants",
                )
            )
        else:
            expression_symbols.add(node.id)
        return

    if isinstance(node, ast.Constant):
        _validate_numeric_constant(node.value, issues=issues)
        return

    issues.append(
        _expression_issue(
            code="unsupported_syntax",
            message=f"expression syntax {type(node).__name__} is not allowed",
        )
    )


def _validate_function_call(
    node: ast.Call,
    *,
    issues: list[SafeEquationValidationIssue],
    expression_functions: set[str],
) -> None:
    if node.keywords:
        issues.append(
            _expression_issue(
                code="unsupported_syntax",
                message="function keyword arguments are not allowed",
            )
        )

    if not isinstance(node.func, ast.Name):
        issues.append(
            _expression_issue(
                code="unsupported_syntax",
                message="function calls must use direct allowlisted function names",
            )
        )
        return

    function_name = node.func.id
    if function_name not in ALLOWED_SAFE_EQUATION_FUNCTIONS:
        issues.append(
            _expression_issue(
                code="unsupported_function",
                message=f"function {function_name!r} is not allowlisted",
            )
        )
        return

    expression_functions.add(function_name)
    min_args, max_args = _ALLOWED_FUNCTION_ARITY[function_name]
    argument_count = len(node.args)
    if argument_count < min_args or (max_args is not None and argument_count > max_args):
        if max_args is None:
            expected = f"at least {min_args}"
        elif min_args == max_args:
            expected = str(min_args)
        else:
            expected = f"{min_args} to {max_args}"
        issues.append(
            _expression_issue(
                code="invalid_function_arity",
                message=(
                    f"function {function_name!r} expects {expected} positional "
                    f"arguments but received {argument_count}"
                ),
            )
        )


def _validate_numeric_constant(
    value: Any,
    *,
    issues: list[SafeEquationValidationIssue],
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        issues.append(
            _expression_issue(
                code="unsupported_constant",
                message="only finite numeric expression constants are allowed",
            )
        )
        return
    if not math.isfinite(float(value)):
        issues.append(
            _expression_issue(
                code="nonfinite_constant",
                message="expression constants must be finite",
            )
        )


def _expression_issue(code: str, message: str) -> SafeEquationValidationIssue:
    return SafeEquationValidationIssue(
        code=code,
        message=message,
        field="expression",
        severity="error",
    )


def _coerce_variable(value: SafeEquationVariable | Mapping[str, Any]) -> SafeEquationVariable:
    if isinstance(value, SafeEquationVariable):
        return value
    if isinstance(value, Mapping):
        return SafeEquationVariable.from_dict(value)
    raise ValueError("variables must contain SafeEquationVariable records or mappings")


def _coerce_parameter(value: SafeEquationParameter | Mapping[str, Any]) -> SafeEquationParameter:
    if isinstance(value, SafeEquationParameter):
        return value
    if isinstance(value, Mapping):
        return SafeEquationParameter.from_dict(value)
    raise ValueError("parameters must contain SafeEquationParameter records or mappings")


def _coerce_constant(value: SafeEquationConstant | Mapping[str, Any]) -> SafeEquationConstant:
    if isinstance(value, SafeEquationConstant):
        return value
    if isinstance(value, Mapping):
        return SafeEquationConstant.from_dict(value)
    raise ValueError("constants must contain SafeEquationConstant records or mappings")


def _coerce_issue(
    value: SafeEquationValidationIssue | Mapping[str, Any],
) -> SafeEquationValidationIssue:
    if isinstance(value, SafeEquationValidationIssue):
        return value
    if isinstance(value, Mapping):
        return SafeEquationValidationIssue(
            code=_required(value, "code", "SafeEquationValidationIssue"),
            message=_required(value, "message", "SafeEquationValidationIssue"),
            field=value.get("field"),
            severity=value.get("severity", "error"),
        )
    raise ValueError("issues must contain SafeEquationValidationIssue records or mappings")


def _validate_unique_symbols(
    variables: Sequence[SafeEquationVariable],
    parameters: Sequence[SafeEquationParameter],
    constants: Sequence[SafeEquationConstant],
) -> None:
    names = [record.name for record in (*variables, *parameters, *constants)]
    reserved = sorted(name for name in names if name in _RESERVED_NAMES)
    if reserved:
        raise ValueError(f"declared symbols cannot use reserved function names: {reserved}")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"declared symbols must be unique; duplicate names: {duplicates}")


def _validate_symbol_name(value: Any, field_name: str) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError(f"{field_name} must be non-empty")
    if name.startswith("__") or not _IDENTIFIER_PATTERN.match(name):
        raise ValueError(
            f"{field_name} must be a conservative identifier matching "
            f"{_IDENTIFIER_PATTERN.pattern!r}"
        )
    return name


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite numeric value")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite")
    return numeric_value


def _optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field_name)


def _validate_bounds(values: Sequence[Any] | None, field_name: str) -> tuple[float, float] | None:
    if values is None:
        return None
    if isinstance(values, str | bytes) or len(values) != 2:
        raise ValueError(f"{field_name} must contain exactly two numeric values")
    lower = _finite_float(values[0], f"{field_name}[0]")
    upper = _finite_float(values[1], f"{field_name}[1]")
    if lower >= upper:
        raise ValueError(f"{field_name} must satisfy lower < upper")
    return lower, upper


def _bounds_to_list(bounds: tuple[float, float] | None) -> list[float] | None:
    return None if bounds is None else [bounds[0], bounds[1]]


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text and field_name.endswith("name"):
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _string_tuple(values: Sequence[Any], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise ValueError(f"{field_name} must be a sequence, not a string")
    return tuple(str(value) for value in values)


def _json_safe_mapping(values: Mapping[str, Any], field_name: str) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return {
        str(key): _json_safe_value(value, f"{field_name}.{key}") for key, value in values.items()
    }


def _json_safe_value(value: Any, field_name: str) -> object:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Real):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{field_name} must be finite for strict JSON serialization")
        return int(value) if isinstance(value, int) else numeric_value
    if isinstance(value, Mapping):
        return _json_safe_mapping(value, field_name)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_safe_value(item, f"{field_name}[]") for item in value]
    raise ValueError(f"{field_name} contains a non-JSON-safe value of type {type(value).__name__}")


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return int(value)


def _required(record: Mapping[str, Any], key: str, record_name: str) -> Any:
    if key not in record:
        raise ValueError(f"{record_name} requires field {key!r}")
    return record[key]


def _spec_name_from_unknown(spec: SafeEquationSpec | Mapping[str, Any]) -> str | None:
    if isinstance(spec, SafeEquationSpec):
        return spec.name
    if isinstance(spec, Mapping) and "name" in spec:
        return str(spec["name"])
    return None
