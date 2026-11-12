from __future__ import annotations

import json
import os
from collections.abc import Callable

import numpy as np
import pytest
import torch

from matgpr.safe_equations import (
    SafeEquationConstant,
    SafeEquationParameter,
    SafeEquationSpec,
    SafeEquationVariable,
    build_safe_equation_mean_function,
    evaluate_safe_equation_numpy,
    evaluate_safe_equation_torch,
    initialize_safe_equation_torch_parameters,
    safe_equation_schema_snapshot,
    validate_safe_equation_spec,
)

pytestmark = pytest.mark.contract


def _spec(expression: str) -> SafeEquationSpec:
    return SafeEquationSpec(
        name="security_characterization",
        expression=expression,
        variables=(
            SafeEquationVariable("x"),
            SafeEquationVariable("y", required=False),
            SafeEquationVariable("z", required=False),
        ),
    )


@pytest.mark.parametrize(
    "expression",
    [
        "x.real",
        "x[0]",
        "x if x else 0",
        "x < 1",
        "x and y",
        "(lambda value: value)(x)",
        "[value for value in x]",
        "(value for value in x)",
        "{'value': x}",
        "{x}",
        "[x]",
        "(x,)",
        "(value := x)",
        "f'{x}'",
        "max(*x)",
        "pow(x, exponent=2)",
        "__import__('os').system('echo unsafe')",
        "open('unsafe.txt')",
    ],
)
def test_validator_rejects_every_non_allowlisted_expression_shape(
    expression: str,
) -> None:
    validation = validate_safe_equation_spec(_spec(expression))

    assert not validation.is_valid
    assert validation.errors
    assert all(issue.field == "expression" for issue in validation.errors)
    json.dumps(validation.to_dict(), allow_nan=False)


def test_injection_shaped_expression_is_never_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(os, "system", lambda command: calls.append(command) or 0)
    spec = _spec("__import__('os').system('echo unsafe')")

    validation = validate_safe_equation_spec(spec)
    evaluation = evaluate_safe_equation_numpy(spec, {"x": [1.0]})

    assert not validation.is_valid
    assert evaluation.values == ()
    assert calls == []


def test_validator_rejects_expression_above_character_limit() -> None:
    expression = "+".join("x" for _ in range(2_100))

    validation = validate_safe_equation_spec(_spec(expression))

    assert not validation.is_valid
    assert validation.errors[0].code == "expression_too_long"


def test_validator_rejects_expression_above_ast_node_limit() -> None:
    expression = f"max({','.join('x' for _ in range(300))})"

    validation = validate_safe_equation_spec(_spec(expression))

    assert not validation.is_valid
    assert validation.errors[0].code == "expression_too_complex"


def test_validator_rejects_expression_above_ast_depth_limit() -> None:
    expression = "-" * 40 + "x"

    validation = validate_safe_equation_spec(_spec(expression))

    assert not validation.is_valid
    assert validation.errors[0].code == "expression_too_deep"


def test_validator_accepts_nontrivial_expression_within_complexity_limits() -> None:
    expression = "-" * 20 + "max(abs(x), min(y, z))"

    validation = validate_safe_equation_spec(_spec(expression))

    assert validation.is_valid
    assert validation.expression_functions == ("abs", "max", "min")


def test_validator_reports_oversized_integer_literal_without_raising() -> None:
    validation = validate_safe_equation_spec(_spec("9" * 309))

    assert not validation.is_valid
    assert validation.errors[0].code == "nonfinite_constant"


def test_validator_reports_malformed_unicode_without_raising() -> None:
    validation = validate_safe_equation_spec(_spec("\ud800"))

    assert not validation.is_valid
    assert validation.errors[0].code == "syntax_error"


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: SafeEquationParameter("scale", initial_value=value),
        lambda value: SafeEquationConstant("scale", value=value),
        lambda value: SafeEquationVariable("x", bounds=(0, value)),
    ],
)
def test_schema_numeric_fields_normalize_oversized_values_to_validation_errors(
    factory: Callable[[int], object],
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        factory(10**1_000)


def test_metadata_rejects_oversized_numeric_values() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        SafeEquationSpec(
            name="oversized_metadata",
            expression="x",
            variables=(SafeEquationVariable("x"),),
            metadata={"oversized": 10**1_000},
        )


def test_metadata_rejects_cyclic_containers() -> None:
    metadata: dict[str, object] = {}
    metadata["cycle"] = metadata

    with pytest.raises(ValueError, match="cyclic reference"):
        SafeEquationSpec(
            name="cyclic_metadata",
            expression="x",
            variables=(SafeEquationVariable("x"),),
            metadata=metadata,
        )


def test_metadata_rejects_excessive_nesting() -> None:
    metadata: dict[str, object] = {}
    cursor = metadata
    for index in range(40):
        child: dict[str, object] = {}
        cursor[str(index)] = child
        cursor = child

    with pytest.raises(ValueError, match="nesting depth"):
        SafeEquationSpec(
            name="deep_metadata",
            expression="x",
            variables=(SafeEquationVariable("x"),),
            metadata=metadata,
        )


def test_schema_snapshot_discloses_enforced_security_limits() -> None:
    snapshot = safe_equation_schema_snapshot()

    assert snapshot["expression_limits"] == {
        "max_characters": 4_096,
        "max_ast_nodes": 256,
        "max_ast_depth": 32,
    }
    assert snapshot["metadata_limits"] == {"max_nesting_depth": 32}
    json.dumps(snapshot, allow_nan=False)


def test_complex_expression_is_rejected_before_either_evaluator_runs() -> None:
    spec = _spec(f"max({','.join('x' for _ in range(300))})")

    numpy_result = evaluate_safe_equation_numpy(spec, {"x": [1.0]})

    assert numpy_result.values == ()
    assert numpy_result.metadata["evaluation_ready"] is False
    assert numpy_result.warnings[0].startswith("expression_too_complex:")
    with pytest.raises(ValueError, match="expression_too_complex"):
        evaluate_safe_equation_torch(spec, {"x": torch.tensor([1.0])})


def test_numpy_and_torch_interpreters_agree_across_allowlisted_grammar() -> None:
    spec = SafeEquationSpec(
        name="backend_parity",
        expression=(
            "offset + abs(x) + exp(y) - log(z) + sqrt(z) "
            "+ pow(z, exponent) + min(x, y, z) + max(x, y, z)"
        ),
        variables=(
            SafeEquationVariable("x"),
            SafeEquationVariable("y"),
            SafeEquationVariable("z"),
        ),
        parameters=(
            SafeEquationParameter("offset", initial_value=-0.25, learnable=False),
            SafeEquationParameter("exponent", initial_value=2.0),
        ),
    )
    numpy_inputs = {
        "x": np.array([-2.0, 1.0]),
        "y": np.array([0.5, 0.25]),
        "z": np.array([1.0, 4.0]),
    }
    torch_inputs = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in numpy_inputs.items()
    }
    parameters = initialize_safe_equation_torch_parameters(spec)

    numpy_result = evaluate_safe_equation_numpy(spec, numpy_inputs)
    torch_result = evaluate_safe_equation_torch(
        spec,
        torch_inputs,
        parameter_values=parameters,
    )
    torch_result.sum().backward()

    np.testing.assert_allclose(torch_result.detach().numpy(), numpy_result.values)
    assert parameters["exponent"].grad is not None
    assert all(value.grad is not None for value in torch_inputs.values())


def test_scalar_expression_expands_to_input_row_count_in_both_backends() -> None:
    spec = SafeEquationSpec(
        name="constant_backend_parity",
        expression="offset",
        variables=(SafeEquationVariable("x"),),
        parameters=(SafeEquationParameter("offset", initial_value=3.5),),
    )

    numpy_result = evaluate_safe_equation_numpy(spec, {"x": [1.0, 2.0, 3.0]})
    torch_result = evaluate_safe_equation_torch(
        spec,
        {"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)},
    )

    assert numpy_result.values == (3.5, 3.5, 3.5)
    torch.testing.assert_close(torch_result, torch.full((3,), 3.5, dtype=torch.float64))


@pytest.mark.parametrize("expression", ["log(x)", "sqrt(x)", "1 / x"])
def test_torch_evaluator_rejects_nonfinite_outputs(expression: str) -> None:
    spec = _spec(expression)
    inputs = {"x": torch.tensor([-1.0, 0.0], dtype=torch.float64)}

    with pytest.raises(ValueError, match="equation produced .* non-finite output"):
        evaluate_safe_equation_torch(spec, inputs)


def test_mean_adapter_rejects_nonfinite_physics_mean_before_gpr_training() -> None:
    spec = SafeEquationSpec(
        name="nonfinite_mean",
        expression="log(x)",
        variables=(SafeEquationVariable("x"),),
    )
    mean = build_safe_equation_mean_function(spec, feature_indices={"x": 0})

    with pytest.raises(ValueError, match="equation produced .* non-finite output"):
        mean(torch.tensor([[-1.0], [1.0]], dtype=torch.float64))
