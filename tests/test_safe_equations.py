from __future__ import annotations

import json
import math
import unittest

import numpy as np
import torch

from matgpr import (
    ALLOWED_SAFE_EQUATION_FUNCTIONS,
    ALLOWED_SAFE_EQUATION_OPERATORS,
    SAFE_EQUATION_SCHEMA_VERSION,
    SafeEquationConstant,
    SafeEquationEvaluationResult,
    SafeEquationMeanFunction,
    SafeEquationMeanPreview,
    SafeEquationParameter,
    SafeEquationSpec,
    SafeEquationValidationIssue,
    SafeEquationValidationResult,
    SafeEquationVariable,
    build_safe_equation_mean_function,
    evaluate_safe_equation_numpy,
    evaluate_safe_equation_torch,
    fit_gpytorch_gpr,
    initialize_safe_equation_torch_parameters,
    preview_safe_equation_mean,
    safe_equation_schema_snapshot,
    validate_safe_equation_spec,
)


def _arrhenius_like_spec() -> SafeEquationSpec:
    return SafeEquationSpec(
        name="custom_arrhenius_rate",
        display_name="Custom Arrhenius Rate",
        expression="prefactor * exp(-activation_energy / (R * temperature_k))",
        variables=[
            SafeEquationVariable(
                name="temperature_k",
                description="Absolute temperature",
                units="K",
                bounds=(1.0, 3000.0),
            )
        ],
        parameters=[
            SafeEquationParameter(
                name="prefactor",
                initial_value=1.0,
                bounds=(1e-12, 1e12),
                units="target units",
            ),
            SafeEquationParameter(
                name="activation_energy",
                initial_value=50000.0,
                bounds=(0.0, 300000.0),
                units="J/mol",
            ),
        ],
        constants=[
            SafeEquationConstant(
                name="R",
                value=8.31446261815324,
                units="J/mol/K",
            )
        ],
        description="Arrhenius-style rate expression for schema validation.",
        target_units="rate",
        assumptions=("single apparent activation barrier",),
        validity_limits={"temperature_k": [250.0, 1200.0]},
        references=("example only",),
        metadata={"source": "unit-test", "version": 1},
    )


class SafeEquationSchemaTests(unittest.TestCase):
    def test_safe_equation_spec_to_dict_is_strict_json_safe(self):
        spec = _arrhenius_like_spec()
        record = spec.to_dict()
        validation = validate_safe_equation_spec(spec)

        self.assertEqual(record["schema_version"], SAFE_EQUATION_SCHEMA_VERSION)
        self.assertEqual(spec.variable_names, ("temperature_k",))
        self.assertEqual(spec.parameter_names, ("prefactor", "activation_energy"))
        self.assertEqual(spec.constant_names, ("R",))
        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.declared_symbols, spec.declared_symbols)
        self.assertEqual(validation.expression_functions, ("exp",))
        self.assertEqual(
            validation.expression_symbols,
            ("R", "activation_energy", "prefactor", "temperature_k"),
        )
        json.dumps(record, allow_nan=False)
        json.dumps(validation.to_dict(), allow_nan=False)

    def test_safe_equation_spec_can_round_trip_from_mapping(self):
        record = _arrhenius_like_spec().to_dict()
        spec = SafeEquationSpec.from_dict(record)

        self.assertEqual(spec.name, "custom_arrhenius_rate")
        self.assertEqual(spec.variables[0].bounds, (1.0, 3000.0))
        self.assertEqual(spec.parameters[1].initial_value, 50000.0)
        self.assertTrue(validate_safe_equation_spec(record).is_valid)

    def test_safe_equation_spec_rejects_invalid_identifier(self):
        with self.assertRaisesRegex(ValueError, "conservative identifier"):
            SafeEquationVariable(name="temperature-k")

        with self.assertRaisesRegex(ValueError, "conservative identifier"):
            SafeEquationSpec(
                name="__private",
                expression="x",
                variables=[SafeEquationVariable("x")],
            )

    def test_safe_equation_spec_rejects_reserved_function_names(self):
        with self.assertRaisesRegex(ValueError, "reserved function names"):
            SafeEquationSpec(
                name="bad_reserved",
                expression="exp + coefficient",
                variables=[SafeEquationVariable("exp")],
                parameters=[SafeEquationParameter("coefficient", initial_value=1.0)],
            )

    def test_safe_equation_spec_rejects_duplicate_symbols(self):
        with self.assertRaisesRegex(ValueError, "duplicate names"):
            SafeEquationSpec(
                name="bad_duplicate",
                expression="x + x",
                variables=[SafeEquationVariable("x")],
                parameters=[SafeEquationParameter("x", initial_value=1.0)],
            )

    def test_parameter_bounds_validation(self):
        with self.assertRaisesRegex(ValueError, "lower < upper"):
            SafeEquationParameter("coefficient", initial_value=1.0, bounds=(2.0, 1.0))

        with self.assertRaisesRegex(ValueError, "within bounds"):
            SafeEquationParameter("coefficient", initial_value=3.0, bounds=(0.0, 2.0))

    def test_validate_safe_equation_spec_returns_errors_for_bad_mapping(self):
        result = validate_safe_equation_spec(
            {
                "name": "bad_mapping",
                "expression": "coefficient * x",
                "variables": [{"name": "x"}],
                "parameters": [{"name": "coefficient", "initial_value": float("nan")}],
            }
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].code, "invalid_spec")
        self.assertIn("must be finite", result.errors[0].message)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_metadata_must_be_json_safe(self):
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            SafeEquationSpec(
                name="bad_metadata",
                expression="x",
                variables=[SafeEquationVariable("x")],
                metadata={"bad": float("inf")},
            )

    def test_schema_snapshot_is_json_safe(self):
        snapshot = safe_equation_schema_snapshot()

        self.assertIn("exp", ALLOWED_SAFE_EQUATION_FUNCTIONS)
        self.assertIn("**", ALLOWED_SAFE_EQUATION_OPERATORS)
        self.assertIn("allowed_operators", snapshot)
        self.assertIn("SafeEquationSpec", snapshot["records"])
        json.dumps(snapshot, allow_nan=False)

    def test_parser_accepts_allowed_math_expression_family(self):
        spec = SafeEquationSpec(
            name="custom_hall_petch_power",
            expression="base + coefficient / sqrt(grain_size) + pow(driving_force, exponent)",
            variables=[
                SafeEquationVariable("grain_size"),
                SafeEquationVariable("driving_force"),
            ],
            parameters=[
                SafeEquationParameter("base", initial_value=1.0),
                SafeEquationParameter("coefficient", initial_value=2.0),
                SafeEquationParameter("exponent", initial_value=0.5),
            ],
        )

        result = validate_safe_equation_spec(spec)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.expression_functions, ("pow", "sqrt"))
        self.assertEqual(
            result.expression_symbols,
            ("base", "coefficient", "driving_force", "exponent", "grain_size"),
        )
        json.dumps(result.to_dict(), allow_nan=False)

    def test_parser_accepts_min_max_abs_and_unary_operators(self):
        spec = SafeEquationSpec(
            name="custom_clipped_residual",
            expression="-offset + max(abs(x), min(y, threshold))",
            variables=[SafeEquationVariable("x"), SafeEquationVariable("y")],
            parameters=[
                SafeEquationParameter("offset", initial_value=0.0),
                SafeEquationParameter("threshold", initial_value=1.0),
            ],
        )

        result = validate_safe_equation_spec(spec)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.expression_functions, ("abs", "max", "min"))

    def test_parser_reports_syntax_errors(self):
        result = validate_safe_equation_spec(
            {
                "name": "bad_syntax",
                "expression": "coefficient *",
                "variables": [{"name": "x"}],
                "parameters": [{"name": "coefficient", "initial_value": 1.0}],
            }
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].code, "syntax_error")

    def test_parser_reports_unknown_symbols(self):
        result = validate_safe_equation_spec(
            SafeEquationSpec(
                name="bad_symbol",
                expression="coefficient * missing_x",
                variables=[SafeEquationVariable("x")],
                parameters=[SafeEquationParameter("coefficient", initial_value=1.0)],
            )
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].code, "unknown_symbol")
        self.assertIn("missing_x", result.errors[0].message)

    def test_parser_reports_unsupported_functions_and_arity(self):
        unsupported = validate_safe_equation_spec(
            SafeEquationSpec(
                name="bad_function",
                expression="sin(x)",
                variables=[SafeEquationVariable("x")],
            )
        )
        bad_arity = validate_safe_equation_spec(
            SafeEquationSpec(
                name="bad_arity",
                expression="pow(x)",
                variables=[SafeEquationVariable("x")],
            )
        )

        self.assertFalse(unsupported.is_valid)
        self.assertEqual(unsupported.errors[0].code, "unsupported_function")
        self.assertFalse(bad_arity.is_valid)
        self.assertEqual(bad_arity.errors[0].code, "invalid_function_arity")

    def test_parser_rejects_unsafe_expression_shapes(self):
        examples = {
            "attribute_call": "__import__('os').system('echo no')",
            "subscript": "x[0]",
            "conditional": "x if x > 0 else 0",
            "comparison": "x > 0",
            "keyword": "pow(x, y=2)",
            "string_constant": "'not numeric'",
        }

        for name, expression in examples.items():
            with self.subTest(name=name):
                result = validate_safe_equation_spec(
                    {
                        "name": name,
                        "expression": expression,
                        "variables": [{"name": "x"}],
                    }
                )
                issue_codes = {issue.code for issue in result.errors}

                self.assertFalse(result.is_valid)
                self.assertTrue(
                    issue_codes.intersection(
                        {
                            "unsupported_syntax",
                            "unsupported_constant",
                            "unsupported_function",
                            "invalid_function_arity",
                        }
                    )
                )

    def test_parser_rejects_nonfinite_expression_constants(self):
        result = validate_safe_equation_spec(
            SafeEquationSpec(
                name="bad_nonfinite_constant",
                expression="1e999 * x",
                variables=[SafeEquationVariable("x")],
            )
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].code, "nonfinite_constant")

    def test_numpy_evaluator_returns_expected_arrhenius_values(self):
        spec = _arrhenius_like_spec()
        temperatures = np.array([300.0, 600.0])

        result = evaluate_safe_equation_numpy(
            spec,
            {"temperature_k": temperatures},
        )

        expected = np.exp(-50000.0 / (8.31446261815324 * temperatures))
        self.assertEqual(result.finite_count, 2)
        self.assertEqual(result.nonfinite_count, 0)
        self.assertTrue(result.metadata["evaluation_ready"])
        np.testing.assert_allclose(result.values, expected)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_numpy_evaluator_supports_overrides_and_dataframe_like_inputs(self):
        class FrameLike:
            columns = ("x",)

            def __getitem__(self, name):
                if name == "x":
                    return [1.0, 2.0, 3.0]
                raise KeyError(name)

        spec = SafeEquationSpec(
            name="custom_affine",
            expression="scale * x + offset",
            variables=[SafeEquationVariable("x")],
            parameters=[
                SafeEquationParameter("scale", initial_value=1.0),
                SafeEquationParameter("offset", initial_value=0.0),
            ],
        )

        result = evaluate_safe_equation_numpy(
            spec,
            FrameLike(),
            parameter_values={"scale": 2.0, "offset": -1.0},
        )

        self.assertEqual(result.values, (1.0, 3.0, 5.0))
        self.assertEqual(result.metadata["parameter_values"]["scale"], 2.0)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_numpy_evaluator_blocks_invalid_expression_before_execution(self):
        result = evaluate_safe_equation_numpy(
            {
                "name": "bad_function",
                "expression": "sin(x)",
                "variables": [{"name": "x"}],
            },
            {"x": [1.0]},
        )

        self.assertEqual(result.values, ())
        self.assertEqual(result.finite_count, 0)
        self.assertFalse(result.metadata["evaluation_ready"])
        self.assertIn("unsupported_function", result.warnings[0])
        json.dumps(result.to_dict(), allow_nan=False)

    def test_numpy_evaluator_reports_nonfinite_outputs(self):
        spec = SafeEquationSpec(
            name="custom_log",
            expression="log(x)",
            variables=[SafeEquationVariable("x")],
        )

        result = evaluate_safe_equation_numpy(spec, {"x": [1.0, -1.0]})

        self.assertEqual(result.values, (0.0, None))
        self.assertEqual(result.finite_count, 1)
        self.assertEqual(result.nonfinite_count, 1)
        self.assertIn("non-finite", result.warnings[0])
        json.dumps(result.to_dict(), allow_nan=False)

    def test_numpy_evaluator_allows_missing_optional_unused_variables(self):
        spec = SafeEquationSpec(
            name="custom_optional_variable",
            expression="scale * x",
            variables=[
                SafeEquationVariable("x"),
                SafeEquationVariable("unused_y", required=False),
            ],
            parameters=[SafeEquationParameter("scale", initial_value=3.0)],
        )

        result = evaluate_safe_equation_numpy(spec, {"x": [1.0, 2.0]})

        self.assertEqual(result.values, (3.0, 6.0))
        self.assertEqual(result.warnings, ())

    def test_torch_evaluator_matches_numpy_for_arrhenius_expression(self):
        spec = _arrhenius_like_spec()
        temperatures = torch.tensor([300.0, 600.0], dtype=torch.float64)

        torch_result = evaluate_safe_equation_torch(
            spec,
            {"temperature_k": temperatures},
        )
        numpy_result = evaluate_safe_equation_numpy(
            spec,
            {"temperature_k": temperatures.detach().numpy()},
        )

        self.assertEqual(torch_result.dtype, torch.float64)
        np.testing.assert_allclose(torch_result.detach().numpy(), numpy_result.values)

    def test_torch_evaluator_preserves_parameter_gradients(self):
        spec = SafeEquationSpec(
            name="custom_torch_affine",
            expression="scale * x + offset",
            variables=[SafeEquationVariable("x")],
            parameters=[
                SafeEquationParameter("scale", initial_value=2.0),
                SafeEquationParameter("offset", initial_value=1.0, learnable=False),
            ],
        )
        parameters = initialize_safe_equation_torch_parameters(spec)

        result = evaluate_safe_equation_torch(
            spec,
            {"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)},
            parameter_values=parameters,
        )
        result.sum().backward()

        self.assertTrue(parameters["scale"].requires_grad)
        self.assertFalse(parameters["offset"].requires_grad)
        self.assertAlmostEqual(parameters["scale"].grad.item(), 6.0)
        np.testing.assert_allclose(result.detach().numpy(), [3.0, 5.0, 7.0])

    def test_torch_evaluator_supports_min_max_abs_pow_and_variable_gradients(self):
        spec = SafeEquationSpec(
            name="custom_torch_math",
            expression="max(abs(x), min(pow(y, exponent), threshold))",
            variables=[SafeEquationVariable("x"), SafeEquationVariable("y")],
            parameters=[
                SafeEquationParameter("exponent", initial_value=2.0),
                SafeEquationParameter("threshold", initial_value=3.0, learnable=False),
            ],
        )
        x = torch.tensor([-1.0, 0.5], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([2.0, 1.0], dtype=torch.float64, requires_grad=True)

        result = evaluate_safe_equation_torch(
            spec,
            {"x": x, "y": y},
            parameter_values=initialize_safe_equation_torch_parameters(spec),
        )
        result.sum().backward()

        np.testing.assert_allclose(result.detach().numpy(), [3.0, 1.0])
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_torch_evaluator_rejects_invalid_spec_and_bad_inputs(self):
        invalid = {
            "name": "bad_torch_function",
            "expression": "sin(x)",
            "variables": [{"name": "x"}],
        }
        spec = SafeEquationSpec(
            name="custom_missing_torch_input",
            expression="scale * x",
            variables=[SafeEquationVariable("x")],
            parameters=[SafeEquationParameter("scale", initial_value=1.0)],
        )

        with self.assertRaisesRegex(ValueError, "unsupported_function"):
            evaluate_safe_equation_torch(invalid, {"x": torch.tensor([1.0])})
        with self.assertRaisesRegex(ValueError, "missing required variable"):
            evaluate_safe_equation_torch(spec, {})
        with self.assertRaisesRegex(ValueError, "unknown parameter override"):
            evaluate_safe_equation_torch(
                spec,
                {"x": torch.tensor([1.0])},
                parameter_values={"bad": 1.0},
            )

    def test_safe_equation_mean_function_evaluates_physics_mean(self):
        spec = SafeEquationSpec(
            name="custom_safe_mean_affine",
            expression="scale * x + offset",
            variables=[SafeEquationVariable("x")],
            parameters=[
                SafeEquationParameter("scale", initial_value=1.0, bounds=(0.0, 10.0)),
                SafeEquationParameter("offset", initial_value=0.0, learnable=False),
            ],
        )

        mean = build_safe_equation_mean_function(
            spec,
            feature_indices={"x": 0},
            parameter_values={"scale": 2.0, "offset": 1.0},
            target_mean=1.0,
            target_std=2.0,
        )
        x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)

        result = mean(x)

        self.assertIsInstance(mean, SafeEquationMeanFunction)
        np.testing.assert_allclose(result.detach().numpy(), [0.0, 1.0, 2.0])
        self.assertEqual(mean.safe_equation_feature_names, ("x",))
        self.assertIn("scale", mean.to_dict()["learnable_parameters"])
        json.dumps(mean.to_dict(), allow_nan=False)

    def test_safe_equation_mean_function_defaults_feature_order_and_constant_row_count(self):
        spec = SafeEquationSpec(
            name="custom_safe_constant_mean",
            expression="offset",
            variables=[SafeEquationVariable("x")],
            parameters=[SafeEquationParameter("offset", initial_value=5.0, learnable=False)],
        )

        mean = build_safe_equation_mean_function(spec)
        result = mean(torch.zeros(4, 1, dtype=torch.float64))

        self.assertEqual(mean.feature_indices, {"x": 0})
        np.testing.assert_allclose(result.detach().numpy(), [5.0, 5.0, 5.0, 5.0])

    def test_safe_equation_mean_function_can_fit_with_gpytorch_gpr(self):
        spec = SafeEquationSpec(
            name="custom_safe_fit_mean",
            expression="scale * x + offset",
            variables=[SafeEquationVariable("x")],
            parameters=[
                SafeEquationParameter("scale", initial_value=0.8, bounds=(0.0, 5.0)),
                SafeEquationParameter("offset", initial_value=0.05, learnable=False),
            ],
        )
        x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        y = 1.5 * x.ravel() + 0.05
        mean = build_safe_equation_mean_function(spec, feature_indices={"x": 0})

        result = fit_gpytorch_gpr(
            x,
            y,
            kernel="rbf",
            mean_module=mean,
            training_iter=2,
            initial_noise=0.05,
            verbose=False,
        )

        self.assertIsInstance(result.model.mean_module, SafeEquationMeanFunction)
        self.assertIn("scale", result.model.mean_module.current_parameter_values())

    def test_safe_equation_mean_function_validates_mappings(self):
        spec = SafeEquationSpec(
            name="custom_safe_invalid_mean",
            expression="scale * x",
            variables=[SafeEquationVariable("x")],
            parameters=[SafeEquationParameter("scale", initial_value=1.0, bounds=(0.0, 2.0))],
        )

        with self.assertRaisesRegex(ValueError, "missing variables"):
            build_safe_equation_mean_function(spec, feature_indices={})
        with self.assertRaisesRegex(ValueError, "undeclared variables"):
            build_safe_equation_mean_function(spec, feature_indices={"bad": 0})
        with self.assertRaisesRegex(ValueError, "undeclared parameters"):
            build_safe_equation_mean_function(spec, parameter_values={"bad": 1.0})
        with self.assertRaisesRegex(ValueError, "within bounds"):
            build_safe_equation_mean_function(spec, parameter_values={"scale": 3.0})
        with self.assertRaisesRegex(ValueError, "positive_parameters"):
            build_safe_equation_mean_function(spec, positive_parameters=("bad",))

    def test_preview_safe_equation_mean_reports_residual_summary(self):
        spec = SafeEquationSpec(
            name="custom_affine_preview",
            expression="scale * x + offset",
            variables=[SafeEquationVariable("x")],
            parameters=[
                SafeEquationParameter("scale", initial_value=2.0),
                SafeEquationParameter("offset", initial_value=1.0),
            ],
        )

        preview = preview_safe_equation_mean(
            spec,
            {"x": [1.0, 2.0, 3.0]},
            target_values=[3.0, 6.0, 8.0],
        )

        self.assertTrue(preview.preview_ready)
        self.assertEqual(preview.valid_input_row_count, 3)
        self.assertEqual(preview.invalid_input_row_count, 0)
        self.assertEqual(preview.finite_mean_count, 3)
        self.assertEqual(preview.physics_mean_summary["mean"], 5.0)
        self.assertAlmostEqual(
            preview.target_residual_summary["mean_absolute_residual"],
            2 / 3,
        )
        json.dumps(preview.to_dict(), allow_nan=False)

    def test_preview_reports_invalid_inputs_and_nonfinite_means(self):
        spec = SafeEquationSpec(
            name="custom_sqrt_preview",
            expression="sqrt(x)",
            variables=[SafeEquationVariable("x")],
        )

        preview = preview_safe_equation_mean(spec, {"x": [4.0, -1.0, math.nan]})

        self.assertFalse(preview.preview_ready)
        self.assertEqual(preview.valid_input_row_count, 2)
        self.assertEqual(preview.invalid_input_row_count, 1)
        self.assertEqual(preview.finite_mean_count, 1)
        self.assertEqual(preview.nonfinite_mean_count, 2)
        self.assertTrue(any("non-finite" in warning for warning in preview.warnings))
        json.dumps(preview.to_dict(), allow_nan=False)

    def test_validation_and_preview_shells_are_json_safe(self):
        issue = SafeEquationValidationIssue(
            code="future_warning",
            message="parser validation is not implemented in this first step",
            severity="warning",
        )
        validation = SafeEquationValidationResult(
            issues=[issue],
            spec_name="custom_arrhenius_rate",
            declared_symbols=("temperature_k", "prefactor"),
        )
        evaluation = SafeEquationEvaluationResult(
            spec_name="custom_arrhenius_rate",
            values=(1.0, None),
            finite_count=1,
            nonfinite_count=1,
            warnings=("one non-finite value was omitted",),
        )
        preview = SafeEquationMeanPreview(
            spec_name="custom_arrhenius_rate",
            preview_ready=False,
            valid_input_row_count=1,
            invalid_input_row_count=1,
            finite_mean_count=1,
            nonfinite_mean_count=1,
            physics_mean_summary={"mean": 1.0},
            target_residual_summary=None,
            warnings=("preview evaluator pending",),
        )

        self.assertTrue(validation.is_valid)
        json.dumps(validation.to_dict(), allow_nan=False)
        json.dumps(evaluation.to_dict(), allow_nan=False)
        json.dumps(preview.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
