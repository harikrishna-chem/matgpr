from __future__ import annotations

import json
import unittest

from matgpr.mcp_tools import (
    get_matgpr_info,
    get_physics_equation,
    list_capabilities,
    list_physics_equations,
    recommend_featurizers,
    suggest_bo_workflow,
    suggest_validation_workflow,
    validate_safe_equation,
)


class MCPToolsTests(unittest.TestCase):
    def test_get_matgpr_info_is_json_safe_and_lists_mcp_extra(self):
        info = get_matgpr_info()

        self.assertEqual(info["name"], "matgpr")
        self.assertIn("mcp", info["optional_dependency_groups"])
        self.assertIn("physics_informed_gpr", info["capability_names"])
        json.dumps(info, allow_nan=False)

    def test_list_capabilities_includes_bo_and_featurization(self):
        payload = list_capabilities()
        names = {record["name"] for record in payload["capabilities"]}

        self.assertIn("materials_featurization", names)
        self.assertIn("bayesian_optimization", names)
        json.dumps(payload, allow_nan=False)

    def test_recommend_featurizers_detects_materials_columns(self):
        payload = recommend_featurizers(
            ["formula", "polymer_smiles", "solvent_smiles", "temperature_C", "diffusivity"],
            dtype_summary={
                "formula": "object",
                "polymer_smiles": "object",
                "solvent_smiles": "object",
                "temperature_C": "float64",
                "diffusivity": "float64",
            },
            sample_rows=[
                {
                    "formula": "LiFePO4",
                    "polymer_smiles": "[*]CC([*])C",
                    "solvent_smiles": "CCO",
                    "temperature_C": 25.0,
                    "diffusivity": 1.2e-8,
                },
                {
                    "formula": "Al2O3",
                    "polymer_smiles": "[*]COC([*])",
                    "solvent_smiles": "O=C=O",
                    "temperature_C": 40.0,
                    "diffusivity": 2.1e-8,
                },
            ],
            target_column="diffusivity",
        )

        roles = {
            record["column"]: record["inferred_role"] for record in payload["column_assessments"]
        }
        self.assertEqual(roles["formula"], "composition")
        self.assertEqual(roles["polymer_smiles"], "polymer_smiles")
        self.assertEqual(roles["solvent_smiles"], "molecule_smiles")
        self.assertEqual(roles["temperature_C"], "processing_condition")
        self.assertEqual(roles["diffusivity"], "target")
        self.assertIn(
            "temperature_k",
            {record["canonical_feature"] for record in payload["physics_feature_map_candidates"]},
        )
        json.dumps(payload, allow_nan=False)

    def test_list_physics_equations_filters_by_application(self):
        payload = list_physics_equations(application="diffusion")
        names = {record["name"] for record in payload["templates"]}

        self.assertIn("arrhenius_rate", names)
        self.assertGreaterEqual(payload["count"], 1)
        json.dumps(payload, allow_nan=False)

    def test_get_physics_equation_returns_implementation_metadata(self):
        payload = get_physics_equation("arrhenius")

        self.assertTrue(payload["found"])
        self.assertEqual(payload["template"]["name"], "arrhenius_rate")
        self.assertIn("activation_energy", payload["implementation"]["learned_parameters"])
        json.dumps(payload, allow_nan=False)

    def test_get_physics_equation_handles_unknown_name(self):
        payload = get_physics_equation("not_real")

        self.assertFalse(payload["found"])
        self.assertIn("available_templates", payload)
        json.dumps(payload, allow_nan=False)

    def test_validate_safe_equation_returns_normalized_spec(self):
        payload = validate_safe_equation(
            {
                "name": "linear_mean",
                "expression": "offset + slope * x",
                "variables": [{"name": "x", "units": "K"}],
                "parameters": [
                    {"name": "offset", "initial_value": 0.0},
                    {"name": "slope", "initial_value": 1.0},
                ],
            }
        )

        self.assertTrue(payload["is_valid"])
        self.assertEqual(payload["normalized_spec"]["name"], "linear_mean")
        json.dumps(payload, allow_nan=False)

    def test_validate_safe_equation_reports_invalid_expression(self):
        payload = validate_safe_equation(
            {
                "name": "bad_mean",
                "expression": "__import__('os').system('echo bad')",
                "variables": [{"name": "x"}],
            }
        )

        self.assertFalse(payload["is_valid"])
        self.assertIsNone(payload["normalized_spec"])
        json.dumps(payload, allow_nan=False)

    def test_suggest_validation_workflow_prefers_repeated_splits_for_low_data(self):
        payload = suggest_validation_workflow(n_rows=80)

        self.assertEqual(payload["recommended_protocol"]["learning_curve"]["splits_per_point"], 20)
        self.assertIn("rmse", payload["recommended_protocol"]["learning_curve"]["metrics"])
        json.dumps(payload, allow_nan=False)

    def test_suggest_bo_workflow_is_json_safe(self):
        payload = suggest_bo_workflow(
            candidate_count=50,
            objective_direction="maximize",
            batch_size=3,
            has_constraints=True,
            duplicate_key_columns=["formula"],
        )

        self.assertEqual(payload["recommended_workflow"]["batch_size"], 3)
        self.assertEqual(payload["optional_extra"], "bo")
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
