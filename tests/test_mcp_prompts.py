from __future__ import annotations

import unittest

from matgpr.mcp_prompts import (
    MCP_PROMPT_REGISTRATIONS,
    plan_bayesian_optimization_workflow,
    plan_featurization_workflow,
    plan_physics_informed_gpr_workflow,
    plan_validation_workflow,
)


class MCPPromptsTests(unittest.TestCase):
    def test_prompt_registration_names_are_stable(self):
        names = [registration.name for registration in MCP_PROMPT_REGISTRATIONS]

        self.assertEqual(
            names,
            [
                "plan_featurization_workflow",
                "plan_physics_informed_gpr_workflow",
                "plan_validation_workflow",
                "plan_bayesian_optimization_workflow",
            ],
        )

    def test_featurization_prompt_points_to_recommendation_tool(self):
        prompt = plan_featurization_workflow(
            columns="formula, polymer_smiles, target",
            target_column="target",
        )

        self.assertIn("recommend_featurizers", prompt)
        self.assertIn("formula, polymer_smiles, target", prompt)
        self.assertIn("PolymerSmilesFeaturizer", prompt)

    def test_physics_prompt_requires_validation_and_preview(self):
        prompt = plan_physics_informed_gpr_workflow(target_property="diffusivity")

        self.assertIn("validate_safe_equation", prompt)
        self.assertIn("preview_safe_equation", prompt)
        self.assertIn("physics_feature_map", prompt)

    def test_validation_prompt_mentions_learning_curves_and_uncertainty(self):
        prompt = plan_validation_workflow(n_rows="75")

        self.assertIn("suggest_validation_workflow", prompt)
        self.assertIn("Learning curves", prompt)
        self.assertIn("uncertainty", prompt.lower())

    def test_bo_prompt_mentions_finite_pool_and_duplicate_avoidance(self):
        prompt = plan_bayesian_optimization_workflow(objective="maximize conductivity")

        self.assertIn("suggest_bo_workflow", prompt)
        self.assertIn("finite-pool", prompt)
        self.assertIn("duplicate avoidance", prompt.lower())


if __name__ == "__main__":
    unittest.main()
