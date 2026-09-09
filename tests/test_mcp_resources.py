from __future__ import annotations

import unittest

from matgpr.mcp_resources import (
    MCP_RESOURCE_REGISTRATIONS,
    bayesian_optimization_guide,
    capabilities_guide,
    featurization_guide,
    physics_informed_gpr_guide,
    validation_guide,
)


class MCPResourcesTests(unittest.TestCase):
    def test_resource_registration_uris_are_stable(self):
        uris = [registration.uri for registration in MCP_RESOURCE_REGISTRATIONS]

        self.assertEqual(
            uris,
            [
                "matgpr://guide/capabilities",
                "matgpr://guide/featurization",
                "matgpr://guide/physics-informed-gpr",
                "matgpr://guide/validation",
                "matgpr://guide/bayesian-optimization",
            ],
        )

    def test_capabilities_guide_documents_safety_boundary(self):
        guide = capabilities_guide()

        self.assertIn("read-only", guide)
        self.assertIn("No model fitting", guide)
        self.assertIn("preview_safe_equation", guide)

    def test_featurization_guide_mentions_material_column_types(self):
        guide = featurization_guide()

        self.assertIn("CompositionFeaturizer", guide)
        self.assertIn("SmilesFeaturizer", guide)
        self.assertIn("PolymerSmilesFeaturizer", guide)

    def test_physics_guide_mentions_mean_function_and_feature_map(self):
        guide = physics_informed_gpr_guide()

        self.assertIn("mean", guide)
        self.assertIn("physics_feature_map", guide)
        self.assertIn("validate_safe_equation", guide)

    def test_validation_guide_mentions_repeated_splits(self):
        guide = validation_guide()

        self.assertIn("Learning curves", guide)
        self.assertIn("Repeated splits", guide)
        self.assertIn("uncertainty", guide.lower())

    def test_bo_guide_mentions_acquisition_and_logging(self):
        guide = bayesian_optimization_guide()

        self.assertIn("expected improvement", guide)
        self.assertIn("acquisition", guide)
        self.assertIn("Log campaign", guide)


if __name__ == "__main__":
    unittest.main()
