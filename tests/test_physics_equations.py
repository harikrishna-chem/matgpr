from __future__ import annotations

import unittest

import torch

from matgpr import (
    R_GAS_CONSTANT_J_MOL_K,
    PhysicsFeatureSpec,
    PhysicsEquationTemplate,
    PhysicsParameterSpec,
    arrhenius_linear_growth_equation,
    arrhenius_rate_equation,
    arrhenius_sqrt_time_equation,
    available_physics_equation_templates,
    describe_physics_equation_template,
    free_volume_exponential_equation,
    get_physics_equation_template,
    hall_petch_equation,
    linear_parabolic_growth_equation,
    linear_growth_equation,
    list_physics_equation_templates,
    power_law_equation,
    rule_of_mixtures_equation,
    search_physics_equation_templates,
    summarize_physics_equation_templates,
)


class PhysicsEquationTemplateTests(unittest.TestCase):
    def test_registry_lists_and_resolves_templates(self):
        names = available_physics_equation_templates()
        names_with_aliases = available_physics_equation_templates(include_aliases=True)
        templates = list_physics_equation_templates()

        self.assertIn("arrhenius_rate", names)
        self.assertIn("arrhenius", names_with_aliases)
        self.assertIn("arrhenius_linear_growth", names)
        self.assertIn("temperature_linear_oxidation", names_with_aliases)
        self.assertIn("linear_growth", names)
        self.assertIn("linear_oxidation", names_with_aliases)
        self.assertIn("linear_parabolic_growth", names)
        self.assertIn("linear_parabolic", names_with_aliases)
        self.assertTrue(all(isinstance(template, PhysicsEquationTemplate) for template in templates))
        self.assertEqual(get_physics_equation_template("arrhenius").name, "arrhenius_rate")
        self.assertEqual(
            get_physics_equation_template("linear_oxidation").name,
            "linear_growth",
        )
        self.assertEqual(
            get_physics_equation_template("temperature_linear_oxidation").name,
            "arrhenius_linear_growth",
        )
        self.assertEqual(
            get_physics_equation_template("linear_parabolic").name,
            "linear_parabolic_growth",
        )

    def test_template_metadata_describes_features_parameters_and_assumptions(self):
        template = get_physics_equation_template("arrhenius_rate")
        feature_specs = template.feature_specs()
        parameter_specs = template.parameter_specs()
        record = describe_physics_equation_template("arrhenius")

        self.assertTrue(all(isinstance(spec, PhysicsFeatureSpec) for spec in feature_specs))
        self.assertTrue(all(isinstance(spec, PhysicsParameterSpec) for spec in parameter_specs))
        self.assertEqual(feature_specs[0].name, "temperature_k")
        self.assertEqual(feature_specs[0].units, "K")
        self.assertIn("activation_energy", template.learnable_parameter_names)
        self.assertTrue(record["parameter_metadata"][1]["learned_on_fit"])
        self.assertEqual(record["parameter_metadata"][1]["units"], "J/mol")
        self.assertGreater(len(record["assumptions"]), 0)
        self.assertIn("temperature", record["tags"])

    def test_search_and_summary_support_discovery_workflows(self):
        transport_templates = search_physics_equation_templates(query="transport")
        grain_templates = search_physics_equation_templates(
            application="strength",
            required_features=("grain_size",),
        )
        polymer_templates = search_physics_equation_templates(tag="polymer")
        summary = summarize_physics_equation_templates()

        self.assertIn("arrhenius_rate", {template.name for template in transport_templates})
        self.assertEqual([template.name for template in grain_templates], ["hall_petch"])
        self.assertEqual([template.name for template in polymer_templates], ["free_volume_exponential"])
        self.assertIn("name", summary.columns)
        self.assertIn("required_features", summary.columns)
        self.assertGreaterEqual(summary.shape[0], 6)

    def test_arrhenius_rate_equation_uses_fixed_gas_constant_default(self):
        temperature = torch.tensor([300.0, 600.0], dtype=torch.float64)
        values = arrhenius_rate_equation(
            {"temperature_k": temperature},
            {
                "prefactor": torch.tensor(2.0, dtype=torch.float64),
                "activation_energy": torch.tensor(0.0, dtype=torch.float64),
            },
        )

        self.assertTrue(torch.allclose(values, torch.tensor([2.0, 2.0], dtype=torch.float64)))
        self.assertAlmostEqual(R_GAS_CONSTANT_J_MOL_K, 8.31446261815324)

    def test_common_equations_return_expected_values(self):
        x = torch.tensor([1.0, 4.0], dtype=torch.float64)

        power_law = power_law_equation(
            {"driving_variable": x},
            {"offset": 1.0, "coefficient": 2.0, "exponent": 0.5},
        )
        hall_petch = hall_petch_equation(
            {"grain_size": x},
            {"intrinsic_property": 10.0, "coefficient": 2.0},
        )
        mixture = rule_of_mixtures_equation(
            {"volume_fraction": torch.tensor([0.0, 0.5], dtype=torch.float64)},
            {"matrix_property": 1.0, "inclusion_property": 3.0, "interaction": 4.0},
        )

        self.assertTrue(torch.allclose(power_law, torch.tensor([3.0, 5.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(hall_petch, torch.tensor([12.0, 11.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(mixture, torch.tensor([1.0, 3.0], dtype=torch.float64)))

    def test_transport_equations_return_expected_values(self):
        free_volume = free_volume_exponential_equation(
            {"free_volume_fraction": torch.tensor([0.5], dtype=torch.float64)},
            {"prefactor": 2.0, "barrier": 0.0, "offset": 1.0},
        )
        linear_growth = linear_growth_equation(
            {"time": torch.tensor([2.0, 5.0], dtype=torch.float64)},
            {"rate": 3.0, "offset": 1.0},
        )
        arrhenius_linear_growth = arrhenius_linear_growth_equation(
            {
                "temperature_k": torch.tensor([300.0], dtype=torch.float64),
                "time": torch.tensor([2.0, 5.0], dtype=torch.float64),
            },
            {"prefactor": 4.0, "activation_energy": 0.0, "offset": 1.0},
        )
        linear_parabolic_growth = linear_parabolic_growth_equation(
            {"time": torch.tensor([0.0, 1.0], dtype=torch.float64)},
            {"linear_rate": 2.0, "parabolic_rate": 3.0, "offset": 1.0},
        )
        sqrt_time = arrhenius_sqrt_time_equation(
            {
                "temperature_k": torch.tensor([300.0], dtype=torch.float64),
                "time": torch.tensor([9.0], dtype=torch.float64),
            },
            {"prefactor": 4.0, "activation_energy": 0.0, "offset": 1.0},
        )

        self.assertTrue(torch.allclose(free_volume, torch.tensor([3.0], dtype=torch.float64)))
        self.assertTrue(
            torch.allclose(
                linear_growth,
                torch.tensor([7.0, 16.0], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.allclose(
                arrhenius_linear_growth,
                torch.tensor([9.0, 21.0], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.allclose(
                linear_parabolic_growth,
                torch.tensor([1.0, 2.0], dtype=torch.float64),
            )
        )
        self.assertTrue(torch.allclose(sqrt_time, torch.tensor([7.0], dtype=torch.float64)))

    def test_template_builds_physics_informed_mean(self):
        template = get_physics_equation_template("power_law")
        mean = template.build_mean_function(
            {"driving_variable": 0},
            learnable_parameter_overrides={
                "coefficient": 2.0,
                "exponent": 0.5,
                "offset": 1.0,
            },
            target_mean=1.0,
            target_std=2.0,
        )

        x = torch.tensor([[4.0], [9.0]], dtype=torch.float64)
        expected = torch.tensor([(5.0 - 1.0) / 2.0, (7.0 - 1.0) / 2.0], dtype=torch.float64)

        self.assertTrue(torch.allclose(mean(x), expected, atol=1e-6))
        self.assertAlmostEqual(mean.current_parameter_values()["coefficient"], 2.0, places=6)

    def test_template_validation_errors_are_explicit(self):
        template = get_physics_equation_template("hall_petch")

        with self.assertRaises(ValueError):
            template.build_mean_function({})
        with self.assertRaises(ValueError):
            get_physics_equation_template("not-a-template")
        with self.assertRaises(KeyError):
            power_law_equation({}, {"coefficient": 1.0, "exponent": 1.0})


if __name__ == "__main__":
    unittest.main()
