from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from matgpr import (
    OPTIONAL_DEPENDENCIES,
    OptionalDependency,
    is_optional_dependency_available,
    list_optional_dependencies,
    require_optional_dependency,
)


class OptionalDependencyTests(unittest.TestCase):
    def test_require_optional_dependency_imports_available_module(self):
        module = require_optional_dependency("math")

        self.assertEqual(module.__name__, "math")

    def test_registered_missing_dependency_has_clear_extra_message(self):
        with patch(
            "matgpr.optional_dependencies.importlib.import_module",
            side_effect=ImportError("missing dscribe"),
        ):
            with self.assertRaises(ImportError) as context:
                require_optional_dependency("dscribe")

        message = str(context.exception)
        self.assertIn("DScribe structure descriptors", message)
        self.assertIn("optional dependency `dscribe`", message)
        self.assertIn("optional structures extra", message)
        self.assertIn('python -m pip install "matgpr[structures]"', message)
        self.assertIn("python -m pip install dscribe", message)

    def test_custom_missing_dependency_can_override_install_metadata(self):
        dependency = OptionalDependency(
            import_name="example_backend",
            package_name="example-package",
            extra="examples",
            purpose="Example backend",
        )
        with patch(
            "matgpr.optional_dependencies.importlib.import_module",
            side_effect=ImportError("missing example backend"),
        ):
            with self.assertRaises(ImportError) as context:
                require_optional_dependency(
                    dependency,
                    purpose="Custom descriptors",
                    extra="custom-extra",
                    package_name="custom-package",
                )

        message = str(context.exception)
        self.assertIn("Custom descriptors", message)
        self.assertIn("custom-package", message)
        self.assertIn("matgpr[custom-extra]", message)

    def test_optional_dependency_availability_uses_import_result(self):
        fake_module = types.ModuleType("fake_backend")
        with patch(
            "matgpr.optional_dependencies.importlib.import_module",
            return_value=fake_module,
        ):
            self.assertTrue(is_optional_dependency_available("fake_backend"))

        with patch(
            "matgpr.optional_dependencies.importlib.import_module",
            side_effect=ImportError("missing fake backend"),
        ):
            self.assertFalse(is_optional_dependency_available("fake_backend"))

    def test_registered_optional_dependencies_are_unique_in_listing(self):
        listed = list_optional_dependencies()
        keys = {(dependency.import_name, dependency.extra) for dependency in listed}

        self.assertEqual(len(listed), len(keys))
        self.assertIn("dscribe", OPTIONAL_DEPENDENCIES)
        self.assertIn(("dscribe", "structures"), keys)
        self.assertIn(("mordred", "molecular-extra"), keys)

    def test_empty_dependency_name_is_rejected(self):
        with self.assertRaises(ValueError):
            require_optional_dependency(" ")


if __name__ == "__main__":
    unittest.main()
