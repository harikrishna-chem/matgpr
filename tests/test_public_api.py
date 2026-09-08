from __future__ import annotations

import importlib
import subprocess
import sys
import unittest
from importlib.metadata import version
from pathlib import Path


class PublicApiTests(unittest.TestCase):
    def test_package_version_matches_installed_metadata(self):
        import matgpr

        self.assertEqual(matgpr.__version__, version("matgpr"))

    def test_package_import_does_not_eagerly_load_heavy_modeling_modules(self):
        script = (
            "import sys; import matgpr; "
            "print('matgpr.gpytorch_gpr' in sys.modules); "
            "print('torch' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.splitlines(), ["False", "False"])

    def test_module_all_exports_are_explicit_and_resolvable(self):
        modules = ["matgpr"]
        modules.extend(
            f"matgpr.{path.stem}"
            for path in sorted(Path("matgpr").glob("*.py"))
            if path.stem != "__init__"
        )

        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                exported = getattr(module, "__all__", None)

                self.assertIsNotNone(exported)
                self.assertEqual(len(exported), len(set(exported)))
                missing = [name for name in exported if not hasattr(module, name)]
                self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
