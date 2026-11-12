"""Verify that matgpr imports from an installed wheel, not the source tree."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

import matgpr


def main() -> None:
    """Validate package identity and installation location."""

    package_path = Path(matgpr.__file__).resolve()
    environment_root = Path(sys.prefix).resolve()
    installed_version = version("matgpr")

    if not package_path.is_relative_to(environment_root):
        raise RuntimeError(
            f"matgpr was not imported from the wheel-smoke environment: {package_path}"
        )
    if matgpr.__version__ != installed_version:
        raise RuntimeError(
            "matgpr package and distribution versions differ: "
            f"{matgpr.__version__!r} != {installed_version!r}"
        )

    print(
        json.dumps(
            {
                "package": "matgpr",
                "version": installed_version,
                "module_path": str(package_path),
                "wheel_smoke": "passed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
