"""Shared deterministic fixtures for matgpr tests."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Sequence
from types import ModuleType

import numpy as np
import pytest
from hypothesis import settings

TEST_RANDOM_SEED = 20261027
MAX_REPORTED_MARKER_VIOLATIONS = 20
PRIMARY_TEST_MARKERS = frozenset({"unit", "contract", "component", "integration", "scientific"})

settings.register_profile(
    "matgpr",
    max_examples=50,
    deadline=None,
    derandomize=True,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "matgpr"))


def pytest_collection_modifyitems(items: Sequence[pytest.Item]) -> None:
    """Require exactly one primary ownership marker on every collected test."""

    violations: list[str] = []
    for item in items:
        primary_markers = sorted(
            {marker.name for marker in item.iter_markers() if marker.name in PRIMARY_TEST_MARKERS}
        )
        if len(primary_markers) != 1:
            ownership = ", ".join(primary_markers) if primary_markers else "none"
            violations.append(f"{item.nodeid} ({ownership})")

    if violations:
        details = "\n".join(
            f"  - {violation}" for violation in violations[:MAX_REPORTED_MARKER_VIOLATIONS]
        )
        remaining = len(violations) - MAX_REPORTED_MARKER_VIOLATIONS
        if remaining > 0:
            details += f"\n  - ... and {remaining} more"
        raise pytest.UsageError(
            "Every test must have exactly one primary ownership marker from "
            f"{sorted(PRIMARY_TEST_MARKERS)}. Violations:\n{details}"
        )


@pytest.fixture
def deterministic_rng() -> np.random.Generator:
    """Return a fresh deterministic NumPy generator for one test."""

    return np.random.default_rng(TEST_RANDOM_SEED)


@pytest.fixture
def compact_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a small deterministic nonlinear regression dataset."""

    x = np.linspace(0.0, 1.0, 16, dtype=float).reshape(-1, 1)
    y = 0.25 + 1.5 * x[:, 0] + 0.1 * np.sin(4.0 * np.pi * x[:, 0])
    return x, y


@pytest.fixture
def require_optional_dependency() -> Callable[[str], ModuleType]:
    """Return a helper that skips when an optional dependency is unavailable."""

    def require(module_name: str) -> ModuleType:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name is None or not (
                module_name == exc.name or module_name.startswith(f"{exc.name}.")
            ):
                raise
            pytest.skip(f"optional dependency {module_name!r} is not installed")

    return require
