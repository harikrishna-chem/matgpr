from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

__all__ = [
    "OptionalDependency",
    "OPTIONAL_DEPENDENCIES",
    "is_optional_dependency_available",
    "list_optional_dependencies",
    "require_optional_dependency",
]


@dataclass(frozen=True)
class OptionalDependency:
    """Metadata for an optional backend dependency.

    Parameters
    ----------
    import_name
        Python import path used to import the package.
    package_name
        Package name users should install.
    extra
        `matgpr` optional dependency extra that includes the package.
    purpose
        Short user-facing description of why the dependency is needed.
    """

    import_name: str
    package_name: str
    extra: str
    purpose: str


OPTIONAL_DEPENDENCIES: dict[str, OptionalDependency] = {
    "botorch": OptionalDependency(
        import_name="botorch",
        package_name="botorch",
        extra="bo",
        purpose="BoTorch Bayesian optimization",
    ),
    "ase": OptionalDependency(
        import_name="ase",
        package_name="ase",
        extra="structures",
        purpose="ASE structure conversion",
    ),
    "dscribe": OptionalDependency(
        import_name="dscribe",
        package_name="dscribe",
        extra="structures",
        purpose="DScribe structure descriptors",
    ),
    "mordred": OptionalDependency(
        import_name="mordred",
        package_name="mordredcommunity",
        extra="molecular-extra",
        purpose="Mordred molecular descriptors",
    ),
    "mordredcommunity": OptionalDependency(
        import_name="mordred",
        package_name="mordredcommunity",
        extra="molecular-extra",
        purpose="Mordred molecular descriptors",
    ),
    "jarvis": OptionalDependency(
        import_name="jarvis",
        package_name="jarvis-tools",
        extra="jarvis",
        purpose="JARVIS materials descriptors",
    ),
    "jarvis-tools": OptionalDependency(
        import_name="jarvis",
        package_name="jarvis-tools",
        extra="jarvis",
        purpose="JARVIS materials descriptors",
    ),
    "deepchem": OptionalDependency(
        import_name="deepchem",
        package_name="deepchem",
        extra="deep",
        purpose="DeepChem featurizers",
    ),
}


def require_optional_dependency(
    dependency: str | OptionalDependency,
    *,
    purpose: str | None = None,
    extra: str | None = None,
    package_name: str | None = None,
) -> ModuleType:
    """Import an optional dependency or raise a clear installation message.

    Parameters
    ----------
    dependency
        Registered dependency name, Python import name, or explicit
        `OptionalDependency` metadata.
    purpose
        Optional purpose override for custom wrappers.
    extra
        Optional `matgpr` extra override for custom wrappers.
    package_name
        Optional install package name override for custom wrappers.
    """
    metadata = _resolve_optional_dependency(
        dependency,
        purpose=purpose,
        extra=extra,
        package_name=package_name,
    )
    try:
        return importlib.import_module(metadata.import_name)
    except ImportError as exc:
        raise ImportError(_missing_dependency_message(metadata)) from exc


def is_optional_dependency_available(dependency: str | OptionalDependency) -> bool:
    """Return `True` when an optional dependency can be imported."""
    metadata = _resolve_optional_dependency(dependency)
    try:
        importlib.import_module(metadata.import_name)
    except ImportError:
        return False
    return True


def list_optional_dependencies() -> tuple[OptionalDependency, ...]:
    """Return unique optional dependency metadata records."""
    unique: dict[tuple[str, str], OptionalDependency] = {}
    for metadata in OPTIONAL_DEPENDENCIES.values():
        unique[(metadata.import_name, metadata.extra)] = metadata
    return tuple(unique.values())


def _resolve_optional_dependency(
    dependency: str | OptionalDependency,
    *,
    purpose: str | None = None,
    extra: str | None = None,
    package_name: str | None = None,
) -> OptionalDependency:
    if isinstance(dependency, OptionalDependency):
        return OptionalDependency(
            import_name=dependency.import_name,
            package_name=package_name or dependency.package_name,
            extra=extra or dependency.extra,
            purpose=purpose or dependency.purpose,
        )

    key = str(dependency).strip()
    if not key:
        raise ValueError("dependency must be a non-empty string")

    registered = OPTIONAL_DEPENDENCIES.get(key.lower())
    if registered is not None:
        return OptionalDependency(
            import_name=registered.import_name,
            package_name=package_name or registered.package_name,
            extra=extra or registered.extra,
            purpose=purpose or registered.purpose,
        )

    return OptionalDependency(
        import_name=key,
        package_name=package_name or key,
        extra=extra or "all-fingerprints",
        purpose=purpose or f"{key} backend",
    )


def _missing_dependency_message(metadata: OptionalDependency) -> str:
    return (
        f"{metadata.purpose} requires optional dependency `{metadata.package_name}`. "
        f"Install the optional {metadata.extra} extra with "
        f"`python -m pip install \"matgpr[{metadata.extra}]\"`, or install the "
        f"backend directly with `python -m pip install {metadata.package_name}`."
    )
