from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .fingerprint_cache import (
    fingerprint_cache_key,
    read_fingerprint_cache_record,
    write_fingerprint_cache_record,
)

__all__ = [
    "DEFAULT_STRUCTURE_FEATURES",
    "StructureFingerprintResult",
    "append_structure_fingerprints",
    "featurize_structures",
    "structure_feature_names",
    "structure_fingerprint",
]

DEFAULT_STRUCTURE_FEATURES: tuple[str, ...] = (
    "log_lattice_length_min",
    "log_lattice_length_mid",
    "log_lattice_length_max",
    "cos_lattice_angle_min",
    "cos_lattice_angle_mid",
    "cos_lattice_angle_max",
    "log_volume_per_atom",
    "density",
)


@dataclass(frozen=True)
class StructureFingerprintResult:
    """Result returned by structure featurization helpers.

    Attributes
    ----------
    features
        Numeric structure descriptors indexed like the input structures.
    failed
        Structures that could not be parsed or featurized. Empty when
        ``errors="raise"`` and no exception was raised.
    cache_keys
        Deterministic cache keys for each input row.
    cache_hit
        Boolean flags indicating whether each row was loaded from cache.
    """

    features: pd.DataFrame
    failed: pd.DataFrame
    cache_keys: pd.Series | None = None
    cache_hit: pd.Series | None = None


def structure_fingerprint(
    structure: object,
    *,
    features: Sequence[str] = DEFAULT_STRUCTURE_FEATURES,
) -> dict[str, float]:
    """Create lightweight global structure descriptors for one crystal structure.

    Parameters
    ----------
    structure
        A ``pymatgen.core.Structure`` object, a path to a structure file, or a
        CIF string.
    features
        Descriptor names to return. Defaults are global lattice-shape and
        packing descriptors intended for small tabular GPR workflows.
    """
    feature_names = _validate_structure_feature_names(features)
    structure_object = _coerce_structure(structure)
    values = _structure_feature_values(structure_object)
    return {name: float(values[name]) for name in feature_names}


def featurize_structures(
    structures: Iterable[object],
    *,
    features: Sequence[str] = DEFAULT_STRUCTURE_FEATURES,
    errors: str = "raise",
    cache_dir: str | Path | None = None,
) -> StructureFingerprintResult:
    """Featurize a sequence of crystal structures.

    Parameters
    ----------
    structures
        Iterable of ``pymatgen.Structure`` objects, structure-file paths, or CIF
        strings.
    features
        Descriptor names passed to ``structure_fingerprint``.
    errors
        ``"raise"`` stops at the first invalid structure. ``"coerce"`` returns
        rows filled with ``NaN`` and records failures in ``failed``.
    cache_dir
        Optional directory for persistent row-level JSON cache files.
    """
    if errors not in {"raise", "coerce"}:
        raise ValueError("errors must be either 'raise' or 'coerce'")

    feature_names = _validate_structure_feature_names(features)
    cache_parameters = {"features": list(feature_names)}
    rows: list[Mapping[str, float]] = []
    failures: list[dict[str, object]] = []
    cache_keys: list[str] = []
    cache_hits: list[bool] = []

    for index, structure in enumerate(structures):
        cache_key = fingerprint_cache_key(
            namespace="structure",
            value=_structure_cache_value(structure),
            parameters=cache_parameters,
        )
        cache_keys.append(cache_key)

        try:
            cached_features = _read_cached_structure_row(
                cache_dir,
                cache_key=cache_key,
                feature_names=feature_names,
            )
            if cached_features is not None:
                rows.append(cached_features)
                cache_hits.append(True)
                continue

            row = structure_fingerprint(structure, features=feature_names)
            rows.append(row)
            cache_hits.append(False)
            _write_cached_structure_row(
                cache_dir,
                cache_key=cache_key,
                input_value=_structure_cache_value(structure),
                features=row,
                metadata=cache_parameters,
            )
        except Exception as exc:
            if errors == "raise":
                raise ValueError(f"Could not featurize structure at position {index}") from exc
            failures.append(
                {
                    "index": index,
                    "structure": _short_structure_label(structure),
                    "cache_key": cache_key,
                    "error": str(exc),
                }
            )
            rows.append({name: np.nan for name in feature_names})
            cache_hits.append(False)

    return StructureFingerprintResult(
        features=pd.DataFrame(rows, columns=feature_names),
        failed=pd.DataFrame(failures, columns=["index", "structure", "cache_key", "error"]),
        cache_keys=pd.Series(cache_keys, name="structure_cache_key"),
        cache_hit=pd.Series(cache_hits, name="structure_cache_hit"),
    )


def append_structure_fingerprints(
    data: pd.DataFrame,
    *,
    structure_column: str = "structure",
    features: Sequence[str] = DEFAULT_STRUCTURE_FEATURES,
    drop_structure_column: bool = False,
    errors: str = "raise",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Append lightweight structure descriptors to a dataframe."""
    if structure_column not in data.columns:
        raise KeyError(f"Structure column '{structure_column}' not found")

    result = featurize_structures(
        data[structure_column],
        features=features,
        errors=errors,
        cache_dir=cache_dir,
    )
    structure_features = result.features.set_index(data.index)
    base = data.drop(columns=[structure_column]) if drop_structure_column else data.copy()
    return pd.concat([base, structure_features], axis=1)


def structure_feature_names(
    features: Sequence[str] = DEFAULT_STRUCTURE_FEATURES,
    *,
    column_prefix: str | None = None,
) -> list[str]:
    """Return validated structure descriptor names with an optional prefix."""
    names = list(_validate_structure_feature_names(features))
    if column_prefix is None:
        return names
    return [f"{column_prefix}_{name}" for name in names]


def _structure_feature_values(structure) -> dict[str, float]:
    if len(structure) == 0:
        raise ValueError("Structure must contain at least one site")

    a, b, c = [float(value) for value in structure.lattice.abc]
    alpha, beta, gamma = [float(value) for value in structure.lattice.angles]
    lengths = sorted((a, b, c))
    angle_cosines = sorted(
        (
            float(np.cos(np.deg2rad(alpha))),
            float(np.cos(np.deg2rad(beta))),
            float(np.cos(np.deg2rad(gamma))),
        )
    )
    volume = float(structure.volume)
    volume_per_atom = volume / len(structure)

    if min(lengths) <= 0 or volume_per_atom <= 0:
        raise ValueError("Structure has non-positive lattice dimensions")

    return {
        "lattice_a": a,
        "lattice_b": b,
        "lattice_c": c,
        "lattice_alpha": alpha,
        "lattice_beta": beta,
        "lattice_gamma": gamma,
        "lattice_length_min": lengths[0],
        "lattice_length_mid": lengths[1],
        "lattice_length_max": lengths[2],
        "log_lattice_length_min": float(np.log(lengths[0])),
        "log_lattice_length_mid": float(np.log(lengths[1])),
        "log_lattice_length_max": float(np.log(lengths[2])),
        "cos_lattice_angle_min": angle_cosines[0],
        "cos_lattice_angle_mid": angle_cosines[1],
        "cos_lattice_angle_max": angle_cosines[2],
        "volume": volume,
        "volume_per_atom": volume_per_atom,
        "log_volume_per_atom": float(np.log(volume_per_atom)),
        "density": float(structure.density),
        "num_sites": float(len(structure)),
        "num_species": float(len(structure.composition.elements)),
    }


def _validate_structure_feature_names(features: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name).strip() for name in features)
    if not names:
        raise ValueError("features must contain at least one descriptor name")
    if any(not name for name in names):
        raise ValueError("features must contain non-empty descriptor names")
    allowed = set(_structure_feature_values(_reference_structure()).keys())
    unsupported = sorted(set(names) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported structure descriptors: {', '.join(unsupported)}")
    return names


def _reference_structure():
    from pymatgen.core import Lattice, Structure

    return Structure(Lattice.cubic(1.0), ["H"], [[0.0, 0.0, 0.0]])


def _coerce_structure(structure):
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ImportError(
            "pymatgen is required for structure fingerprints. "
            "Install it with `conda install -c conda-forge pymatgen`."
        ) from exc

    if isinstance(structure, Structure):
        return structure

    if isinstance(structure, Path):
        return Structure.from_file(str(structure.expanduser()))

    if isinstance(structure, str):
        possible_path = _path_from_short_string(structure)
        if possible_path is not None and possible_path.exists():
            return Structure.from_file(str(possible_path))
        return Structure.from_str(structure, fmt="cif")

    raise TypeError("structure must be a pymatgen Structure, structure-file path, or CIF string")


def _path_from_short_string(value: str) -> Path | None:
    if "\n" in value or len(value) > 512:
        return None
    return Path(value).expanduser()


def _structure_cache_value(structure: object) -> str:
    if isinstance(structure, Path):
        return _path_cache_value(structure)
    if isinstance(structure, str):
        possible_path = _path_from_short_string(structure)
        if possible_path is not None and possible_path.exists():
            return _path_cache_value(possible_path)
        return structure
    if hasattr(structure, "as_dict"):
        return json.dumps(structure.as_dict(), sort_keys=True, default=str)
    return repr(structure)


def _path_cache_value(path: Path) -> str:
    expanded = path.expanduser()
    try:
        stat = expanded.stat()
    except OSError:
        return str(expanded)
    return f"{expanded.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"


def _short_structure_label(structure: object) -> str:
    label = _structure_cache_value(structure)
    return label if len(label) <= 120 else f"{label[:117]}..."


def _read_cached_structure_row(
    cache_dir: str | Path | None,
    *,
    cache_key: str,
    feature_names: Sequence[str],
) -> dict[str, float] | None:
    record = read_fingerprint_cache_record(cache_dir, namespace="structure", cache_key=cache_key)
    if record is None:
        return None
    features = record.get("features")
    if not isinstance(features, Mapping):
        return None
    if any(name not in features for name in feature_names):
        return None
    return {name: float(features[name]) for name in feature_names}


def _write_cached_structure_row(
    cache_dir: str | Path | None,
    *,
    cache_key: str,
    input_value: object,
    features: Mapping[str, float],
    metadata: Mapping[str, object],
) -> None:
    write_fingerprint_cache_record(
        cache_dir,
        namespace="structure",
        cache_key=cache_key,
        record={
            "input": input_value,
            "features": dict(features),
            "metadata": dict(metadata),
        },
    )
