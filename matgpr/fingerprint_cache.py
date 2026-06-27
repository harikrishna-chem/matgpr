from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "FINGERPRINT_CACHE_SCHEMA_VERSION",
    "fingerprint_cache_key",
    "fingerprint_cache_path",
    "read_fingerprint_cache_record",
    "write_fingerprint_cache_record",
]

FINGERPRINT_CACHE_SCHEMA_VERSION = "matgpr-fingerprint-cache-v1"
_SAFE_NAMESPACE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def fingerprint_cache_key(
    *,
    namespace: str,
    value: object,
    parameters: Mapping[str, object],
) -> str:
    """Return a deterministic cache key for one fingerprint input.

    The key is based on a versioned JSON payload containing the fingerprint
    namespace, the input value, and all featurization parameters. The same
    input and settings produce the same SHA-256 hash across Python sessions.
    """
    payload = {
        "schema_version": FINGERPRINT_CACHE_SCHEMA_VERSION,
        "namespace": namespace,
        "value": _json_ready(value),
        "parameters": _json_ready(parameters),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def fingerprint_cache_path(
    cache_dir: str | Path,
    *,
    namespace: str,
    cache_key: str,
) -> Path:
    """Return the JSON file path for a cache key."""
    return Path(cache_dir) / _safe_namespace(namespace) / f"{cache_key}.json"


def read_fingerprint_cache_record(
    cache_dir: str | Path | None,
    *,
    namespace: str,
    cache_key: str,
) -> dict[str, Any] | None:
    """Read a cached fingerprint record, returning ``None`` for misses."""
    if cache_dir is None:
        return None

    path = fingerprint_cache_path(cache_dir, namespace=namespace, cache_key=cache_key)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(record, dict):
        return None
    if record.get("schema_version") != FINGERPRINT_CACHE_SCHEMA_VERSION:
        return None
    if record.get("namespace") != namespace:
        return None
    if record.get("cache_key") != cache_key:
        return None
    return record


def write_fingerprint_cache_record(
    cache_dir: str | Path | None,
    *,
    namespace: str,
    cache_key: str,
    record: Mapping[str, object],
) -> None:
    """Write one fingerprint cache record as JSON.

    Writes are atomic within the cache directory: data are first written to a
    temporary sibling file, then moved into place.
    """
    if cache_dir is None:
        return

    path = fingerprint_cache_path(cache_dir, namespace=namespace, cache_key=cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FINGERPRINT_CACHE_SCHEMA_VERSION,
        "namespace": namespace,
        "cache_key": cache_key,
        **dict(record),
    }
    temporary_path = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
    temporary_path.replace(path)


def _safe_namespace(namespace: str) -> str:
    safe = _SAFE_NAMESPACE_PATTERN.sub("_", namespace.strip())
    if not safe:
        raise ValueError("cache namespace must not be empty")
    return safe


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return str(value)
