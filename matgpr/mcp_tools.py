"""Read-only helper functions exposed by the optional `matgpr` MCP server."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from ._version import __version__
from .optional_dependencies import list_optional_dependencies

__all__ = [
    "MCP_HELPER_SCHEMA_VERSION",
    "get_matgpr_info",
    "get_physics_equation",
    "list_capabilities",
    "list_physics_equations",
    "recommend_featurizers",
    "suggest_bo_workflow",
    "suggest_validation_workflow",
    "validate_safe_equation",
]


MCP_HELPER_SCHEMA_VERSION = "0.1"
_DOCS_BASE_URL = "https://harikrishnasahu.com/matgpr/"
_PYPI_URL = "https://pypi.org/project/matgpr/"
_GITHUB_URL = "https://github.com/harikrishna-chem/matgpr"
_CONCEPT_DOI = "10.5281/zenodo.21210386"
_MAX_SAMPLE_ROWS = 5

_MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "<na>"}
_FORMULA_PATTERN = re.compile(r"^(?:[A-Z][a-z]?(?:\d+(?:\.\d*)?|\.\d+)*)+$")
_STRUCTURE_SUFFIXES = (".cif", ".poscar", ".vasp", ".json")
_SMILES_CHARS = frozenset("[]=#()@+\\/-.")


def get_matgpr_info() -> dict[str, object]:
    """Return public package metadata for AI-agent discovery."""
    optional_groups = _optional_dependency_groups()
    return {
        "schema_version": MCP_HELPER_SCHEMA_VERSION,
        "name": "matgpr",
        "version": __version__,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "python_requires": ">=3.10",
        "license": "Apache-2.0",
        "docs_url": _DOCS_BASE_URL,
        "pypi_url": _PYPI_URL,
        "github_url": _GITHUB_URL,
        "concept_doi": _CONCEPT_DOI,
        "install": {
            "base": "python -m pip install matgpr",
            "mcp": 'python -m pip install "matgpr[mcp]"',
        },
        "capabilities": [
            "materials featurization",
            "standard GPR",
            "physics-informed GPR",
            "uncertainty diagnostics",
            "validation",
            "finite-pool Bayesian optimization",
        ],
        "optional_dependency_groups": optional_groups,
        "capability_names": [record["name"] for record in list_capabilities()["capabilities"]],
    }


def list_capabilities() -> dict[str, object]:
    """Return a structured map of public `matgpr` capabilities."""
    return {
        "schema_version": MCP_HELPER_SCHEMA_VERSION,
        "capabilities": [
            _capability(
                "data_cleaning",
                "Clean tabular materials datasets before modeling.",
                "api/data/",
                [
                    "normalize_column_names",
                    "replace_missing_placeholders",
                    "drop_duplicate_rows",
                    "impute_missing_values",
                    "filter_iqr_outliers",
                ],
            ),
            _capability(
                "materials_featurization",
                "Create composition, molecule, polymer, and structure descriptors.",
                "fingerprinting_options/",
                [
                    "CompositionFeaturizer",
                    "MagpieCompositionFeaturizer",
                    "SmilesFeaturizer",
                    "PolymerSmilesFeaturizer",
                    "StructureFeaturizer",
                ],
            ),
            _capability(
                "standard_gpr",
                "Fit standard Gaussian Process Regression models with uncertainty.",
                "api/estimators/",
                ["MatGPRRegressor", "fit_gpytorch_gpr", "build_sklearn_gpr_model"],
            ),
            _capability(
                "physics_informed_gpr",
                "Introduce physics through mean functions, constraints, and kernels.",
                "physics_informed_gpr/",
                [
                    "PhysicsInformedGPRRegressor",
                    "PhysicsEquationTemplate",
                    "build_safe_equation_mean_function",
                    "append_virtual_observations",
                    "fit_derivative_constrained_gpr",
                ],
            ),
            _capability(
                "multitask_gpr",
                "Model correlated materials properties with complete or sparse tasks.",
                "multitask_gpr/",
                [
                    "MultitaskGPRRegressor",
                    "SparseMultitaskGPRRegressor",
                    "fit_multitask_gpytorch_gpr",
                    "fit_sparse_multitask_gpytorch_gpr",
                ],
            ),
            _capability(
                "multifidelity_gpr",
                "Fuse low-fidelity and high-fidelity observations.",
                "multifidelity_gpr/",
                [
                    "MultiFidelityGPRRegressor",
                    "CoKrigingGPRRegressor",
                    "fit_delta_multifidelity_gpr",
                    "fit_cokriging_gpr",
                ],
            ),
            _capability(
                "uncertainty_diagnostics",
                "Assess predictive uncertainty, coverage, and calibration.",
                "api/uncertainty/",
                [
                    "prediction_interval",
                    "empirical_coverage",
                    "standardized_residuals",
                    "gaussian_nlpd",
                    "calibration_curve",
                ],
            ),
            _capability(
                "validation",
                "Build learning curves, train/test summaries, and report-ready metrics.",
                "api/validation/",
                [
                    "learning_curve",
                    "train_test_validation",
                    "cross_validate_regressor",
                    "regression_metrics",
                ],
            ),
            _capability(
                "bayesian_optimization",
                "Rank finite candidate pools for next-experiment selection.",
                "api/bayesian_optimization/",
                [
                    "fit_botorch_surrogate",
                    "rank_discrete_candidates",
                    "suggest_next_experiments",
                    "CandidateDuplicatePolicy",
                    "CandidateTrustRegion",
                ],
                optional_extra="bo",
            ),
            _capability(
                "visualization",
                "Create parity, residual, uncertainty, and learning-curve plots.",
                "api/visualization/",
                [
                    "plot_parity",
                    "plot_residuals",
                    "plot_uncertainty_calibration",
                    "plot_learning_curve",
                ],
            ),
        ],
    }


def recommend_featurizers(
    column_names: Sequence[str],
    dtype_summary: Mapping[str, str] | None = None,
    sample_rows: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]] | None = None,
    target_column: str | None = None,
    task_type: str | None = None,
    max_sample_rows: int = _MAX_SAMPLE_ROWS,
) -> dict[str, object]:
    """Recommend public `matgpr` featurizers from column names and small samples.

    The heuristics are intentionally conservative. They inspect only column
    names, dtype hints, and up to a few user-provided sample rows; they do not
    parse files or run model training.
    """
    columns = [str(column) for column in column_names]
    dtypes = {str(key): str(value) for key, value in (dtype_summary or {}).items()}
    rows, sample_warnings = _normalize_sample_rows(sample_rows, max_rows=max_sample_rows)

    assessments = [
        _assess_column(column, dtypes.get(column), rows, target_column=target_column)
        for column in columns
    ]
    role_to_columns = _columns_by_role(assessments)
    featurizers = _recommended_featurizers(role_to_columns)
    physics_maps = _physics_feature_map_candidates(assessments)
    target_candidates = _target_candidates(assessments, target_column=target_column)
    warnings = [*sample_warnings, *_recommendation_warnings(assessments, target_candidates)]

    return {
        "schema_version": MCP_HELPER_SCHEMA_VERSION,
        "task_type": _clean_optional_string(task_type),
        "target_column": _clean_optional_string(target_column),
        "sample_row_count_used": len(rows),
        "column_assessments": assessments,
        "recommended_featurizers": featurizers,
        "physics_feature_map_candidates": physics_maps,
        "target_candidates": target_candidates,
        "warnings": warnings,
        "notes": [
            "Review inferred roles before fitting models.",
            "Use the same featurizer settings for training and prediction.",
            "Scale continuous descriptors before GPR unless a kernel handles them explicitly.",
        ],
    }


def list_physics_equations(
    query: str | None = None,
    application: str | None = None,
    tag: str | None = None,
    required_features: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return public physics-equation templates matching optional filters."""
    from .physics_equations import (
        list_physics_equation_templates,
        search_physics_equation_templates,
    )

    if any(value is not None for value in (query, application, tag, required_features)):
        templates = search_physics_equation_templates(
            query=query,
            application=application,
            tag=tag,
            required_features=required_features,
        )
    else:
        templates = list_physics_equation_templates()
    records = [template.discovery_record() for template in templates]
    return {
        "schema_version": MCP_HELPER_SCHEMA_VERSION,
        "filters": {
            "query": _clean_optional_string(query),
            "application": _clean_optional_string(application),
            "tag": _clean_optional_string(tag),
            "required_features": list(required_features or ()),
        },
        "count": len(records),
        "templates": records,
    }


def get_physics_equation(name: str) -> dict[str, object]:
    """Return one public physics-equation template by name or alias."""
    from .physics_equations import (
        available_physics_equation_templates,
        describe_physics_equation_template,
    )

    try:
        record = describe_physics_equation_template(name)
    except ValueError as exc:
        return {
            "schema_version": MCP_HELPER_SCHEMA_VERSION,
            "found": False,
            "name": str(name),
            "error": str(exc),
            "available_templates": list(available_physics_equation_templates(include_aliases=True)),
        }

    return {
        "schema_version": MCP_HELPER_SCHEMA_VERSION,
        "found": True,
        "template": record,
        "implementation": {
            "physics_entry_point": "mean function",
            "matgpr_api": "get_physics_equation_template(...).build_mean_function(...)",
            "features_used": record["features"],
            "learned_parameters": [
                parameter["name"]
                for parameter in record["parameter_metadata"]
                if parameter["kind"] == "learnable"
            ],
            "fixed_parameters": [
                parameter["name"]
                for parameter in record["parameter_metadata"]
                if parameter["kind"] == "fixed"
            ],
            "feature_map_note": (
                "Map user dataframe columns to the canonical template feature names "
                "before constructing the physics-informed mean function."
            ),
        },
    }


def validate_safe_equation(spec: Mapping[str, Any]) -> dict[str, object]:
    """Validate a safe custom equation spec and return JSON-safe diagnostics."""
    from .safe_equations import (
        SafeEquationSpec,
        safe_equation_schema_snapshot,
        validate_safe_equation_spec,
    )

    validation = validate_safe_equation_spec(spec)
    normalized_spec = None
    if validation.is_valid:
        normalized_spec = SafeEquationSpec.from_dict(spec).to_dict()

    hints = [
        "Declare every symbol as a variable, parameter, or constant.",
        "Use only allowed math functions from the schema snapshot.",
        "Provide units and bounds where they matter physically.",
    ]
    if validation.is_valid:
        hints.append(
            "Use build_safe_equation_mean_function(...) to introduce this as a PI-GPR mean."
        )

    return {
        "schema_version": MCP_HELPER_SCHEMA_VERSION,
        "is_valid": validation.is_valid,
        "validation": validation.to_dict(),
        "normalized_spec": normalized_spec,
        "safe_equation_schema": safe_equation_schema_snapshot(),
        "hints": hints,
    }


def suggest_validation_workflow(
    n_rows: int | None = None,
    task_type: str = "regression",
    low_data_focus: bool = True,
    metrics: Sequence[str] | None = None,
    show_train_results: bool = False,
) -> dict[str, object]:
    """Suggest a validation protocol for public `matgpr` modeling workflows."""
    resolved_metrics = list(metrics or ("rmse", "mae", "r2", "pearson_r"))
    rows = None if n_rows is None else max(int(n_rows), 0)
    low_data = bool(low_data_focus or (rows is not None and rows < 200))
    train_percentages = list(range(10, 101, 10)) if low_data else [20, 40, 60, 80, 100]
    split_count = 20 if low_data else 10
    cv_folds = 5 if rows is not None and rows < 100 else 10

    return {
        "schema_version": MCP_HELPER_SCHEMA_VERSION,
        "task_type": str(task_type),
        "n_rows": rows,
        "recommended_protocol": {
            "learning_curve": {
                "train_percentages": train_percentages,
                "splits_per_point": split_count,
                "report_train_results": bool(show_train_results),
                "report_test_results": True,
                "metrics": resolved_metrics,
            },
            "held_out_test": {
                "default_train_fraction": 0.8 if low_data else 0.9,
                "random_state": 42,
                "stratify": "classification only when class counts allow it",
            },
            "cross_validation": {
                "folds": cv_folds,
                "shuffle": True,
                "random_state": 42,
            },
            "figures": [
                "learning curve with mean and standard deviation error bars",
                "parity plot with predictive uncertainty where available",
                "residual plot",
                "uncertainty calibration plot when predictive standard deviations are available",
            ],
        },
        "matgpr_apis": [
            "learning_curve",
            "train_test_validation",
            "cross_validate_regressor",
            "regression_metrics",
        ],
        "notes": [
            "Use identical splits for standard GPR and PI-GPR comparisons.",
            "For small datasets, report repeated-split variability rather than a single split.",
            "Keep a held-out test set untouched when the final model comparison is important.",
        ],
    }


def suggest_bo_workflow(
    candidate_count: int | None = None,
    objective_direction: str = "maximize",
    batch_size: int = 1,
    has_constraints: bool = False,
    duplicate_key_columns: Sequence[str] | None = None,
) -> dict[str, object]:
    """Suggest a finite-pool Bayesian-optimization workflow."""
    count = None if candidate_count is None else max(int(candidate_count), 0)
    duplicate_policy = (
        "Use CandidateDuplicatePolicy with key columns and feature distance checks."
        if duplicate_key_columns
        else "Add duplicate key columns when material identity, formula, or SMILES are available."
    )
    acquisition = (
        "expected_improvement" if objective_direction == "maximize" else "lower_confidence_bound"
    )

    return {
        "schema_version": MCP_HELPER_SCHEMA_VERSION,
        "candidate_count": count,
        "objective_direction": str(objective_direction),
        "recommended_workflow": {
            "candidate_pool": [
                "Use the same featurization pipeline as the measured training data.",
                "Run candidate-pool diagnostics before acquisition scoring.",
                "Remove rows with missing required descriptors before ranking.",
            ],
            "surrogate": "Fit a GPR surrogate with predictive mean and standard deviation.",
            "acquisition": acquisition,
            "batch_size": max(int(batch_size), 1),
            "constraints": (
                "Apply CandidateConstraint before ranking."
                if has_constraints
                else "No feasibility constraints requested."
            ),
            "duplicates": duplicate_policy,
            "diversity": (
                "Use select_diverse_batch for batches larger than one."
                if int(batch_size) > 1
                else "Single-candidate recommendation does not need batch diversity."
            ),
            "audit": "Save recommendation tables, scores, duplicate flags, and exclusion reasons.",
        },
        "matgpr_apis": [
            "summarize_candidate_pool",
            "fit_botorch_surrogate",
            "rank_discrete_candidates",
            "suggest_next_experiments",
            "log_bo_recommendations",
            "summarize_bo_recommendation_audit",
        ],
        "optional_extra": "bo",
        "warnings": _bo_warnings(count),
    }


def _capability(
    name: str,
    summary: str,
    docs_path: str,
    public_apis: Sequence[str],
    *,
    optional_extra: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "summary": summary,
        "docs_url": f"{_DOCS_BASE_URL}{docs_path}",
        "public_apis": list(public_apis),
        "optional_extra": optional_extra,
    }


def _optional_dependency_groups() -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for dependency in list_optional_dependencies():
        groups.setdefault(dependency.extra, []).append(
            {
                "import_name": dependency.import_name,
                "package_name": dependency.package_name,
                "purpose": dependency.purpose,
            }
        )
    return {
        extra: sorted(records, key=lambda record: record["import_name"])
        for extra, records in sorted(groups.items())
    }


def _normalize_sample_rows(
    sample_rows: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]] | None,
    *,
    max_rows: int,
) -> tuple[list[dict[str, object]], list[str]]:
    if sample_rows is None:
        return [], []
    if max_rows < 1:
        raise ValueError("max_sample_rows must be at least 1")

    warnings: list[str] = []
    if hasattr(sample_rows, "head") and hasattr(sample_rows, "to_dict"):
        records = sample_rows.head(max_rows).to_dict("records")  # type: ignore[union-attr]
        total = len(sample_rows)  # type: ignore[arg-type]
    elif isinstance(sample_rows, Mapping):
        items = [(str(key), value) for key, value in sample_rows.items()]
        lengths = [len(value) for value in sample_rows.values() if _has_length(value)]
        total = min(lengths) if lengths else 0
        records = [
            {key: _sequence_value(value, index) for key, value in items}
            for index in range(min(total, max_rows))
        ]
    else:
        records = [dict(row) for row in sample_rows[:max_rows]]
        total = len(sample_rows)

    if total > max_rows:
        warnings.append(f"sample_rows clipped to the first {max_rows} row(s)")
    return [_json_safe_row(row) for row in records], warnings


def _assess_column(
    column: str,
    dtype: str | None,
    rows: Sequence[Mapping[str, object]],
    *,
    target_column: str | None,
) -> dict[str, object]:
    values = [_present_value(row.get(column)) for row in rows if _present_value(row.get(column))]
    name = _normalize_text(column)
    numeric = _is_numeric_dtype(dtype) or _mostly_numeric(values)

    evidence: list[str] = []
    role = "unknown"
    confidence = 0.2

    if target_column is not None and column == target_column:
        return _column_record(column, dtype, "target", 0.98, ["matches target_column"], values)

    if _is_polymer_column(name, values):
        role = "polymer_smiles"
        confidence = 0.9
        evidence.append("column name or values indicate polymer repeat-unit SMILES")
        if any(str(value).count("[*]") != 2 for value in values if "[*]" in str(value)):
            evidence.append("some polymer samples do not contain exactly two [*] dummy atoms")
    elif _is_smiles_column(name, values):
        role = "molecule_smiles"
        confidence = 0.86
        evidence.append("column name or values indicate molecule SMILES")
    elif _is_structure_column(name, values):
        role = "structure"
        confidence = 0.82
        evidence.append("column name or values indicate crystal-structure records or files")
    elif _is_composition_column(name, values):
        role = "composition"
        confidence = 0.82
        evidence.append("column name or values indicate inorganic formulas/compositions")
    elif _is_likely_target_name(name):
        role = "target_candidate"
        confidence = 0.72 if numeric else 0.5
        evidence.append("column name resembles a materials property target")
    elif numeric and _is_processing_name(name):
        role = "processing_condition"
        confidence = 0.78
        evidence.append("numeric column name resembles a process or measurement condition")
    elif numeric:
        role = "numeric_feature"
        confidence = 0.62
        evidence.append("numeric dtype or numeric sample values")
    elif values:
        role = "categorical_feature"
        confidence = 0.5
        evidence.append("non-numeric values with no recognized materials syntax")

    return _column_record(column, dtype, role, confidence, evidence, values)


def _column_record(
    column: str,
    dtype: str | None,
    role: str,
    confidence: float,
    evidence: Sequence[str],
    values: Sequence[object],
) -> dict[str, object]:
    return {
        "column": column,
        "dtype": dtype,
        "inferred_role": role,
        "confidence": round(float(confidence), 3),
        "evidence": list(evidence),
        "non_missing_sample_count": len(values),
        "matgpr_apis": _role_apis(role),
        "recommended_action": _role_action(role),
    }


def _role_apis(role: str) -> list[str]:
    mapping = {
        "composition": ["CompositionFeaturizer", "append_composition_fingerprints"],
        "molecule_smiles": ["SmilesFeaturizer", "featurize_smiles"],
        "polymer_smiles": ["PolymerSmilesFeaturizer", "canonicalize_polymer_smiles"],
        "structure": ["StructureFeaturizer", "append_structure_fingerprints"],
        "processing_condition": ["build_preprocessor", "build_scaler"],
        "numeric_feature": ["build_preprocessor", "build_scaler"],
        "target": ["separate_features_target", "train_test_validation"],
        "target_candidate": ["separate_features_target", "train_test_validation"],
        "categorical_feature": ["build_preprocessor"],
    }
    return mapping.get(role, [])


def _role_action(role: str) -> str:
    mapping = {
        "composition": "Featurize with CompositionFeaturizer; consider Magpie with materials-extra.",
        "molecule_smiles": "Canonicalize and featurize with SmilesFeaturizer.",
        "polymer_smiles": "Use PolymerSmilesFeaturizer; it canonicalizes cyclic trimer surrogates.",
        "structure": "Use StructureFeaturizer for lightweight descriptors.",
        "processing_condition": "Keep as continuous numeric features after scaling.",
        "numeric_feature": "Use as numeric model feature after scaling.",
        "target": "Use as the supervised prediction target.",
        "target_candidate": "Ask the user to confirm whether this is the target.",
        "categorical_feature": "Encode or review as metadata before modeling.",
    }
    return mapping.get(role, "Review manually before modeling.")


def _columns_by_role(assessments: Sequence[Mapping[str, object]]) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    for record in assessments:
        roles.setdefault(str(record["inferred_role"]), []).append(str(record["column"]))
    return roles


def _recommended_featurizers(
    role_to_columns: Mapping[str, Sequence[str]],
) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    if role_to_columns.get("composition"):
        recommendations.append(
            {
                "input_role": "composition",
                "columns": list(role_to_columns["composition"]),
                "recommended_api": "CompositionFeaturizer",
                "starter_configuration": {"errors": "coerce", "return_dataframe": True},
                "optional_upgrade": "MagpieCompositionFeaturizer with matgpr[materials-extra]",
            }
        )
    if role_to_columns.get("molecule_smiles"):
        recommendations.append(
            {
                "input_role": "molecule_smiles",
                "columns": list(role_to_columns["molecule_smiles"]),
                "recommended_api": "SmilesFeaturizer",
                "starter_configuration": {
                    "smiles_type": "molecule",
                    "fingerprint_type": "morgan+descriptors",
                    "n_bits": 1024,
                    "radius": 2,
                },
            }
        )
    if role_to_columns.get("polymer_smiles"):
        recommendations.append(
            {
                "input_role": "polymer_smiles",
                "columns": list(role_to_columns["polymer_smiles"]),
                "recommended_api": "PolymerSmilesFeaturizer",
                "starter_configuration": {
                    "fingerprint_type": "morgan+descriptors",
                    "n_bits": 1024,
                    "radius": 2,
                },
                "note": "Repeat units should contain exactly two [*] dummy atoms.",
            }
        )
    if role_to_columns.get("structure"):
        recommendations.append(
            {
                "input_role": "structure",
                "columns": list(role_to_columns["structure"]),
                "recommended_api": "StructureFeaturizer",
                "starter_configuration": {"errors": "coerce", "return_dataframe": True},
                "optional_upgrade": "Use matgpr[structures] for heavier structure descriptors later.",
            }
        )
    direct_columns = [
        *role_to_columns.get("processing_condition", ()),
        *role_to_columns.get("numeric_feature", ()),
    ]
    if direct_columns:
        recommendations.append(
            {
                "input_role": "numeric_features",
                "columns": direct_columns,
                "recommended_api": "build_preprocessor",
                "starter_configuration": {"scale_numeric": True},
            }
        )
    return recommendations


def _physics_feature_map_candidates(
    assessments: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for record in assessments:
        column = str(record["column"])
        name = _normalize_text(column)
        for canonical, reason, note in _canonical_physics_feature_matches(name):
            candidates.append(
                {
                    "column": column,
                    "canonical_feature": canonical,
                    "reason": reason,
                    "note": note,
                }
            )
    return candidates


def _canonical_physics_feature_matches(name: str) -> list[tuple[str, str, str]]:
    matches: list[tuple[str, str, str]] = []
    if any(token in name for token in ("temperature", "temp", "anneal_t")):
        note = "Convert Celsius to Kelvin before using Arrhenius templates." if "c" in name else ""
        matches.append(("temperature_k", "temperature-like column", note))
    if any(token in name for token in ("time", "duration", "aging", "exposure")):
        matches.append(("time", "time or exposure column", "Keep time units consistent."))
    if "grain" in name and "size" in name:
        matches.append(("grain_size", "grain-size column", "Use only for Hall-Petch-like trends."))
    if "free_volume" in name or "fractional_free_volume" in name or name == "fvf":
        matches.append(("free_volume_fraction", "free-volume column", "Values should be positive."))
    if "volume_fraction" in name or "vol_frac" in name or name in {"phi", "fraction"}:
        matches.append(
            ("volume_fraction", "mixture-fraction column", "Values should be in [0, 1].")
        )
    if any(token in name for token in ("load", "rate", "concentration", "field")):
        matches.append(
            ("driving_variable", "generic driving variable", "Use for power-law templates.")
        )
    return matches


def _target_candidates(
    assessments: Sequence[Mapping[str, object]],
    *,
    target_column: str | None,
) -> list[dict[str, object]]:
    candidates = []
    for record in assessments:
        role = str(record["inferred_role"])
        if role in {"target", "target_candidate"}:
            candidates.append(
                {
                    "column": record["column"],
                    "confidence": record["confidence"],
                    "confirmed": bool(target_column and record["column"] == target_column),
                }
            )
    return candidates


def _recommendation_warnings(
    assessments: Sequence[Mapping[str, object]],
    target_candidates: Sequence[Mapping[str, object]],
) -> list[str]:
    warnings: list[str] = []
    if not target_candidates:
        warnings.append("No target column was confidently identified.")
    unknown = [record["column"] for record in assessments if record["inferred_role"] == "unknown"]
    if unknown:
        warnings.append(f"Review unrecognized columns manually: {', '.join(map(str, unknown))}")
    return warnings


def _is_polymer_column(name: str, values: Sequence[object]) -> bool:
    return (
        ("polymer" in name and "smiles" in name)
        or "repeat_unit" in name
        or any("[*]" in str(value) for value in values)
    )


def _is_smiles_column(name: str, values: Sequence[object]) -> bool:
    return "smiles" in name or any(_looks_like_smiles(value) for value in values)


def _is_structure_column(name: str, values: Sequence[object]) -> bool:
    if any(token in name for token in ("structure", "crystal", "cif", "poscar")):
        return True
    return any(_looks_like_structure_value(value) for value in values)


def _is_composition_column(name: str, values: Sequence[object]) -> bool:
    if any(token in name for token in ("formula", "composition", "stoichiometry")):
        return True
    return bool(values) and sum(_looks_like_formula(value) for value in values) / len(values) >= 0.6


def _is_processing_name(name: str) -> bool:
    return any(
        token in name
        for token in (
            "temperature",
            "temp",
            "time",
            "pressure",
            "strain_rate",
            "rate",
            "load",
            "anneal",
            "aging",
            "exposure",
            "concentration",
            "ph",
            "humidity",
            "speed",
        )
    )


def _is_likely_target_name(name: str) -> bool:
    return any(
        token in name
        for token in (
            "target",
            "property",
            "pce",
            "efficiency",
            "diffusivity",
            "permeability",
            "conductivity",
            "hardness",
            "strength",
            "modulus",
            "band_gap",
            "energy",
            "yield",
            "elongation",
            "toughness",
        )
    )


def _looks_like_formula(value: object) -> bool:
    text = _present_value(value)
    if not text or len(text) > 80:
        return False
    if any(char in text for char in "[]=#@\\/"):
        return False
    compact = text.replace(" ", "")
    return bool(_FORMULA_PATTERN.fullmatch(compact))


def _looks_like_smiles(value: object) -> bool:
    text = _present_value(value)
    if not text or len(text) > 250:
        return False
    if "[*]" in text:
        return True
    if _is_number(text):
        return False
    if _looks_like_formula(text):
        return False
    if any(char in _SMILES_CHARS for char in text):
        return True
    return bool(re.search(r"[cnops]", text)) and not _looks_like_formula(text)


def _looks_like_structure_value(value: object) -> bool:
    text = _present_value(value).lower()
    if not text:
        return False
    return (
        text.startswith("data_")
        or "_cell_length" in text
        or any(text.endswith(suffix) for suffix in _STRUCTURE_SUFFIXES)
    )


def _mostly_numeric(values: Sequence[object]) -> bool:
    if not values:
        return False
    return sum(_is_number(value) for value in values) / len(values) >= 0.8


def _is_number(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_numeric_dtype(dtype: str | None) -> bool:
    if dtype is None:
        return False
    normalized = dtype.lower()
    return any(token in normalized for token in ("int", "float", "double", "number", "numeric"))


def _present_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _MISSING_STRINGS else text


def _normalize_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _clean_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _has_length(value: object) -> bool:
    try:
        len(value)  # type: ignore[arg-type]
    except TypeError:
        return False
    return True


def _sequence_value(value: Sequence[Any], index: int) -> object:
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        return None


def _json_safe_row(row: Mapping[str, Any]) -> dict[str, object]:
    return {str(key): _json_safe_scalar(value) for key, value in row.items()}


def _json_safe_scalar(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    return str(value)


def _bo_warnings(candidate_count: int | None) -> list[str]:
    warnings: list[str] = []
    if candidate_count is None:
        warnings.append("Candidate count was not provided.")
    elif candidate_count == 0:
        warnings.append("Candidate pool is empty.")
    elif candidate_count < 10:
        warnings.append("Candidate pool is very small; BO ranking may add limited value.")
    return warnings
