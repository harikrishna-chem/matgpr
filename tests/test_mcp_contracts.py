"""Stable JSON and input contracts for the public read-only MCP surface."""

from __future__ import annotations

import json

import pytest

from matgpr.mcp_prompts import MCP_PROMPT_REGISTRATIONS
from matgpr.mcp_resources import MCP_RESOURCE_REGISTRATIONS
from matgpr.mcp_server import MCP_TOOL_FUNCTIONS
from matgpr.mcp_tools import (
    MCP_HELPER_SCHEMA_VERSION,
    get_matgpr_info,
    get_physics_equation,
    list_capabilities,
    list_physics_equations,
    preview_safe_equation,
    recommend_featurizers,
    suggest_bo_workflow,
    suggest_validation_workflow,
    validate_safe_equation,
)

pytestmark = pytest.mark.contract


_LINEAR_SPEC = {
    "name": "linear_mean",
    "expression": "offset + slope * x",
    "variables": [{"name": "x"}],
    "parameters": [
        {"name": "offset", "initial_value": 0.0},
        {"name": "slope", "initial_value": 1.0},
    ],
}


def test_tool_registry_and_top_level_payload_contracts_are_stable() -> None:
    payloads = {
        "get_matgpr_info": get_matgpr_info(),
        "list_capabilities": list_capabilities(),
        "recommend_featurizers": recommend_featurizers(
            ["formula", "temperature_k", "target"],
            dtype_summary={"temperature_k": "float64", "target": "float64"},
            sample_rows=[{"formula": "Al2O3", "temperature_k": 300.0, "target": 1.0}],
            target_column="target",
        ),
        "list_physics_equations": list_physics_equations(application="diffusion"),
        "get_physics_equation": get_physics_equation("arrhenius"),
        "validate_safe_equation": validate_safe_equation(_LINEAR_SPEC),
        "preview_safe_equation": preview_safe_equation(
            _LINEAR_SPEC,
            variable_values={"x": [1.0, 2.0]},
        ),
        "suggest_validation_workflow": suggest_validation_workflow(n_rows=80),
        "suggest_bo_workflow": suggest_bo_workflow(candidate_count=20, batch_size=2),
    }

    assert tuple(payloads) == tuple(function.__name__ for function in MCP_TOOL_FUNCTIONS)
    assert {name: sorted(payload) for name, payload in payloads.items()} == {
        "get_matgpr_info": [
            "capabilities",
            "capability_names",
            "concept_doi",
            "docs_url",
            "github_url",
            "install",
            "license",
            "name",
            "optional_dependency_groups",
            "pypi_url",
            "python_requires",
            "python_version",
            "schema_version",
            "version",
        ],
        "list_capabilities": ["capabilities", "schema_version"],
        "recommend_featurizers": [
            "column_assessments",
            "notes",
            "physics_feature_map_candidates",
            "recommended_featurizers",
            "sample_row_count_used",
            "schema_version",
            "target_candidates",
            "target_column",
            "task_type",
            "warnings",
        ],
        "list_physics_equations": ["count", "filters", "schema_version", "templates"],
        "get_physics_equation": ["found", "implementation", "schema_version", "template"],
        "validate_safe_equation": [
            "hints",
            "is_valid",
            "normalized_spec",
            "safe_equation_schema",
            "schema_version",
            "validation",
        ],
        "preview_safe_equation": [
            "hints",
            "input",
            "is_valid_equation",
            "normalized_spec",
            "physics_mean_values",
            "preview",
            "preview_ready",
            "schema_version",
            "validation",
            "warnings",
        ],
        "suggest_validation_workflow": [
            "matgpr_apis",
            "n_rows",
            "notes",
            "recommended_protocol",
            "schema_version",
            "task_type",
        ],
        "suggest_bo_workflow": [
            "candidate_count",
            "matgpr_apis",
            "objective_direction",
            "optional_extra",
            "recommended_workflow",
            "schema_version",
            "warnings",
        ],
    }
    for payload in payloads.values():
        assert payload["schema_version"] == MCP_HELPER_SCHEMA_VERSION
        json.dumps(payload, allow_nan=False)


def test_prompt_registration_contract_is_unique_and_complete() -> None:
    records = [
        (record.name, record.title, record.description, record.function.__name__)
        for record in MCP_PROMPT_REGISTRATIONS
    ]

    assert records == [
        (
            "plan_featurization_workflow",
            "Plan Featurization Workflow",
            "Guide an agent through matgpr featurizer selection from dataset schema evidence.",
            "plan_featurization_workflow",
        ),
        (
            "plan_physics_informed_gpr_workflow",
            "Plan Physics-Informed GPR Workflow",
            "Guide an agent through physics-informed GPR setup with explicit physics equations.",
            "plan_physics_informed_gpr_workflow",
        ),
        (
            "plan_validation_workflow",
            "Plan Validation Workflow",
            "Guide an agent through learning curves, splits, metrics, and uncertainty diagnostics.",
            "plan_validation_workflow",
        ),
        (
            "plan_bayesian_optimization_workflow",
            "Plan Bayesian Optimization Workflow",
            "Guide an agent through finite-pool BO and active-learning planning with matgpr.",
            "plan_bayesian_optimization_workflow",
        ),
    ]
    assert len({record.name for record in MCP_PROMPT_REGISTRATIONS}) == len(records)
    for record in MCP_PROMPT_REGISTRATIONS:
        prompt = record.function()
        assert prompt.startswith("# ")
        assert "## Required Agent Steps" in prompt
        assert "## Guardrails" in prompt


def test_resource_registration_contract_is_unique_and_complete() -> None:
    records = [
        (record.uri, record.name, record.mime_type, record.function.__name__)
        for record in MCP_RESOURCE_REGISTRATIONS
    ]

    assert records == [
        (
            "matgpr://guide/capabilities",
            "matgpr_capabilities_guide",
            "text/markdown",
            "capabilities_guide",
        ),
        (
            "matgpr://guide/featurization",
            "matgpr_featurization_guide",
            "text/markdown",
            "featurization_guide",
        ),
        (
            "matgpr://guide/physics-informed-gpr",
            "matgpr_physics_informed_gpr_guide",
            "text/markdown",
            "physics_informed_gpr_guide",
        ),
        (
            "matgpr://guide/validation",
            "matgpr_validation_guide",
            "text/markdown",
            "validation_guide",
        ),
        (
            "matgpr://guide/bayesian-optimization",
            "matgpr_bayesian_optimization_guide",
            "text/markdown",
            "bayesian_optimization_guide",
        ),
    ]
    assert len({record.uri for record in MCP_RESOURCE_REGISTRATIONS}) == len(records)
    assert len({record.name for record in MCP_RESOURCE_REGISTRATIONS}) == len(records)
    for record in MCP_RESOURCE_REGISTRATIONS:
        content = record.function()
        assert content.startswith("# ")
        assert "GenMatics" not in content


def test_recommend_featurizers_rejects_scalar_and_duplicate_column_names() -> None:
    with pytest.raises(TypeError, match="column_names must be a non-string iterable"):
        recommend_featurizers("formula")
    with pytest.raises(ValueError, match="column_names must contain at least one value"):
        recommend_featurizers([])
    with pytest.raises(ValueError, match="column_names must not contain duplicates"):
        recommend_featurizers(["formula", "formula"])


def test_recommend_featurizers_enforces_five_row_safety_limit() -> None:
    payload = recommend_featurizers(
        ["formula"],
        sample_rows=[{"formula": "Al2O3"} for _ in range(8)],
        max_sample_rows=100,
    )

    assert payload["sample_row_count_used"] == 5
    assert any("clipped to 5" in warning for warning in payload["warnings"])


def test_recommend_featurizers_rejects_misaligned_column_oriented_rows() -> None:
    with pytest.raises(ValueError, match="same length"):
        recommend_featurizers(
            ["formula", "target"],
            sample_rows={"formula": ["Al2O3", "SiO2"], "target": [1.0]},
        )


def test_list_physics_equations_rejects_scalar_required_features() -> None:
    with pytest.raises(TypeError, match="required_features must be a non-string iterable"):
        list_physics_equations(required_features="temperature_k")


def test_preview_safe_equation_keeps_one_top_level_shape_across_outcomes() -> None:
    valid = preview_safe_equation(_LINEAR_SPEC, variable_values={"x": [1.0]})
    missing_input = preview_safe_equation(_LINEAR_SPEC)
    invalid = preview_safe_equation(
        {
            "name": "unsafe",
            "expression": "__import__('os')",
            "variables": [{"name": "x"}],
        },
        variable_values={"x": [1.0]},
    )

    for payload in (missing_input, invalid):
        assert set(payload) == set(valid)
        assert payload["physics_mean_values"] == []
        json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("n_rows", [True, 2.5])
def test_validation_workflow_rejects_noninteger_row_counts(n_rows: object) -> None:
    with pytest.raises(TypeError, match="n_rows must be an integer"):
        suggest_validation_workflow(n_rows=n_rows)  # type: ignore[arg-type]


def test_validation_workflow_rejects_negative_rows_and_scalar_metrics() -> None:
    with pytest.raises(ValueError, match="n_rows must be non-negative"):
        suggest_validation_workflow(n_rows=-1)
    with pytest.raises(TypeError, match="metrics must be a non-string iterable"):
        suggest_validation_workflow(metrics="rmse")
    with pytest.raises(ValueError, match="metrics must contain at least one value"):
        suggest_validation_workflow(metrics=[])
    with pytest.raises(TypeError, match="low_data_focus must be a boolean"):
        suggest_validation_workflow(low_data_focus="false")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="show_train_results must be a boolean"):
        suggest_validation_workflow(show_train_results=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("candidate_count", [True, 2.5])
def test_bo_workflow_rejects_noninteger_candidate_counts(candidate_count: object) -> None:
    with pytest.raises(TypeError, match="candidate_count must be an integer"):
        suggest_bo_workflow(candidate_count=candidate_count)  # type: ignore[arg-type]


def test_bo_workflow_rejects_invalid_counts_direction_and_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="candidate_count must be non-negative"):
        suggest_bo_workflow(candidate_count=-1)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        suggest_bo_workflow(batch_size=0)
    with pytest.raises(ValueError, match="objective_direction must be 'maximize' or 'minimize'"):
        suggest_bo_workflow(objective_direction="sideways")
    with pytest.raises(TypeError, match="duplicate_key_columns must be a non-string iterable"):
        suggest_bo_workflow(duplicate_key_columns="formula")
    with pytest.raises(ValueError, match="duplicate_key_columns must not contain duplicates"):
        suggest_bo_workflow(duplicate_key_columns=["formula", "formula"])
    with pytest.raises(TypeError, match="has_constraints must be a boolean"):
        suggest_bo_workflow(has_constraints="false")  # type: ignore[arg-type]


def test_bo_workflow_normalizes_supported_objective_direction() -> None:
    payload = suggest_bo_workflow(objective_direction=" Minimize ")

    assert payload["objective_direction"] == "minimize"
    assert payload["recommended_workflow"]["acquisition"] == "lower_confidence_bound"
