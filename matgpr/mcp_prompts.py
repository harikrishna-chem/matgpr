"""Read-only prompt templates exposed by the optional `matgpr` MCP server."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

__all__ = [
    "MCPPromptRegistration",
    "MCP_PROMPT_REGISTRATIONS",
    "plan_bayesian_optimization_workflow",
    "plan_featurization_workflow",
    "plan_physics_informed_gpr_workflow",
    "plan_validation_workflow",
]


class MCPPromptRegistration(NamedTuple):
    """Metadata needed to register a prompt with an MCP server."""

    name: str
    title: str
    description: str
    function: Callable[..., str]


def plan_featurization_workflow(
    columns: str = "",
    sample_rows: str = "",
    target_column: str = "",
    task_type: str = "regression",
) -> str:
    """Guide an agent through `matgpr` featurizer selection."""
    return _workflow_prompt(
        title="Plan a matgpr featurization workflow",
        objective=(
            "Choose documented matgpr featurizers for a materials-informatics dataset "
            "without guessing column roles."
        ),
        context={
            "columns": columns,
            "sample_rows": sample_rows,
            "target_column": target_column,
            "task_type": task_type,
        },
        steps=[
            "Call recommend_featurizers with column names, dtype hints, sample rows, target column, and task type.",
            "Classify only suitable columns as composition, molecule SMILES, polymer SMILES, structure, numeric process condition, or target.",
            "Ask the user to confirm ambiguous or low-confidence columns before training.",
            "Use matgpr featurizer classes such as CompositionFeaturizer, SmilesFeaturizer, PolymerSmilesFeaturizer, and StructureFeaturizer.",
            "Use documented fingerprint helpers instead of writing notebook-only descriptors.",
            "Record failed rows, units, and preprocessing choices in the notebook or report.",
        ],
        tools=[
            "recommend_featurizers",
            "list_capabilities",
            "get_matgpr_info",
        ],
        docs=[
            "docs/fingerprinting_options.md",
            "docs/api/featurizers.md",
            "docs/api/fingerprints.md",
            "docs/api/data.md",
        ],
        guardrails=[
            "Do not fingerprint arbitrary text columns.",
            "For polymer SMILES, confirm repeat-unit dummy atoms before cyclic-trimer handling.",
            "Keep heavy optional descriptor packages opt-in.",
        ],
    )


def plan_physics_informed_gpr_workflow(
    target_property: str = "",
    candidate_equation: str = "",
    feature_columns: str = "",
    sample_rows: str = "",
) -> str:
    """Guide an agent through PI-GPR mean-function setup."""
    return _workflow_prompt(
        title="Plan a matgpr physics-informed GPR workflow",
        objective=(
            "Set up a physics-informed GPR model by introducing physics through a "
            "documented mean function and explicit feature map."
        ),
        context={
            "target_property": target_property,
            "candidate_equation": candidate_equation,
            "feature_columns": feature_columns,
            "sample_rows": sample_rows,
        },
        steps=[
            "Search built-in physics equation templates before proposing a custom equation.",
            "If a custom equation is needed, validate it with validate_safe_equation before model fitting.",
            "Preview a valid equation on a few representative rows with preview_safe_equation.",
            "Define an explicit physics_feature_map from dataset columns to equation variables.",
            "State learned parameters, fixed constants, units, and assumptions in the report.",
            "Compare against standard GPR with the same split protocol and uncertainty diagnostics.",
        ],
        tools=[
            "list_physics_equations",
            "get_physics_equation",
            "validate_safe_equation",
            "preview_safe_equation",
            "suggest_validation_workflow",
        ],
        docs=[
            "docs/physics_informed_gpr.md",
            "docs/pi_gpr_guarantees.md",
            "docs/api/physics_equations.md",
            "docs/api/safe_equations.md",
            "docs/api/estimators.md",
        ],
        guardrails=[
            "Do not hard-code user column names inside the physics equation.",
            "Do not claim a physics model is better without a held-out or repeated-split comparison.",
            "Avoid nonstandard physics unless the user explicitly approves the assumption.",
        ],
    )


def plan_validation_workflow(
    n_rows: str = "",
    task_type: str = "regression",
    metrics: str = "rmse, mae, r2, pearson_r",
    low_data_focus: str = "true",
) -> str:
    """Guide an agent through validation and uncertainty diagnostics."""
    return _workflow_prompt(
        title="Plan a matgpr validation workflow",
        objective=(
            "Choose a reproducible validation protocol for a materials model, with "
            "low-data learning curves and uncertainty checks where appropriate."
        ),
        context={
            "n_rows": n_rows,
            "task_type": task_type,
            "metrics": metrics,
            "low_data_focus": low_data_focus,
        },
        steps=[
            "Call suggest_validation_workflow with row count, task type, low-data focus, and desired metrics.",
            "Use Learning curves with repeated train/test splits when the dataset is small.",
            "Report RMSE, MAE, R2, and Pearson r only where each metric is scientifically useful.",
            "For GPR-style models, include predictive uncertainty, coverage, calibration, and residual checks.",
            "Save split seeds, feature columns, target transforms, and preprocessing choices.",
        ],
        tools=[
            "suggest_validation_workflow",
            "list_capabilities",
        ],
        docs=[
            "docs/api/validation.md",
            "docs/api/uncertainty.md",
            "docs/api/visualization.md",
            "docs/benchmark_summary.md",
        ],
        guardrails=[
            "Do not tune model choices on the final held-out test set.",
            "Do not compare models trained with different preprocessing unless the difference is intentional and reported.",
        ],
    )


def plan_bayesian_optimization_workflow(
    objective: str = "",
    candidate_pool_columns: str = "",
    constraints: str = "",
    batch_size: str = "",
) -> str:
    """Guide an agent through finite-pool BO planning with `matgpr`."""
    return _workflow_prompt(
        title="Plan a matgpr Bayesian optimization workflow",
        objective=(
            "Design a finite-pool Bayesian optimization or active-learning workflow "
            "that uses documented matgpr recommendation utilities."
        ),
        context={
            "objective": objective,
            "candidate_pool_columns": candidate_pool_columns,
            "constraints": constraints,
            "batch_size": batch_size,
        },
        steps=[
            "Call suggest_bo_workflow with objective direction, candidate count, batch size, constraints, and duplicate-key columns.",
            "Use an already trained model that returns predictive mean and uncertainty for candidate rows.",
            "Select acquisition mode deliberately: maximum uncertainty, expected improvement, probability of improvement, or UCB.",
            "Apply feasibility constraints, duplicate avoidance, and diversity-aware batch selection when relevant.",
            "Return recommendation tables with prediction, uncertainty, acquisition score, feasibility, and duplicate status.",
            "Log campaign configuration and selected candidates so the next experiment cycle is reproducible.",
        ],
        tools=[
            "suggest_bo_workflow",
            "list_capabilities",
            "get_matgpr_info",
        ],
        docs=[
            "docs/api/bayesian_optimization.md",
            "docs/api/candidate_generation.md",
            "docs/api/multi_objective.md",
            "docs/api/experiment_logging.md",
            "docs/api/bo_benchmarking.md",
        ],
        guardrails=[
            "Do not run BO on rows that were already measured unless duplicate selection is intentional.",
            "Do not hide feasibility or diversity filters from the recommendation report.",
            "Keep BoTorch an optional dependency and explain how to install the BO extra.",
        ],
    )


MCP_PROMPT_REGISTRATIONS = (
    MCPPromptRegistration(
        name="plan_featurization_workflow",
        title="Plan Featurization Workflow",
        description="Guide an agent through matgpr featurizer selection from dataset schema evidence.",
        function=plan_featurization_workflow,
    ),
    MCPPromptRegistration(
        name="plan_physics_informed_gpr_workflow",
        title="Plan Physics-Informed GPR Workflow",
        description="Guide an agent through physics-informed GPR setup with explicit physics equations.",
        function=plan_physics_informed_gpr_workflow,
    ),
    MCPPromptRegistration(
        name="plan_validation_workflow",
        title="Plan Validation Workflow",
        description="Guide an agent through learning curves, splits, metrics, and uncertainty diagnostics.",
        function=plan_validation_workflow,
    ),
    MCPPromptRegistration(
        name="plan_bayesian_optimization_workflow",
        title="Plan Bayesian Optimization Workflow",
        description="Guide an agent through finite-pool BO and active-learning planning with matgpr.",
        function=plan_bayesian_optimization_workflow,
    ),
)


def _workflow_prompt(
    *,
    title: str,
    objective: str,
    context: dict[str, str],
    steps: Sequence[str],
    tools: Sequence[str],
    docs: Sequence[str],
    guardrails: Sequence[str],
) -> str:
    return "\n\n".join(
        [
            f"# {title}",
            f"Objective: {objective}",
            _section("User Context", _context_lines(context)),
            _section("Required Agent Steps", _numbered_lines(steps)),
            _section("Relevant MCP Tools", _bullet_lines(tools)),
            _section("Documentation To Check", _bullet_lines(docs)),
            _section("Guardrails", _bullet_lines(guardrails)),
        ]
    )


def _context_lines(context: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for key, value in context.items():
        cleaned = str(value).strip()
        if cleaned:
            lines.append(f"- {key}: {cleaned}")
    return lines or ["- No user context supplied yet."]


def _numbered_lines(values: Sequence[str]) -> list[str]:
    return [f"{index}. {value}" for index, value in enumerate(values, start=1)]


def _bullet_lines(values: Sequence[str]) -> list[str]:
    return [f"- {value}" for value in values]


def _section(title: str, lines: Sequence[str]) -> str:
    return "\n".join([f"## {title}", *lines])
