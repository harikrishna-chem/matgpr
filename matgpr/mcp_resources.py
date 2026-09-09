"""Read-only guide resources exposed by the optional `matgpr` MCP server."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

__all__ = [
    "MCPResourceRegistration",
    "MCP_RESOURCE_REGISTRATIONS",
    "bayesian_optimization_guide",
    "capabilities_guide",
    "featurization_guide",
    "physics_informed_gpr_guide",
    "validation_guide",
]


class MCPResourceRegistration(NamedTuple):
    """Metadata needed to register a static resource with an MCP server."""

    uri: str
    name: str
    title: str
    description: str
    mime_type: str
    function: Callable[[], str]


def capabilities_guide() -> str:
    """Return a compact overview of public `matgpr` MCP capabilities."""
    return """# matgpr MCP Capabilities

The public matgpr MCP server is a local, read-only companion to the open-source
matgpr package. It helps AI agents discover package APIs and plan materials
informatics workflows without guessing function names.

## Available Tool Families

- Package metadata and install guidance: `get_matgpr_info`, `list_capabilities`.
- Featurizer recommendations from schema evidence: `recommend_featurizers`.
- Physics equation discovery: `list_physics_equations`, `get_physics_equation`.
- Safe custom equation checks: `validate_safe_equation`, `preview_safe_equation`.
- Validation planning: `suggest_validation_workflow`.
- Finite-pool Bayesian optimization planning: `suggest_bo_workflow`.

## Safety Boundary

- Local STDIO server only.
- No file reads or writes.
- No model fitting.
- No BO execution.
- No shell or arbitrary Python execution.
- No downstream project-specific logic.
"""


def featurization_guide() -> str:
    """Return a compact featurization guide for MCP clients."""
    return """# matgpr Featurization Guide

Use `recommend_featurizers` before writing featurization code. Pass column
names, dtype hints, sample rows, target column, and task type.

## Common Column Types

- Inorganic formula or composition: use `CompositionFeaturizer` or
  `MagpieCompositionFeaturizer` when optional matminer descriptors are
  available.
- Molecule SMILES: use `SmilesFeaturizer` and RDKit fingerprints/descriptors.
- Polymer SMILES with two `[*]` dummy atoms: use `PolymerSmilesFeaturizer`;
  matgpr can build a cyclic trimer before fingerprinting.
- Structure objects or files: use `StructureFeaturizer` and lightweight
  structure descriptors, with optional ASE/DScribe extras when needed.
- Numeric processing conditions: keep as numeric features after cleaning,
  unit checks, and scaling when appropriate.

## Guardrails

- Do not fingerprint arbitrary text columns.
- Keep optional heavy descriptor backends opt-in.
- Report failed rows and ambiguous columns instead of silently dropping them.
"""


def physics_informed_gpr_guide() -> str:
    """Return a compact PI-GPR setup guide for MCP clients."""
    return """# matgpr Physics-Informed GPR Guide

matgpr supports physics-informed GPR primarily by modifying the GP mean
function. The physics equation produces a prior mean; the GP then learns
residual structure from data.

## Recommended Flow

1. Search built-in templates with `list_physics_equations`.
2. Inspect a candidate with `get_physics_equation`.
3. If the user supplies a custom equation, call `validate_safe_equation`.
4. Preview valid equations on a few rows with `preview_safe_equation`.
5. Define an explicit `physics_feature_map` from dataset columns to equation
   variables.
6. Train and compare standard GPR and PI-GPR using the same validation
   protocol.

## Report

Always state the equation, feature map, learned parameters, fixed constants,
units, assumptions, and whether the PI-GPR model improves over the standard GPR
baseline under the same data split.
"""


def validation_guide() -> str:
    """Return a compact validation guide for MCP clients."""
    return """# matgpr Validation Guide

Use `suggest_validation_workflow` to choose a split protocol from dataset size,
task type, metrics, and low-data focus.

## Recommended Outputs

- Learning curves over training percentage or training count.
- Repeated splits with mean and standard deviation.
- Held-out parity plots with uncertainty error bars when available.
- RMSE, MAE, R2, and Pearson r where appropriate.
- Residual plots and uncertainty calibration diagnostics for GPR models.

## Guardrails

- Keep the final test set separate from model selection.
- Use the same preprocessing and split protocol when comparing models.
- Save random seeds, feature columns, target transform, and model config.
"""


def bayesian_optimization_guide() -> str:
    """Return a compact finite-pool BO guide for MCP clients."""
    return """# matgpr Bayesian Optimization Guide

Use `suggest_bo_workflow` before writing finite-pool BO code. The first public
MCP server plans BO workflows but does not execute BO.

## Recommended Flow

1. Train a model that returns predictive mean and uncertainty.
2. Prepare a finite candidate pool with duplicate-key columns.
3. Choose acquisition behavior: maximum uncertainty, expected improvement,
   probability of improvement, or upper confidence bound.
4. Apply feasibility constraints, duplicate avoidance, and diversity-aware
   batch selection when relevant.
5. Return a recommendation table with predictions, uncertainty, acquisition
   score, feasibility status, duplicate status, and rank.
6. Log campaign settings and selected candidates for reproducibility.

## Optional Dependencies

Install `matgpr[bo]` for BoTorch-backed utilities.
"""


MCP_RESOURCE_REGISTRATIONS = (
    MCPResourceRegistration(
        uri="matgpr://guide/capabilities",
        name="matgpr_capabilities_guide",
        title="matgpr MCP Capabilities",
        description="Compact overview of public matgpr MCP tools and safety boundaries.",
        mime_type="text/markdown",
        function=capabilities_guide,
    ),
    MCPResourceRegistration(
        uri="matgpr://guide/featurization",
        name="matgpr_featurization_guide",
        title="matgpr Featurization Guide",
        description="Guide to choosing matgpr featurizers from materials dataset columns.",
        mime_type="text/markdown",
        function=featurization_guide,
    ),
    MCPResourceRegistration(
        uri="matgpr://guide/physics-informed-gpr",
        name="matgpr_physics_informed_gpr_guide",
        title="matgpr Physics-Informed GPR Guide",
        description="Guide to setting up PI-GPR mean functions and feature maps.",
        mime_type="text/markdown",
        function=physics_informed_gpr_guide,
    ),
    MCPResourceRegistration(
        uri="matgpr://guide/validation",
        name="matgpr_validation_guide",
        title="matgpr Validation Guide",
        description="Guide to matgpr validation protocols and uncertainty diagnostics.",
        mime_type="text/markdown",
        function=validation_guide,
    ),
    MCPResourceRegistration(
        uri="matgpr://guide/bayesian-optimization",
        name="matgpr_bayesian_optimization_guide",
        title="matgpr Bayesian Optimization Guide",
        description="Guide to finite-pool Bayesian optimization planning with matgpr.",
        mime_type="text/markdown",
        function=bayesian_optimization_guide,
    ),
)
