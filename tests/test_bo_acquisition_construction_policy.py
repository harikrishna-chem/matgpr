from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from matgpr import bayesian_optimization as bo

pytestmark = pytest.mark.unit

_SINGLE_ALIASES = {
    "logei": "log_expected_improvement",
    "log_ei": "log_expected_improvement",
    "log_expected_improvement": "log_expected_improvement",
    "lognei": "log_noisy_expected_improvement",
    "log_noisy_ei": "log_noisy_expected_improvement",
    "log_nei": "log_noisy_expected_improvement",
    "log_noisy_expected_improvement": "log_noisy_expected_improvement",
    "q_log_nei": "log_noisy_expected_improvement",
    "q_log_noisy_expected_improvement": "log_noisy_expected_improvement",
    "qlognoisyexpectedimprovement": "log_noisy_expected_improvement",
    "qlognei": "log_noisy_expected_improvement",
    "qlog_noisy_expected_improvement": "log_noisy_expected_improvement",
    "ei": "expected_improvement",
    "expected_improvement": "expected_improvement",
    "nei": "noisy_expected_improvement",
    "noisy_ei": "noisy_expected_improvement",
    "noisy_expected_improvement": "noisy_expected_improvement",
    "q_nei": "noisy_expected_improvement",
    "qnoisyexpectedimprovement": "noisy_expected_improvement",
    "qnei": "noisy_expected_improvement",
    "q_noisy_expected_improvement": "noisy_expected_improvement",
    "pi": "probability_of_improvement",
    "probability_improvement": "probability_of_improvement",
    "probability_of_improvement": "probability_of_improvement",
    "ucb": "upper_confidence_bound",
    "upper_confidence_bound": "upper_confidence_bound",
}

_MULTI_ALIASES = {
    "ehvi": "q_expected_hypervolume_improvement",
    "qehvi": "q_expected_hypervolume_improvement",
    "q_ehvi": "q_expected_hypervolume_improvement",
    "expected_hypervolume_improvement": "q_expected_hypervolume_improvement",
    "q_expected_hypervolume_improvement": "q_expected_hypervolume_improvement",
    "nehvi": "q_noisy_expected_hypervolume_improvement",
    "qnehvi": "q_noisy_expected_hypervolume_improvement",
    "q_nehvi": "q_noisy_expected_hypervolume_improvement",
    "noisy_expected_hypervolume_improvement": "q_noisy_expected_hypervolume_improvement",
    "q_noisy_expected_hypervolume_improvement": "q_noisy_expected_hypervolume_improvement",
    "log_ehvi": "q_log_expected_hypervolume_improvement",
    "logehvi": "q_log_expected_hypervolume_improvement",
    "qlogehvi": "q_log_expected_hypervolume_improvement",
    "q_log_ehvi": "q_log_expected_hypervolume_improvement",
    "log_expected_hypervolume_improvement": "q_log_expected_hypervolume_improvement",
    "q_log_expected_hypervolume_improvement": "q_log_expected_hypervolume_improvement",
    "log_nehvi": "q_log_noisy_expected_hypervolume_improvement",
    "lognehvi": "q_log_noisy_expected_hypervolume_improvement",
    "qlognehvi": "q_log_noisy_expected_hypervolume_improvement",
    "q_log_nehvi": "q_log_noisy_expected_hypervolume_improvement",
    "log_noisy_expected_hypervolume_improvement": ("q_log_noisy_expected_hypervolume_improvement"),
    "q_log_noisy_expected_hypervolume_improvement": (
        "q_log_noisy_expected_hypervolume_improvement"
    ),
}


class _RecordedConstruction:
    def __init__(self, construction_name: str, **kwargs: Any) -> None:
        self.construction_name = construction_name
        self.kwargs = kwargs


def _recording_constructor(name: str) -> type[_RecordedConstruction]:
    class _Constructor(_RecordedConstruction):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(name, **kwargs)

    _Constructor.__name__ = name
    return _Constructor


@pytest.fixture
def fake_botorch_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "botorch.acquisition.analytic": (
            "LogExpectedImprovement",
            "ExpectedImprovement",
            "ProbabilityOfImprovement",
            "UpperConfidenceBound",
        ),
        "botorch.acquisition.logei": ("qLogNoisyExpectedImprovement",),
        "botorch.acquisition.monte_carlo": ("qNoisyExpectedImprovement",),
        "botorch.sampling.normal": ("SobolQMCNormalSampler",),
        "botorch.acquisition.multi_objective.logei": (
            "qLogExpectedHypervolumeImprovement",
            "qLogNoisyExpectedHypervolumeImprovement",
        ),
        "botorch.acquisition.multi_objective.monte_carlo": (
            "qExpectedHypervolumeImprovement",
            "qNoisyExpectedHypervolumeImprovement",
        ),
        "botorch.utils.multi_objective.box_decompositions.non_dominated": (
            "NondominatedPartitioning",
        ),
    }
    for module_name, constructor_names in modules.items():
        module = ModuleType(module_name)
        for constructor_name in constructor_names:
            setattr(module, constructor_name, _recording_constructor(constructor_name))
        monkeypatch.setitem(sys.modules, module_name, module)


@pytest.mark.parametrize(("value", "expected"), tuple(_SINGLE_ALIASES.items()))
def test_single_objective_acquisition_aliases(value: str, expected: str) -> None:
    assert bo._normalize_acquisition_name(value) == expected


def test_single_objective_acquisition_normalization_is_case_and_separator_insensitive() -> None:
    assert bo._normalize_acquisition_name(" Log EI ") == "log_expected_improvement"
    assert bo._normalize_acquisition_name("upper-confidence-bound") == "upper_confidence_bound"


def test_single_objective_acquisition_normalization_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported acquisition_function: 'entropy'"):
        bo._normalize_acquisition_name("entropy")


@pytest.mark.parametrize(("value", "expected"), tuple(_MULTI_ALIASES.items()))
def test_multi_objective_acquisition_aliases(value: str, expected: str) -> None:
    assert bo._normalize_multi_objective_acquisition_name(value) == expected


def test_multi_objective_normalization_is_case_and_separator_insensitive() -> None:
    assert (
        bo._normalize_multi_objective_acquisition_name(" Q Log EHVI ")
        == "q_log_expected_hypervolume_improvement"
    )
    assert (
        bo._normalize_multi_objective_acquisition_name("q-log-nehvi")
        == "q_log_noisy_expected_hypervolume_improvement"
    )


def test_multi_objective_acquisition_normalization_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported multi-objective acquisition_function"):
        bo._normalize_multi_objective_acquisition_name("parego")


@pytest.mark.usefixtures("fake_botorch_modules")
@pytest.mark.parametrize(
    ("name", "constructor", "expected_kwargs"),
    [
        (
            "log_ei",
            "LogExpectedImprovement",
            {"model": "model", "best_f": 3.5},
        ),
        (
            "qlognei",
            "qLogNoisyExpectedImprovement",
            {"model": "model", "X_baseline": "train-x"},
        ),
        (
            "ei",
            "ExpectedImprovement",
            {"model": "model", "best_f": 3.5},
        ),
        (
            "q_nei",
            "qNoisyExpectedImprovement",
            {"model": "model", "X_baseline": "train-x"},
        ),
        (
            "pi",
            "ProbabilityOfImprovement",
            {"model": "model", "best_f": 3.5},
        ),
        (
            "ucb",
            "UpperConfidenceBound",
            {"model": "model", "beta": 0.25},
        ),
    ],
)
def test_single_objective_factory_wires_expected_arguments(
    name: str,
    constructor: str,
    expected_kwargs: dict[str, Any],
) -> None:
    surrogate = SimpleNamespace(
        model="model",
        train_X="train-x",
        best_observed_objective=3.5,
    )
    acquisition = bo._build_acquisition_function(name, surrogate=surrogate, beta=0.25)
    assert acquisition.construction_name == constructor
    assert acquisition.kwargs == expected_kwargs


def test_single_objective_factory_defensively_rejects_unknown_normalized_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        bo._build_acquisition_function.__globals__,
        "_normalize_acquisition_name",
        lambda name: "unsupported",
    )
    with pytest.raises(ValueError, match="acquisition_function must be one of"):
        bo._build_acquisition_function(
            "anything",
            surrogate=SimpleNamespace(model="model"),
            beta=0.1,
        )


@pytest.mark.usefixtures("fake_botorch_modules")
@pytest.mark.parametrize(
    ("name", "constructor", "uses_partitioning", "uses_baseline"),
    [
        ("q_log_ehvi", "qLogExpectedHypervolumeImprovement", True, False),
        ("q_log_nehvi", "qLogNoisyExpectedHypervolumeImprovement", False, True),
        ("q_ehvi", "qExpectedHypervolumeImprovement", True, False),
        ("q_nehvi", "qNoisyExpectedHypervolumeImprovement", False, True),
    ],
)
def test_multi_objective_factory_wires_sampler_partitioning_baseline_and_pending(
    name: str,
    constructor: str,
    uses_partitioning: bool,
    uses_baseline: bool,
) -> None:
    import torch

    surrogate = SimpleNamespace(
        model="model",
        train_X="train-x",
        reference_point="reference-point",
        objective_y="objective-y",
    )
    acquisition = bo._build_multi_objective_acquisition_function(
        name,
        surrogate=surrogate,
        mc_samples=32,
        sampler_seed=17,
        x_pending="pending-x",
    )

    assert acquisition.construction_name == constructor
    assert acquisition.kwargs["model"] == "model"
    assert acquisition.kwargs["ref_point"] == "reference-point"
    assert acquisition.kwargs["X_pending"] == "pending-x"
    sampler = acquisition.kwargs["sampler"]
    assert sampler.construction_name == "SobolQMCNormalSampler"
    assert sampler.kwargs == {"sample_shape": torch.Size([32]), "seed": 17}

    if uses_partitioning:
        partitioning = acquisition.kwargs["partitioning"]
        assert partitioning.construction_name == "NondominatedPartitioning"
        assert partitioning.kwargs == {
            "ref_point": "reference-point",
            "Y": "objective-y",
        }
        assert "X_baseline" not in acquisition.kwargs
    else:
        assert "partitioning" not in acquisition.kwargs
    if uses_baseline:
        assert acquisition.kwargs["X_baseline"] == "train-x"
    else:
        assert "X_baseline" not in acquisition.kwargs


def test_multi_objective_factory_defensively_rejects_unknown_normalized_name(
    monkeypatch: pytest.MonkeyPatch,
    fake_botorch_modules: None,
) -> None:
    monkeypatch.setitem(
        bo._build_multi_objective_acquisition_function.__globals__,
        "_normalize_multi_objective_acquisition_name",
        lambda name: "unsupported",
    )
    with pytest.raises(ValueError, match="acquisition_function must be one of"):
        bo._build_multi_objective_acquisition_function(
            "anything",
            surrogate=SimpleNamespace(model="model"),
            mc_samples=8,
            sampler_seed=None,
        )


def test_parent_acquisition_construction_helpers_are_exact_internal_aliases() -> None:
    policy = importlib.import_module("matgpr._bo_acquisition")
    names = (
        "_build_acquisition_function",
        "_build_multi_objective_acquisition_function",
        "_normalize_acquisition_name",
        "_normalize_multi_objective_acquisition_name",
    )
    for name in names:
        assert getattr(bo, name) is getattr(policy, name)
