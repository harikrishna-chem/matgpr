from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import matgpr.bayesian_optimization as bo


class _CountingPosteriorModel:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.posterior_shapes: list[tuple[int, ...]] = []

    def eval(self) -> _CountingPosteriorModel:
        self.eval_calls += 1
        return self

    def posterior(self, candidate_x: torch.Tensor) -> SimpleNamespace:
        self.posterior_shapes.append(tuple(candidate_x.shape))
        mean = candidate_x.sum(dim=-1, keepdim=True)
        variance = torch.full_like(mean, 0.04)
        return SimpleNamespace(mean=mean, variance=variance)


def _candidate_features(n_candidates: int) -> pd.DataFrame:
    x = np.linspace(0.0, 1.0, n_candidates)
    return pd.DataFrame(
        {
            "feature_a": x,
            "feature_b": x**2,
            "feature_c": np.cos(np.pi * x),
        }
    )


@pytest.mark.component
def test_finite_pool_bo_uses_one_batched_model_and_scoring_call_per_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_batch_sizes: list[int] = []
    scoring_batch_sizes: list[int] = []
    models: list[_CountingPosteriorModel] = []
    original_rank_predictions = bo.rank_bo_candidate_predictions

    def fake_fit_surrogate(X_train, y_train, **kwargs):
        del kwargs
        fit_batch_sizes.append(len(X_train))
        model = _CountingPosteriorModel()
        models.append(model)
        train_x = torch.as_tensor(np.asarray(X_train), dtype=torch.float64)
        train_y = torch.as_tensor(np.asarray(y_train), dtype=torch.float64).reshape(-1, 1)
        return bo.BoTorchSurrogate(
            model=model,
            train_X=train_x,
            train_y=train_y,
            objective_y=train_y,
            maximize=True,
            best_observed_objective=float(train_y.max()),
        )

    def count_saved_prediction_scoring(predicted_mean, *args, **kwargs):
        scoring_batch_sizes.append(len(predicted_mean))
        return original_rank_predictions(predicted_mean, *args, **kwargs)

    monkeypatch.setattr(bo, "fit_botorch_surrogate", fake_fit_surrogate)
    monkeypatch.setattr(bo, "rank_bo_candidate_predictions", count_saved_prediction_scoring)

    training_features = _candidate_features(8)
    training_target = pd.Series(np.linspace(0.2, 1.0, 8), name="strength")
    pool_sizes = (16, 1_024)

    ranked_pools = [
        bo.rank_bo_candidate_pool(
            training_features,
            training_target,
            _candidate_features(pool_size),
            objective={"target_name": "strength", "mode": "maximize"},
            normalize_features=False,
            standardize_target=False,
        )
        for pool_size in pool_sizes
    ]

    assert fit_batch_sizes == [len(training_features), len(training_features)]
    assert scoring_batch_sizes == list(pool_sizes)
    assert [model.eval_calls for model in models] == [1, 1]
    assert [model.posterior_shapes for model in models] == [
        [(pool_sizes[0], training_features.shape[1])],
        [(pool_sizes[1], training_features.shape[1])],
    ]
    assert [len(ranked) for ranked in ranked_pools] == list(pool_sizes)
