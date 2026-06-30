from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BOBenchmarkComparison",
    "BOBenchmarkResult",
    "BOBenchmarkStrategy",
    "compare_bo_strategies",
    "simulate_bo_strategy",
    "summarize_bo_benchmark",
]

_HISTORY_COLUMNS = (
    "matgpr_strategy",
    "matgpr_iteration",
    "matgpr_batch_order",
    "matgpr_evaluation",
    "matgpr_candidate_index",
    "matgpr_candidate_id",
    "matgpr_strategy_score",
    "matgpr_target_value",
    "matgpr_best_before_batch",
    "matgpr_best_so_far",
    "matgpr_simple_regret",
    "matgpr_is_optimum",
)


@dataclass(frozen=True)
class BOBenchmarkStrategy:
    """Finite-pool strategy definition for offline BO benchmarking.

    Parameters
    ----------
    name
        Human-readable strategy label.
    score_column
        Optional dataframe column containing precomputed utility scores.
    score_function
        Optional callable that receives `(observed, candidates, rng)` and
        returns one score per remaining candidate. This supports lightweight
        custom policies without requiring BoTorch inside the benchmark loop.
    direction
        Whether larger or smaller strategy scores should be selected first.
        Use `"maximize"` for acquisition functions and `"minimize"` for costs
        or uncertainty/error scores where lower values are preferred.

    Notes
    -----
    When neither `score_column` nor `score_function` is provided, the strategy
    samples candidates uniformly at random.
    """

    name: str
    score_column: str | None = None
    score_function: Callable[[pd.DataFrame, pd.DataFrame, np.random.Generator], Any] | None = None
    direction: str = "maximize"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("BOBenchmarkStrategy.name must be non-empty")
        object.__setattr__(self, "name", name)

        if self.score_column is not None:
            score_column = str(self.score_column).strip()
            if not score_column:
                raise ValueError("score_column must be non-empty when provided")
            object.__setattr__(self, "score_column", score_column)

        if self.score_column is not None and self.score_function is not None:
            raise ValueError("Provide only one of score_column or score_function")

        object.__setattr__(self, "direction", _normalize_direction(self.direction))


@dataclass(frozen=True)
class BOBenchmarkResult:
    """Result from one finite-pool BO replay."""

    strategy_name: str
    target_column: str
    maximize: bool
    history: pd.DataFrame
    summary: pd.DataFrame

    def summary_frame(self) -> pd.DataFrame:
        """Return the one-row benchmark summary."""
        return self.summary.copy()


@dataclass(frozen=True)
class BOBenchmarkComparison:
    """Result from comparing several finite-pool BO strategies."""

    history: pd.DataFrame
    summary: pd.DataFrame
    target_column: str
    maximize: bool

    def summary_by_strategy(self) -> pd.DataFrame:
        """Aggregate benchmark outcomes by strategy."""
        return summarize_bo_benchmark(self.summary)


def simulate_bo_strategy(
    data: pd.DataFrame,
    strategy: BOBenchmarkStrategy,
    *,
    target_column: str,
    maximize: bool = True,
    budget: int | None = None,
    batch_size: int = 1,
    initial_observed: Sequence[Any] | None = None,
    candidate_id_column: str | None = None,
    random_state: int | None = None,
) -> BOBenchmarkResult:
    """Replay one finite-pool strategy against known candidate outcomes.

    This is useful for historical or synthetic benchmarks where every
    candidate has a known target value. The simulator selects candidates from
    the finite pool, reveals their known outcomes, and records the best value
    found as a function of experimental budget.
    """
    frame = _validate_benchmark_data(
        data,
        target_column=target_column,
        candidate_id_column=candidate_id_column,
    )
    if not isinstance(strategy, BOBenchmarkStrategy):
        raise TypeError("strategy must be a BOBenchmarkStrategy")
    batch_size = _validate_positive_int(batch_size, name="batch_size")

    candidate_ids = _candidate_ids(frame, candidate_id_column)
    target_values = _numeric_column(frame, target_column)
    initial_observed_values = None if initial_observed is None else list(initial_observed)
    observed_mask = _initial_observed_mask(
        frame,
        candidate_ids,
        initial_observed=initial_observed_values,
        candidate_id_column=candidate_id_column,
    )
    initial_mask = observed_mask.copy()
    remaining_count = int((~observed_mask).sum())
    budget = remaining_count if budget is None else _validate_positive_int(budget, name="budget")
    budget = min(budget, remaining_count)

    optimum_position = _best_position(target_values, maximize=maximize)
    optimum_value = float(target_values[optimum_position])
    optimum_candidate_id = candidate_ids[optimum_position]

    rng = np.random.default_rng(random_state)
    history_rows: list[dict[str, Any]] = []
    selected_count = 0
    iteration = 0

    while selected_count < budget and np.any(~observed_mask):
        iteration += 1
        batch_limit = min(batch_size, budget - selected_count, int((~observed_mask).sum()))
        selected_positions, selected_scores = _select_strategy_batch(
            frame,
            strategy,
            observed_mask=observed_mask,
            batch_size=batch_limit,
            rng=rng,
        )
        best_before = _best_observed_value(target_values, observed_mask, maximize=maximize)

        for batch_order, (position, score) in enumerate(
            zip(selected_positions, selected_scores),
            start=1,
        ):
            selected_count += 1
            observed_mask[position] = True
            best_so_far = _best_observed_value(target_values, observed_mask, maximize=maximize)
            regret = _simple_regret(
                best_so_far,
                optimum_value=optimum_value,
                maximize=maximize,
            )
            history_rows.append(
                {
                    "matgpr_strategy": strategy.name,
                    "matgpr_iteration": iteration,
                    "matgpr_batch_order": batch_order,
                    "matgpr_evaluation": selected_count,
                    "matgpr_candidate_index": frame.index[position],
                    "matgpr_candidate_id": candidate_ids[position],
                    "matgpr_strategy_score": float(score),
                    "matgpr_target_value": float(target_values[position]),
                    "matgpr_best_before_batch": best_before,
                    "matgpr_best_so_far": best_so_far,
                    "matgpr_simple_regret": regret,
                    "matgpr_is_optimum": bool(position == optimum_position),
                }
            )

    history = pd.DataFrame(history_rows, columns=list(_HISTORY_COLUMNS))
    final_best = _best_observed_value(target_values, observed_mask, maximize=maximize)
    final_regret = _simple_regret(
        final_best,
        optimum_value=optimum_value,
        maximize=maximize,
    )
    evaluations_to_optimum = (
        0 if initial_mask[optimum_position] else _evaluations_to_optimum(history)
    )
    hit_optimum = not pd.isna(evaluations_to_optimum)

    summary = pd.DataFrame(
        [
            {
                "matgpr_strategy": strategy.name,
                "matgpr_target_column": target_column,
                "matgpr_goal": "maximize" if maximize else "minimize",
                "matgpr_n_candidates": frame.shape[0],
                "matgpr_n_initial": int(_initial_count(initial_observed_values)),
                "matgpr_budget": budget,
                "matgpr_batch_size": batch_size,
                "matgpr_n_selected": selected_count,
                "matgpr_n_iterations": iteration,
                "matgpr_optimum_value": optimum_value,
                "matgpr_optimum_candidate_id": optimum_candidate_id,
                "matgpr_final_best": final_best,
                "matgpr_final_regret": final_regret,
                "matgpr_hit_optimum": bool(hit_optimum),
                "matgpr_evaluations_to_optimum": evaluations_to_optimum,
            }
        ]
    )
    return BOBenchmarkResult(
        strategy_name=strategy.name,
        target_column=target_column,
        maximize=maximize,
        history=history,
        summary=summary,
    )


def compare_bo_strategies(
    data: pd.DataFrame,
    strategies: BOBenchmarkStrategy | Sequence[BOBenchmarkStrategy],
    *,
    target_column: str,
    maximize: bool = True,
    budget: int | None = None,
    batch_size: int = 1,
    initial_observed: Sequence[Any] | None = None,
    candidate_id_column: str | None = None,
    n_repeats: int = 1,
    random_state: int | None = None,
) -> BOBenchmarkComparison:
    """Compare several finite-pool BO strategies over repeated replays."""
    strategy_tuple = _as_strategy_tuple(strategies)
    n_repeats = _validate_positive_int(n_repeats, name="n_repeats")
    root_rng = np.random.default_rng(random_state)
    history_tables = []
    summary_tables = []

    for repeat in range(n_repeats):
        for strategy in strategy_tuple:
            seed = int(root_rng.integers(0, np.iinfo(np.uint32).max))
            result = simulate_bo_strategy(
                data,
                strategy,
                target_column=target_column,
                maximize=maximize,
                budget=budget,
                batch_size=batch_size,
                initial_observed=initial_observed,
                candidate_id_column=candidate_id_column,
                random_state=seed,
            )
            history = result.history.copy()
            summary = result.summary.copy()
            history.insert(1, "matgpr_repeat", repeat)
            summary.insert(1, "matgpr_repeat", repeat)
            history_tables.append(history)
            summary_tables.append(summary)

    history = pd.concat(history_tables, ignore_index=True, sort=False)
    summary = pd.concat(summary_tables, ignore_index=True, sort=False)
    return BOBenchmarkComparison(
        history=history,
        summary=summary,
        target_column=target_column,
        maximize=maximize,
    )


def summarize_bo_benchmark(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated BO benchmark summaries by strategy."""
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame")
    required = {
        "matgpr_strategy",
        "matgpr_final_best",
        "matgpr_final_regret",
        "matgpr_hit_optimum",
        "matgpr_evaluations_to_optimum",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"summary is missing benchmark columns: {missing}")

    working = summary.copy()
    working["matgpr_hit_optimum"] = working["matgpr_hit_optimum"].astype(bool)
    working["matgpr_evaluations_to_optimum"] = pd.to_numeric(
        working["matgpr_evaluations_to_optimum"],
        errors="coerce",
    )
    grouped = working.groupby("matgpr_strategy", dropna=False)
    return (
        grouped.agg(
            matgpr_n_runs=("matgpr_strategy", "size"),
            matgpr_final_best_mean=("matgpr_final_best", "mean"),
            matgpr_final_best_std=("matgpr_final_best", "std"),
            matgpr_final_regret_mean=("matgpr_final_regret", "mean"),
            matgpr_final_regret_std=("matgpr_final_regret", "std"),
            matgpr_hit_optimum_rate=("matgpr_hit_optimum", "mean"),
            matgpr_evaluations_to_optimum_mean=(
                "matgpr_evaluations_to_optimum",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            ["matgpr_final_regret_mean", "matgpr_hit_optimum_rate"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def _validate_benchmark_data(
    data: pd.DataFrame,
    *,
    target_column: str,
    candidate_id_column: str | None,
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one candidate")
    target_column = str(target_column).strip()
    if not target_column:
        raise ValueError("target_column must be non-empty")
    if target_column not in data.columns:
        raise ValueError(f"target_column {target_column!r} is missing")
    if candidate_id_column is not None:
        candidate_id_column = str(candidate_id_column).strip()
        if not candidate_id_column:
            raise ValueError("candidate_id_column must be non-empty when provided")
        if candidate_id_column not in data.columns:
            raise ValueError(f"candidate_id_column {candidate_id_column!r} is missing")
        if data[candidate_id_column].isna().any():
            raise ValueError("candidate_id_column contains missing values")
        if data[candidate_id_column].duplicated().any():
            raise ValueError("candidate_id_column must contain unique values")
    _numeric_column(data, target_column)
    return data.reset_index(drop=True).copy()


def _candidate_ids(data: pd.DataFrame, candidate_id_column: str | None) -> np.ndarray:
    if candidate_id_column is None:
        return data.index.to_numpy()
    return data[candidate_id_column].to_numpy(dtype=object)


def _numeric_column(data: pd.DataFrame, column: str) -> np.ndarray:
    if column not in data.columns:
        raise ValueError(f"column {column!r} is missing")
    values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
    if values.ndim != 1:
        raise ValueError(f"column {column!r} must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"column {column!r} contains missing, non-numeric, or infinite values")
    return values


def _initial_observed_mask(
    data: pd.DataFrame,
    candidate_ids: np.ndarray,
    *,
    initial_observed: Sequence[Any] | None,
    candidate_id_column: str | None,
) -> np.ndarray:
    mask = np.zeros(data.shape[0], dtype=bool)
    if initial_observed is None:
        return mask
    initial_values = list(initial_observed)
    if not initial_values:
        return mask
    if len(set(initial_values)) != len(initial_values):
        raise ValueError("initial_observed must not contain duplicate candidates")

    if candidate_id_column is None:
        positions = []
        for value in initial_values:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(
                    "initial_observed must contain integer row positions when "
                    "candidate_id_column is not provided"
                )
            position = int(value)
            if position < 0 or position >= data.shape[0]:
                raise ValueError("initial_observed contains row positions outside the data")
            positions.append(position)
    else:
        id_to_position = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
        missing = [value for value in initial_values if value not in id_to_position]
        if missing:
            raise ValueError(f"initial_observed contains unknown candidate ids: {missing}")
        positions = [id_to_position[value] for value in initial_values]

    mask[np.asarray(positions, dtype=int)] = True
    return mask


def _select_strategy_batch(
    data: pd.DataFrame,
    strategy: BOBenchmarkStrategy,
    *,
    observed_mask: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    remaining_positions = np.flatnonzero(~observed_mask)
    remaining = data.iloc[remaining_positions].copy()
    observed = data.iloc[np.flatnonzero(observed_mask)].copy()

    if strategy.score_column is None and strategy.score_function is None:
        scores = rng.random(remaining.shape[0])
    elif strategy.score_column is not None:
        scores = _numeric_column(remaining, strategy.score_column)
    else:
        raw_scores = strategy.score_function(observed, remaining, rng)
        scores = _coerce_strategy_scores(raw_scores, remaining, strategy=strategy)

    effective_scores = scores if strategy.direction == "maximize" else -scores
    order = np.lexsort((np.arange(effective_scores.shape[0]), -effective_scores))
    chosen_local = order[:batch_size]
    return remaining_positions[chosen_local], scores[chosen_local]


def _coerce_strategy_scores(
    raw_scores: Any,
    remaining: pd.DataFrame,
    *,
    strategy: BOBenchmarkStrategy,
) -> np.ndarray:
    if isinstance(raw_scores, pd.Series):
        if raw_scores.index.equals(remaining.index):
            values = raw_scores.to_numpy(dtype=float)
        else:
            values = np.asarray(raw_scores, dtype=float).reshape(-1)
    else:
        values = np.asarray(raw_scores, dtype=float).reshape(-1)

    if values.shape[0] != remaining.shape[0]:
        raise ValueError(
            f"score_function for strategy {strategy.name!r} must return one "
            f"score per remaining candidate ({values.shape[0]} != {remaining.shape[0]})"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"strategy {strategy.name!r} returned non-finite scores")
    return values


def _best_position(values: np.ndarray, *, maximize: bool) -> int:
    return int(np.argmax(values) if maximize else np.argmin(values))


def _best_observed_value(
    values: np.ndarray,
    observed_mask: np.ndarray,
    *,
    maximize: bool,
) -> float:
    if not np.any(observed_mask):
        return float("nan")
    observed = values[observed_mask]
    return float(np.max(observed) if maximize else np.min(observed))


def _simple_regret(
    best_value: float,
    *,
    optimum_value: float,
    maximize: bool,
) -> float:
    if not np.isfinite(best_value):
        return float("nan")
    regret = optimum_value - best_value if maximize else best_value - optimum_value
    return float(max(regret, 0.0))


def _evaluations_to_optimum(history: pd.DataFrame) -> int | Any:
    if history.empty:
        return pd.NA
    optimum_rows = history.loc[history["matgpr_is_optimum"]]
    if optimum_rows.empty:
        return pd.NA
    return int(optimum_rows["matgpr_evaluation"].iloc[0])


def _initial_count(initial_observed: Sequence[Any] | None) -> int:
    if initial_observed is None:
        return 0
    return len(list(initial_observed))


def _as_strategy_tuple(
    strategies: BOBenchmarkStrategy | Sequence[BOBenchmarkStrategy],
) -> tuple[BOBenchmarkStrategy, ...]:
    if isinstance(strategies, BOBenchmarkStrategy):
        strategy_tuple = (strategies,)
    else:
        try:
            strategy_tuple = tuple(strategies)
        except TypeError as exc:
            raise TypeError(
                "strategies must be a BOBenchmarkStrategy or an iterable of "
                "BOBenchmarkStrategy objects"
            ) from exc
    if not strategy_tuple:
        raise ValueError("strategies must contain at least one strategy")
    invalid = [
        type(strategy).__name__
        for strategy in strategy_tuple
        if not isinstance(strategy, BOBenchmarkStrategy)
    ]
    if invalid:
        raise TypeError(f"strategies must contain only BOBenchmarkStrategy objects; got {invalid}")
    names = [strategy.name for strategy in strategy_tuple]
    if len(set(names)) != len(names):
        raise ValueError("strategy names must be unique")
    return strategy_tuple


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _normalize_direction(direction: str) -> str:
    normalized = str(direction).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "max": "maximize",
        "maximize": "maximize",
        "higher": "maximize",
        "higher_is_better": "maximize",
        "min": "minimize",
        "minimize": "minimize",
        "lower": "minimize",
        "lower_is_better": "minimize",
    }
    if normalized not in aliases:
        raise ValueError("direction must be 'maximize' or 'minimize'")
    return aliases[normalized]
