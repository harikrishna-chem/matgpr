from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .metrics import regression_metrics
    from .uncertainty import calibration_curve, uncertainty_diagnostics
except ImportError:
    from metrics import regression_metrics
    from uncertainty import calibration_curve, uncertainty_diagnostics


def plot_parity(
    y_train_true,
    y_train_pred,
    *,
    y_train_std=None,
    y_test_true=None,
    y_test_pred=None,
    y_test_std=None,
    figsize: tuple[float, float] = (6.5, 6.5),
    title: str = "Parity Plot",
    xlabel: str = "True values",
    ylabel: str = "Predicted values",
    save_path: str | None = None,
    dpi: int = 300,
):
    """Plot true versus predicted values with optional uncertainty bars.

    If test predictions are supplied, the metric annotation is calculated from
    the test set. Otherwise, training metrics are shown.
    """
    y_train_true = _to_1d_array(y_train_true)
    y_train_pred = _to_1d_array(y_train_pred)
    y_train_std = None if y_train_std is None else _to_1d_array(y_train_std)

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        y_train_true,
        y_train_pred,
        yerr=y_train_std,
        fmt="o",
        color="gray",
        ecolor="gray",
        elinewidth=1.0,
        capsize=2,
        alpha=0.65,
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=0.4,
        label="Train",
    )

    all_true = [y_train_true]
    all_pred = [y_train_pred]
    metric_true = y_train_true
    metric_pred = y_train_pred
    metric_label = "Train"

    if y_test_true is not None and y_test_pred is not None:
        y_test_true = _to_1d_array(y_test_true)
        y_test_pred = _to_1d_array(y_test_pred)
        y_test_std = None if y_test_std is None else _to_1d_array(y_test_std)
        ax.errorbar(
            y_test_true,
            y_test_pred,
            yerr=y_test_std,
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            elinewidth=1.2,
            capsize=2,
            alpha=0.9,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Test",
        )
        all_true.append(y_test_true)
        all_pred.append(y_test_pred)
        metric_true = y_test_true
        metric_pred = y_test_pred
        metric_label = "Test"

    metrics = regression_metrics(metric_true, metric_pred)
    ax.text(
        0.05,
        0.95,
        f"{metric_label} metrics\n$R^2$ = {metrics['R2']:.3f}\nPearson r = {metrics['r']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    all_true_values = np.concatenate(all_true)
    all_pred_values = np.concatenate(all_pred)
    min_val = min(all_true_values.min(), all_pred_values.min())
    max_val = max(all_true_values.max(), all_pred_values.max())
    padding = 0.05 * (max_val - min_val) if max_val > min_val else 1.0
    min_val -= padding
    max_val += padding

    ax.plot([min_val, max_val], [min_val, max_val], "--", linewidth=1.5, color="black", label="Ideal")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, ax


def plot_distribution(
    values,
    *,
    bins: int | str = 30,
    density: bool = False,
    color: str = "tab:blue",
    alpha: float = 0.8,
    figsize: tuple[float, float] = (6, 4.5),
    title: str = "Distribution Plot",
    xlabel: str = "Value",
    ylabel: str | None = None,
    show_mean: bool = True,
    show_median: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
):
    """Plot a histogram with optional mean and median markers."""
    values = _to_1d_array(values)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("No valid numeric values provided")

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins, density=density, color=color, alpha=alpha, edgecolor="black", linewidth=0.8)

    if show_mean:
        mean_value = np.mean(values)
        ax.axvline(mean_value, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_value:.3f}")
    if show_median:
        median_value = np.median(values)
        ax.axvline(median_value, color="darkorange", linestyle="-.", linewidth=2, label=f"Median = {median_value:.3f}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or ("Density" if density else "Count"))
    ax.grid(True, alpha=0.25)
    if show_mean or show_median:
        ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, ax


def plot_correlation_matrix(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    method: str = "pearson",
    figsize: tuple[float, float] = (8, 6),
    title: str = "Correlation Matrix",
    cmap: str = "coolwarm",
    annotate: bool = True,
    fmt: str = ".2f",
    save_path: str | None = None,
    dpi: int = 300,
):
    """Plot a numeric feature correlation matrix and return the correlation table."""
    if columns is None:
        data = df.select_dtypes(include=np.number)
    else:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in dataframe: {missing}")
        data = df[columns].select_dtypes(include=np.number)

    if data.shape[1] < 2:
        raise ValueError("At least two numeric columns are required")

    corr = data.corr(method=method)
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(f"{method.title()} correlation")

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    if annotate:
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                value = corr.iloc[i, j]
                ax.text(
                    j,
                    i,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(value) < 0.6 else "white",
                )

    ax.set_title(title)
    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, ax, corr


def plot_learning_curve(
    df: pd.DataFrame,
    *,
    train_size_col: str = "train_size",
    metric_col: str | Sequence[str] | None = "test_R2",
    metric: str | None = None,
    split: str | Sequence[str] = "test",
    x_axis: str = "percent",
    model_col: str = "model",
    models: list[str] | None = None,
    figsize: tuple[float, float] = (7, 5),
    title: str | None = None,
    ylabel: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
):
    """Plot mean metric versus train-set size for one or more models.

    Repeated runs at the same train size are averaged. Error bars show one
    standard deviation across runs. Use ``x_axis="percent"`` to show training
    data in percent, ``x_axis="count"`` to show sample counts, or
    ``x_axis="fraction"`` to show the raw fraction. Use ``metric`` and
    ``split`` as a convenient alternative to a full metric column name. For
    example, ``metric="RMSE", split="test"`` resolves to ``test_RMSE`` and
    ``metric="RMSE", split=("train", "test")`` plots both train and test
    RMSE curves.
    """
    metric_columns = _resolve_learning_curve_metric_columns(
        metric_col=metric_col,
        metric=metric,
        split=split,
    )

    plot_df = df.copy()
    x_col, x_label = _resolve_learning_curve_x_axis(
        plot_df,
        train_size_col=train_size_col,
        x_axis=x_axis,
    )

    required = [x_col, model_col, *metric_columns]
    missing = [column for column in required if column not in plot_df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if models is not None:
        plot_df = plot_df[plot_df[model_col].isin(models)]
    if plot_df.empty:
        raise ValueError("No data available after filtering models")

    metric_frames = []
    for column in metric_columns:
        split_name, metric_name = _metric_column_parts(column)
        metric_frame = plot_df[[model_col, x_col, column]].rename(
            columns={column: "metric_value"}
        )
        metric_frame["metric_col"] = column
        metric_frame["split"] = split_name
        metric_frame["metric"] = metric_name
        metric_frames.append(metric_frame)

    long_df = pd.concat(metric_frames, ignore_index=True)
    summary = (
        long_df.groupby([model_col, "metric_col", "split", "metric", x_col], dropna=False)
        .agg(
            metric_mean=("metric_value", "mean"),
            metric_std=("metric_value", "std"),
            n_runs=("metric_value", "count"),
        )
        .reset_index()
        .sort_values([x_col, model_col, "metric_col"])
    )
    summary["metric_std"] = summary["metric_std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=figsize)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    linestyles = ["-", "--", "-.", ":"]

    curve_keys = (
        summary[[model_col, "metric_col"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for i, (model_name, metric_column) in enumerate(curve_keys):
        model_data = summary[
            (summary[model_col] == model_name)
            & (summary["metric_col"] == metric_column)
        ]
        label = _learning_curve_label(
            model_name,
            metric_column,
            multiple_metrics=len(metric_columns) > 1,
        )
        ax.errorbar(
            model_data[x_col],
            model_data["metric_mean"],
            yerr=model_data["metric_std"],
            fmt=markers[i % len(markers)] + linestyles[i % len(linestyles)],
            elinewidth=1.3,
            capsize=3,
            markersize=7,
            linewidth=2.2,
            label=label,
        )

    y_axis_label = ylabel or _learning_curve_y_label(metric=metric, metric_columns=metric_columns)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(title or f"{y_axis_label} vs Train Set Size")
    ax.set_xticks(sorted(summary[x_col].unique()))
    if all(
        column.lower() in {"test_r2", "train_r2", "r2", "test_r", "train_r", "r"}
        for column in metric_columns
    ):
        ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, ax, summary


def _resolve_learning_curve_metric_columns(
    *,
    metric_col: str | Sequence[str] | None,
    metric: str | None,
    split: str | Sequence[str],
) -> tuple[str, ...]:
    if metric is not None:
        metric_name = _normalize_plot_metric(metric)
        return tuple(
            f"{split_name}_{metric_name}"
            for split_name in _normalize_plot_splits(split)
        )
    if metric_col is None:
        raise ValueError("metric_col is required when metric is not supplied")
    if isinstance(metric_col, str):
        return (metric_col,)
    columns = tuple(dict.fromkeys(str(column) for column in metric_col))
    if not columns:
        raise ValueError("metric_col must contain at least one column")
    return columns


def _resolve_learning_curve_x_axis(
    df: pd.DataFrame,
    *,
    train_size_col: str,
    x_axis: str,
) -> tuple[str, str]:
    normalized = str(x_axis).strip().lower()
    if normalized in {"percent", "percentage", "%"}:
        if "train_size_percent" not in df.columns:
            _require_columns(df, [train_size_col], "learning curve dataframe")
            df["train_size_percent"] = df[train_size_col] * 100
        return "train_size_percent", "Train set size (%)"
    if normalized in {"count", "counts", "n"}:
        if "n_train" in df.columns:
            return "n_train", "Train set size (samples)"
        if (
            train_size_col not in {"train_size", "train_size_percent"}
            and train_size_col in df.columns
        ):
            return train_size_col, "Train set size (samples)"
        raise KeyError(
            "x_axis='count' requires an 'n_train' column or train_size_col "
            "pointing to a sample-count column"
        )
    if normalized in {"fraction", "proportion"}:
        _require_columns(df, [train_size_col], "learning curve dataframe")
        return train_size_col, "Train set size (fraction)"
    raise ValueError("x_axis must be one of: percent, count, fraction")


def _normalize_plot_metric(metric: str) -> str:
    aliases = {
        "rmse": "RMSE",
        "root_mean_squared_error": "RMSE",
        "r2": "R2",
        "r^2": "R2",
        "mae": "MAE",
        "mean_absolute_error": "MAE",
        "r": "r",
        "pearson": "r",
        "pearson_r": "r",
    }
    normalized = str(metric).strip().lower()
    if normalized not in aliases:
        raise ValueError("metric must be one of: RMSE, R2, MAE, r")
    return aliases[normalized]


def _normalize_plot_splits(splits: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(splits, str):
        if splits.lower().replace("_", "-") in {"train-test", "both"}:
            splits = ("train", "test")
        else:
            splits = (splits,)
    normalized = tuple(dict.fromkeys(str(split).strip().lower() for split in splits))
    if not normalized:
        raise ValueError("split must contain at least one split")
    unknown = [split for split in normalized if split not in {"train", "test"}]
    if unknown:
        raise ValueError("split must include only 'train' and 'test'")
    return normalized


def _metric_column_parts(metric_column: str) -> tuple[str, str]:
    split_name, separator, metric_name = str(metric_column).partition("_")
    if separator and split_name in {"train", "test"} and metric_name:
        return split_name, metric_name
    return "", str(metric_column)


def _learning_curve_label(model_name, metric_column: str, *, multiple_metrics: bool) -> str:
    if not multiple_metrics:
        return str(model_name)
    split_name, metric_name = _metric_column_parts(metric_column)
    label_suffix = f"{split_name} {metric_name}".strip()
    return f"{model_name} ({label_suffix})" if label_suffix else f"{model_name} ({metric_column})"


def _learning_curve_y_label(
    *,
    metric: str | None,
    metric_columns: Sequence[str],
) -> str:
    if metric is not None:
        return _normalize_plot_metric(metric)
    if len(metric_columns) == 1:
        return metric_columns[0]
    return "Metric"


def plot_bo_benchmark_trace(
    history: pd.DataFrame,
    *,
    value_col: str = "matgpr_best_so_far",
    strategy_col: str = "matgpr_strategy",
    evaluation_col: str = "matgpr_evaluation",
    strategies: Sequence[str] | None = None,
    show_std: bool = True,
    std_style: str = "band",
    markers: bool = True,
    figsize: tuple[float, float] = (7, 5),
    title: str | None = None,
    xlabel: str = "Evaluated candidates",
    ylabel: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
):
    """Plot repeated finite-pool BO benchmark traces.

    The expected input is the ``history`` table returned by
    ``compare_bo_strategies`` or ``simulate_bo_strategy``. Repeated runs at the
    same strategy and evaluation count are averaged. When ``show_std=True``,
    one standard deviation across repeated replays is drawn as either a shaded
    band or error bars.
    """
    plot_df, strategy_order = _prepare_bo_trace_dataframe(
        history,
        strategy_col=strategy_col,
        evaluation_col=evaluation_col,
        value_col=value_col,
        strategies=strategies,
    )
    summary = _summarize_bo_trace(
        plot_df,
        strategy_col=strategy_col,
        evaluation_col=evaluation_col,
        value_col=value_col,
    )
    std_style = _normalize_std_style(std_style)

    fig, ax = plt.subplots(figsize=figsize)
    marker_styles = ["o", "s", "^", "D", "v", "P", "X"]
    linestyles = ["-", "--", "-.", ":"]

    for i, strategy_name in enumerate(strategy_order):
        strategy_data = summary.loc[summary[strategy_col] == strategy_name]
        if strategy_data.empty:
            continue
        x_values = strategy_data[evaluation_col].to_numpy(dtype=float)
        means = strategy_data["value_mean"].to_numpy(dtype=float)
        stds = strategy_data["value_std"].to_numpy(dtype=float)
        marker = marker_styles[i % len(marker_styles)] if markers else None
        linestyle = linestyles[i % len(linestyles)]

        if show_std and std_style == "errorbar":
            ax.errorbar(
                x_values,
                means,
                yerr=stds,
                fmt=marker or "",
                linestyle=linestyle,
                elinewidth=1.2,
                capsize=3,
                markersize=6,
                linewidth=2.1,
                label=str(strategy_name),
            )
        else:
            (line,) = ax.plot(
                x_values,
                means,
                marker=marker,
                linestyle=linestyle,
                markersize=6,
                linewidth=2.1,
                label=str(strategy_name),
            )
            if show_std and std_style == "band":
                ax.fill_between(
                    x_values,
                    means - stds,
                    means + stds,
                    color=line.get_color(),
                    alpha=0.16,
                    linewidth=0,
                )

    y_axis_label = ylabel or _bo_trace_y_label(value_col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_axis_label)
    ax.set_title(title or f"{y_axis_label} vs BO Evaluations")
    _set_integer_xticks_if_compact(ax, summary[evaluation_col])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, ax, summary


def plot_bo_regret_trace(
    history: pd.DataFrame,
    *,
    value_col: str = "matgpr_simple_regret",
    strategy_col: str = "matgpr_strategy",
    evaluation_col: str = "matgpr_evaluation",
    strategies: Sequence[str] | None = None,
    show_std: bool = True,
    std_style: str = "band",
    markers: bool = True,
    figsize: tuple[float, float] = (7, 5),
    title: str | None = None,
    xlabel: str = "Evaluated candidates",
    ylabel: str = "Simple regret",
    save_path: str | None = None,
    dpi: int = 300,
):
    """Plot simple regret from repeated finite-pool BO benchmarks."""
    return plot_bo_benchmark_trace(
        history,
        value_col=value_col,
        strategy_col=strategy_col,
        evaluation_col=evaluation_col,
        strategies=strategies,
        show_std=show_std,
        std_style=std_style,
        markers=markers,
        figsize=figsize,
        title=title or "Simple Regret vs BO Evaluations",
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
        dpi=dpi,
    )


def plot_bo_campaign_progress(
    log_data: pd.DataFrame | str | Path,
    *,
    campaign_id: str | None = None,
    target_column: str | None = None,
    maximize: bool = True,
    campaign_col: str = "matgpr_campaign_id",
    iteration_col: str = "matgpr_iteration",
    record_type_col: str = "matgpr_record_type",
    observation_record_type: str = "observation",
    figsize: tuple[float, float] = (7, 5),
    title: str | None = None,
    xlabel: str = "BO iteration",
    ylabel: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
):
    """Plot closed-loop BO campaign progress from a log dataframe or CSV path.

    With ``target_column=None``, the plot shows the number of recommendation,
    selection, and observation records logged at each iteration. With
    ``target_column`` supplied, the plot shows the best observed target value
    so far across observation records.
    """
    log = _prepare_bo_campaign_log(
        log_data,
        campaign_id=campaign_id,
        campaign_col=campaign_col,
        iteration_col=iteration_col,
        record_type_col=record_type_col,
    )
    if target_column is None:
        summary = _summarize_bo_campaign_record_counts(
            log,
            iteration_col=iteration_col,
            record_type_col=record_type_col,
        )
        fig, ax = _plot_bo_campaign_record_counts(
            summary,
            iteration_col=iteration_col,
            record_type_col=record_type_col,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
    else:
        summary = _summarize_bo_campaign_target_progress(
            log,
            target_column=target_column,
            maximize=maximize,
            iteration_col=iteration_col,
            record_type_col=record_type_col,
            observation_record_type=observation_record_type,
        )
        fig, ax = _plot_bo_campaign_target_progress(
            summary,
            target_column=target_column,
            iteration_col=iteration_col,
            maximize=maximize,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    _save_figure(fig, save_path, dpi)
    return fig, ax, summary


def _prepare_bo_trace_dataframe(
    history: pd.DataFrame,
    *,
    strategy_col: str,
    evaluation_col: str,
    value_col: str,
    strategies: Sequence[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame")
    _require_columns(
        history,
        [strategy_col, evaluation_col, value_col],
        "BO benchmark history",
    )
    plot_df = history[[strategy_col, evaluation_col, value_col]].copy()
    plot_df[strategy_col] = plot_df[strategy_col].astype(str)
    _coerce_finite_numeric_columns(
        plot_df,
        [evaluation_col, value_col],
        "BO benchmark history",
    )

    if strategies is None:
        strategy_order = plot_df[strategy_col].drop_duplicates().tolist()
    else:
        strategy_order = [str(strategy).strip() for strategy in strategies]
        if not strategy_order or any(not strategy for strategy in strategy_order):
            raise ValueError("strategies must contain non-empty strategy names")
        plot_df = plot_df.loc[plot_df[strategy_col].isin(strategy_order)].copy()

    if plot_df.empty:
        raise ValueError("No BO benchmark history rows available for plotting")
    return plot_df, strategy_order


def _summarize_bo_trace(
    plot_df: pd.DataFrame,
    *,
    strategy_col: str,
    evaluation_col: str,
    value_col: str,
) -> pd.DataFrame:
    summary = (
        plot_df.groupby([strategy_col, evaluation_col], dropna=False, sort=False)
        .agg(
            value_mean=(value_col, "mean"),
            value_std=(value_col, "std"),
            n_runs=(value_col, "count"),
        )
        .reset_index()
        .sort_values([strategy_col, evaluation_col])
        .reset_index(drop=True)
    )
    summary["value_std"] = summary["value_std"].fillna(0.0)
    summary["value_column"] = value_col
    return summary


def _normalize_std_style(std_style: str) -> str:
    normalized = str(std_style).strip().lower().replace("-", "_")
    aliases = {
        "band": "band",
        "shade": "band",
        "shaded": "band",
        "fill": "band",
        "errorbar": "errorbar",
        "errorbars": "errorbar",
        "error_bar": "errorbar",
    }
    if normalized not in aliases:
        raise ValueError("std_style must be 'band' or 'errorbar'")
    return aliases[normalized]


def _bo_trace_y_label(value_col: str) -> str:
    labels = {
        "matgpr_best_so_far": "Best value found",
        "matgpr_simple_regret": "Simple regret",
        "matgpr_target_value": "Observed target value",
        "matgpr_strategy_score": "Strategy score",
    }
    return labels.get(value_col, value_col)


def _prepare_bo_campaign_log(
    log_data: pd.DataFrame | str | Path,
    *,
    campaign_id: str | None,
    campaign_col: str,
    iteration_col: str,
    record_type_col: str,
) -> pd.DataFrame:
    log = _coerce_plot_dataframe(log_data, "log_data")
    _require_columns(log, [iteration_col, record_type_col], "BO campaign log")
    if campaign_id is not None:
        _require_columns(log, [campaign_col], "BO campaign log")
        log = log.loc[log[campaign_col].astype(str) == str(campaign_id)].copy()
        if log.empty:
            raise ValueError(f"No BO campaign log rows found for campaign_id={campaign_id!r}")
    _coerce_finite_numeric_columns(log, [iteration_col], "BO campaign log")
    log[record_type_col] = log[record_type_col].astype(str)
    if log.empty:
        raise ValueError("No BO campaign log rows available for plotting")
    return log


def _summarize_bo_campaign_record_counts(
    log: pd.DataFrame,
    *,
    iteration_col: str,
    record_type_col: str,
) -> pd.DataFrame:
    return (
        log.groupby([iteration_col, record_type_col], dropna=False, sort=False)
        .size()
        .reset_index(name="matgpr_record_count")
        .sort_values([iteration_col, record_type_col])
        .reset_index(drop=True)
    )


def _plot_bo_campaign_record_counts(
    summary: pd.DataFrame,
    *,
    iteration_col: str,
    record_type_col: str,
    figsize: tuple[float, float],
    title: str | None,
    xlabel: str,
    ylabel: str | None,
):
    fig, ax = plt.subplots(figsize=figsize)
    marker_styles = ["o", "s", "^", "D", "v", "P", "X"]
    for i, record_type in enumerate(_ordered_record_types(summary[record_type_col])):
        record_data = summary.loc[summary[record_type_col] == record_type]
        ax.plot(
            record_data[iteration_col].to_numpy(dtype=float),
            record_data["matgpr_record_count"].to_numpy(dtype=float),
            marker=marker_styles[i % len(marker_styles)],
            linewidth=2.0,
            markersize=6,
            label=str(record_type),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or "Logged records")
    ax.set_title(title or "Closed-Loop BO Records by Iteration")
    _set_integer_xticks_if_compact(ax, summary[iteration_col])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig, ax


def _summarize_bo_campaign_target_progress(
    log: pd.DataFrame,
    *,
    target_column: str,
    maximize: bool,
    iteration_col: str,
    record_type_col: str,
    observation_record_type: str,
) -> pd.DataFrame:
    target_column = str(target_column).strip()
    if not target_column:
        raise ValueError("target_column must be non-empty when provided")
    _require_columns(log, [target_column], "BO campaign log")
    observations = log.loc[log[record_type_col] == str(observation_record_type)].copy()
    if observations.empty:
        raise ValueError(
            f"No BO campaign observation rows found for record type {observation_record_type!r}"
        )
    _coerce_finite_numeric_columns(
        observations,
        [iteration_col, target_column],
        "BO campaign observation log",
    )
    best_agg = "max" if maximize else "min"
    summary = (
        observations.groupby(iteration_col, dropna=False, sort=True)
        .agg(
            matgpr_observation_count=(target_column, "count"),
            matgpr_target_mean=(target_column, "mean"),
            matgpr_target_min=(target_column, "min"),
            matgpr_target_max=(target_column, "max"),
            matgpr_iteration_best=(target_column, best_agg),
        )
        .reset_index()
        .sort_values(iteration_col)
        .reset_index(drop=True)
    )
    if maximize:
        summary["matgpr_best_so_far"] = summary["matgpr_iteration_best"].cummax()
    else:
        summary["matgpr_best_so_far"] = summary["matgpr_iteration_best"].cummin()
    return summary


def _plot_bo_campaign_target_progress(
    summary: pd.DataFrame,
    *,
    target_column: str,
    iteration_col: str,
    maximize: bool,
    figsize: tuple[float, float],
    title: str | None,
    xlabel: str,
    ylabel: str | None,
):
    fig, ax = plt.subplots(figsize=figsize)
    iterations = summary[iteration_col].to_numpy(dtype=float)
    iteration_best = summary["matgpr_iteration_best"].to_numpy(dtype=float)
    best_so_far = summary["matgpr_best_so_far"].to_numpy(dtype=float)

    ax.scatter(
        iterations,
        iteration_best,
        s=55,
        alpha=0.8,
        color="tab:gray",
        edgecolor="black",
        linewidth=0.4,
        label="Best this iteration",
    )
    ax.plot(
        iterations,
        best_so_far,
        marker="o",
        linewidth=2.2,
        markersize=6,
        color="tab:blue",
        label="Best so far",
    )
    goal = "maximum" if maximize else "minimum"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or target_column)
    ax.set_title(title or f"Closed-Loop BO Target Progress ({goal})")
    _set_integer_xticks_if_compact(ax, summary[iteration_col])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig, ax


def _coerce_plot_dataframe(data, name: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (str, Path)):
        return pd.read_csv(Path(data))
    raise TypeError(f"{name} must be a pandas DataFrame or CSV path")


def _coerce_finite_numeric_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    invalid_columns = []
    for column in columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().any() or not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
            invalid_columns.append(column)
        df[column] = numeric.astype(float)
    if invalid_columns:
        raise ValueError(f"{name} has missing, non-numeric, or infinite values in {invalid_columns}")


def _ordered_record_types(record_types: pd.Series) -> list[str]:
    values = [str(value) for value in record_types.drop_duplicates()]
    preferred = ["recommendation", "selection", "observation"]
    return [value for value in preferred if value in values] + [
        value for value in values if value not in preferred
    ]


def _set_integer_xticks_if_compact(ax, values: pd.Series) -> None:
    unique_values = np.sort(pd.Series(values).dropna().unique().astype(float))
    if 0 < unique_values.size <= 20 and np.allclose(unique_values, np.round(unique_values)):
        ax.set_xticks(unique_values.astype(int))


def plot_pca_scree(
    pca,
    *,
    figsize: tuple[float, float] = (6, 4),
    title: str = "PCA Scree Plot",
):
    """Plot explained variance and cumulative variance from a fitted PCA."""
    explained = pca.explained_variance_ratio_
    components = np.arange(1, len(explained) + 1)
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(components, explained, alpha=0.7, label="Explained variance")
    ax.plot(components, cumulative, marker="o", label="Cumulative variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance ratio")
    ax.set_title(title)
    ax.set_xticks(components)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_pca_scores(
    train_scores: pd.DataFrame,
    *,
    test_scores: pd.DataFrame | None = None,
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    figsize: tuple[float, float] = (6, 5),
    title: str = "PCA Scores",
):
    """Plot PCA scores for training data and optionally overlay test data."""
    _require_columns(train_scores, [pc_x, pc_y], "train_scores")
    if test_scores is not None:
        _require_columns(test_scores, [pc_x, pc_y], "test_scores")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        train_scores[pc_x],
        train_scores[pc_y],
        s=55,
        alpha=0.75,
        label="Train",
        edgecolor="k",
        linewidth=0.4,
    )

    if test_scores is not None:
        ax.scatter(
            test_scores[pc_x],
            test_scores[pc_y],
            s=75,
            alpha=0.95,
            label="Test",
            marker="^",
            edgecolor="k",
            linewidth=0.5,
        )

    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_uncertainty_calibration(
    y_true,
    y_pred,
    y_std,
    *,
    confidence_levels=None,
    figsize: tuple[float, float] = (6, 5),
    title: str = "Uncertainty Calibration",
    save_path: str | None = None,
    dpi: int = 300,
):
    """Plot observed versus expected Gaussian interval coverage."""
    curve = calibration_curve(
        y_true,
        y_pred,
        y_std,
        confidence_levels=confidence_levels,
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        curve["expected_coverage"],
        curve["observed_coverage"],
        marker="o",
        linewidth=2.0,
        markersize=6,
        color="tab:blue",
        label="Model",
    )
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1.4, label="Ideal")
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, ax, curve


def plot_uncertainty_vs_error(
    y_true,
    y_pred,
    y_std,
    *,
    figsize: tuple[float, float] = (6, 5),
    title: str = "Uncertainty vs Error",
    save_path: str | None = None,
    dpi: int = 300,
):
    """Plot predictive standard deviation versus absolute prediction error."""
    y_true = _to_1d_float_array(y_true, "y_true")
    y_pred = _to_1d_float_array(y_pred, "y_pred")
    y_std = _to_1d_float_array(y_std, "y_std")
    if not (len(y_true) == len(y_pred) == len(y_std)):
        raise ValueError("y_true, y_pred, and y_std must have the same length")
    if np.any(y_std <= 0):
        raise ValueError("y_std must contain only positive values")

    absolute_error = np.abs(y_true - y_pred)
    diagnostics = uncertainty_diagnostics(y_true, y_pred, y_std)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        y_std,
        absolute_error,
        s=55,
        alpha=0.8,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Predictive standard deviation")
    ax.set_ylabel("Absolute prediction error")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.05,
        0.95,
        f"NLPD = {diagnostics['NLPD']:.3f}\n"
        f"Coverage = {diagnostics['observed_coverage']:.3f}\n"
        f"Spearman = {diagnostics['uncertainty_error_spearman']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )
    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, ax, diagnostics


def _to_1d_array(values) -> np.ndarray:
    return np.asarray(values).ravel()


def _to_1d_float_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _save_figure(fig, save_path: str | None, dpi: int) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")
