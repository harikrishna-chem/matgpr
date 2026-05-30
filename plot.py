from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .analysis import calculate_regression_metrics
except ImportError:
    from analysis import calculate_regression_metrics


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

    metrics = calculate_regression_metrics(metric_true, metric_pred)
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
    metric_col: str = "test_R2",
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
    standard deviation across runs.
    """
    required = [train_size_col, metric_col, model_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    plot_df = df.copy()
    if models is not None:
        plot_df = plot_df[plot_df[model_col].isin(models)]
    if plot_df.empty:
        raise ValueError("No data available after filtering models")

    plot_df["train_size_percent"] = plot_df[train_size_col] * 100
    summary = (
        plot_df.groupby([model_col, "train_size_percent"])
        .agg(metric_mean=(metric_col, "mean"), metric_std=(metric_col, "std"), n_runs=(metric_col, "count"))
        .reset_index()
        .sort_values(["train_size_percent", model_col])
    )
    summary["metric_std"] = summary["metric_std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=figsize)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    linestyles = ["-", "--", "-.", ":"]

    for i, model_name in enumerate(summary[model_col].unique()):
        model_data = summary[summary[model_col] == model_name]
        ax.errorbar(
            model_data["train_size_percent"],
            model_data["metric_mean"],
            yerr=model_data["metric_std"],
            fmt=markers[i % len(markers)] + linestyles[i % len(linestyles)],
            elinewidth=1.3,
            capsize=3,
            markersize=7,
            linewidth=2.2,
            label=str(model_name),
        )

    ax.set_xlabel("Train set size (%)")
    ax.set_ylabel(ylabel or metric_col)
    ax.set_title(title or f"{metric_col} vs Train Set Size")
    ax.set_xticks(sorted(summary["train_size_percent"].unique()))
    if metric_col.lower() in {"test_r2", "train_r2", "r2", "test_r", "train_r", "r"}:
        ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, ax, summary


def _to_1d_array(values) -> np.ndarray:
    return np.asarray(values).ravel()


def _save_figure(fig, save_path: str | None, dpi: int) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
