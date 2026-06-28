from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.model_selection import GridSearchCV

from .kernels import (
    build_element_fraction_gpr_kernel,
    build_structure_gpr_kernel,
    build_tanimoto_gpr_kernel,
)


def build_sklearn_gpr_kernel(
    name: str = "matern",
    *,
    n_features: int | None = None,
    noise_level: float = 1.0,
):
    """Create a common scikit-learn Gaussian-process kernel.

    ``"ard_rbf"`` and ``"ard_matern"`` use one length scale per feature, so
    ``n_features`` is required for those options.
    """
    normalized = name.lower()

    if normalized == "rbf":
        return ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level)
    if normalized == "matern":
        return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level)
    if normalized == "ard_rbf":
        if n_features is None:
            raise ValueError("n_features is required for ard_rbf")
        return ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features)) + WhiteKernel(noise_level)
    if normalized == "ard_matern":
        if n_features is None:
            raise ValueError("n_features is required for ard_matern")
        return ConstantKernel(1.0) * Matern(length_scale=np.ones(n_features), nu=2.5) + WhiteKernel(noise_level)
    if normalized == "tanimoto":
        return build_tanimoto_gpr_kernel(noise_level=noise_level)
    if normalized in {"element_fraction", "composition"}:
        return build_element_fraction_gpr_kernel(noise_level=noise_level)
    if normalized in {"structure", "structure_features"}:
        return build_structure_gpr_kernel(noise_level=noise_level)

    raise ValueError(
        "name must be one of: rbf, matern, ard_rbf, ard_matern, tanimoto, "
        "element_fraction, composition, structure, structure_features"
    )


def build_sklearn_gpr_model(
    *,
    kernel: str = "matern",
    n_features: int | None = None,
    alpha: float | Sequence[float] | np.ndarray = 1e-8,
    normalize_y: bool = True,
    n_restarts_optimizer: int = 5,
    random_state: int = 42,
) -> GaussianProcessRegressor:
    """Build a scikit-learn Gaussian Process Regressor.

    This returns an unfitted model. Put it inside a scikit-learn Pipeline with
    ``build_preprocessor`` when raw dataframe columns need preprocessing.
    ``alpha`` may be a scalar or one variance per training row, which is useful
    when appending soft virtual physics observations.
    """
    kernel_object = build_sklearn_gpr_kernel(kernel, n_features=n_features)
    return GaussianProcessRegressor(
        kernel=kernel_object,
        alpha=alpha,
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )


def build_sklearn_gpr_grid_search(
    *,
    n_features: int,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Build a grid search over useful scikit-learn GPR kernels and settings."""
    kernels = [
        build_sklearn_gpr_kernel("rbf"),
        build_sklearn_gpr_kernel("matern"),
        ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0),
        build_sklearn_gpr_kernel("ard_rbf", n_features=n_features),
        build_sklearn_gpr_kernel("ard_matern", n_features=n_features),
    ]

    model = GaussianProcessRegressor(
        normalize_y=True,
        random_state=random_state,
    )

    param_grid = {
        "kernel": kernels,
        "alpha": [1e-10, 1e-8, 1e-6, 1e-4],
        "n_restarts_optimizer": [2, 5, 10],
    }

    return GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )
