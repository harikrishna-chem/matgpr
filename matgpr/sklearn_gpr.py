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

__all__ = [
    "build_sklearn_gpr_grid_search",
    "build_sklearn_gpr_kernel",
    "build_sklearn_gpr_model",
]


def build_sklearn_gpr_kernel(
    name: str = "matern",
    *,
    n_features: int | None = None,
    constant_value: float = 1.0,
    constant_value_bounds: tuple[float, float] | str = (1e-3, 1e3),
    length_scale: float | np.ndarray = 1.0,
    length_scale_bounds: tuple[float, float] | str = (1e-5, 1e5),
    noise_level: float = 1e-6,
    noise_level_bounds: tuple[float, float] | str = (1e-10, 1e-1),
    nu: float = 2.5,
    metric: str | None = None,
    normalize: bool = True,
    feature_scales=None,
    validate_nonnegative: bool = True,
):
    """Create a common scikit-learn Gaussian-process kernel.

    ``"ard_rbf"`` and ``"ard_matern"`` use one length scale per feature, so
    ``n_features`` is required for those options.
    """
    normalized = _normalize_kernel_name(name)

    if normalized == "rbf":
        return _build_constant_scaled_kernel(
            RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds),
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
        )
    if normalized in {"matern", "matern25", "matern52"}:
        return _build_constant_scaled_kernel(
            Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu),
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
        )
    if normalized == "ardrbf":
        n_features = _validate_n_features(n_features, name="ard_rbf")
        return _build_constant_scaled_kernel(
            RBF(
                length_scale=np.ones(n_features),
                length_scale_bounds=length_scale_bounds,
            ),
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
        )
    if normalized == "ardmatern":
        n_features = _validate_n_features(n_features, name="ard_matern")
        return _build_constant_scaled_kernel(
            Matern(
                length_scale=np.ones(n_features),
                length_scale_bounds=length_scale_bounds,
                nu=nu,
            ),
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
        )
    if normalized == "tanimoto":
        return build_tanimoto_gpr_kernel(
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
            validate_nonnegative=validate_nonnegative,
        )
    if normalized in {"elementfraction", "composition"}:
        return build_element_fraction_gpr_kernel(
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds,
            length_scale=float(length_scale),
            length_scale_bounds=length_scale_bounds,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
            metric="l1" if metric is None else metric,
            normalize=normalize,
        )
    if normalized in {"structure", "structurefeatures"}:
        return build_structure_gpr_kernel(
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds,
            length_scale=float(length_scale),
            length_scale_bounds=length_scale_bounds,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
            metric="l2" if metric is None else metric,
            feature_scales=feature_scales,
        )

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
    random_state: int | None = 42,
    **kernel_kwargs,
) -> GaussianProcessRegressor:
    """Build a scikit-learn Gaussian Process Regressor.

    This returns an unfitted model. Put it inside a scikit-learn Pipeline with
    ``build_preprocessor`` when raw dataframe columns need preprocessing.
    ``alpha`` may be a scalar or one variance per training row, which is useful
    when appending soft virtual physics observations.
    """
    kernel_object = build_sklearn_gpr_kernel(
        kernel,
        n_features=n_features,
        **kernel_kwargs,
    )
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
    random_state: int | None = 42,
    n_jobs: int = 1,
) -> GridSearchCV:
    """Build a grid search over useful scikit-learn GPR kernels and settings."""
    _validate_n_features(n_features, name="grid search")
    kernels = [
        build_sklearn_gpr_kernel("rbf"),
        build_sklearn_gpr_kernel("matern"),
        build_sklearn_gpr_kernel("matern", nu=1.5),
        build_sklearn_gpr_kernel("ard_rbf", n_features=n_features),
        build_sklearn_gpr_kernel("ard_matern", n_features=n_features),
    ]

    model = GaussianProcessRegressor(
        normalize_y=True,
        random_state=random_state,
    )

    param_grid = {
        "kernel": kernels,
        "alpha": [1e-10, 1e-8, 1e-6],
        "n_restarts_optimizer": [0, 2],
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


def _build_constant_scaled_kernel(
    base_kernel,
    *,
    constant_value: float,
    constant_value_bounds: tuple[float, float] | str,
    noise_level: float,
    noise_level_bounds: tuple[float, float] | str,
):
    return ConstantKernel(
        constant_value, constant_value_bounds=constant_value_bounds
    ) * base_kernel + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)


def _normalize_kernel_name(name: str) -> str:
    return str(name).lower().replace("-", "").replace("_", "").replace(" ", "")


def _validate_n_features(n_features: int | None, *, name: str) -> int:
    if n_features is None:
        raise ValueError(f"n_features is required for {name}")
    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    return n_features
