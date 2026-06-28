from __future__ import annotations

from functools import reduce
from operator import add, mul

import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, WhiteKernel

__all__ = [
    "FeatureSubsetKernel",
    "TanimotoKernel",
    "build_additive_kernel",
    "build_product_kernel",
    "build_tanimoto_gpr_kernel",
    "pairwise_tanimoto_similarity",
]


class TanimotoKernel(Kernel):
    """Tanimoto kernel for binary or non-negative count fingerprints.

    For vectors ``x`` and ``y``, the generalized Tanimoto similarity is:

    ``k(x, y) = dot(x, y) / (||x||^2 + ||y||^2 - dot(x, y))``.

    This is commonly used for molecular fingerprints and polymer fingerprints.
    Inputs must be binary or non-negative count descriptors.
    """

    def __init__(self, *, eps: float = 1e-12, validate_nonnegative: bool = True):
        self.eps = eps
        self.validate_nonnegative = validate_nonnegative

    def __call__(self, X, Y=None, eval_gradient: bool = False):
        """Return the Tanimoto kernel matrix."""
        if eval_gradient and Y is not None:
            raise ValueError("Gradient can only be evaluated when Y is None")

        kernel = pairwise_tanimoto_similarity(
            X,
            Y,
            eps=self.eps,
            validate_nonnegative=self.validate_nonnegative,
        )
        if eval_gradient:
            return kernel, np.empty((*kernel.shape, 0))
        return kernel

    def diag(self, X) -> np.ndarray:
        """Return the diagonal of the kernel matrix."""
        X = _to_2d_float_array(X, "X")
        if self.validate_nonnegative:
            _validate_nonnegative(X, "X")
        return np.ones(X.shape[0])

    def is_stationary(self) -> bool:
        """Return whether the kernel is stationary."""
        return False

    def __repr__(self) -> str:
        return f"TanimotoKernel(eps={self.eps}, validate_nonnegative={self.validate_nonnegative})"


class FeatureSubsetKernel(Kernel):
    """Apply a scikit-learn kernel to selected feature columns.

    This wrapper is useful for mixed materials descriptors. For example, a
    Tanimoto kernel can act on fingerprint columns while an RBF or Matern kernel
    acts on continuous physics descriptors, then the kernels can be added or
    multiplied.
    """

    def __init__(self, kernel: Kernel, columns):
        self.kernel = kernel
        self.columns = columns

    @property
    def hyperparameters(self):
        return self.kernel.hyperparameters

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def clone_with_theta(self, theta):
        """Return a clone with updated wrapped-kernel hyperparameters."""
        cloned = clone(self)
        cloned.kernel = self.kernel.clone_with_theta(theta)
        return cloned

    def __call__(self, X, Y=None, eval_gradient: bool = False):
        """Return the wrapped-kernel matrix on selected columns."""
        X_subset = _select_columns(X, self.columns, "X")
        Y_subset = None if Y is None else _select_columns(Y, self.columns, "Y")
        return self.kernel(X_subset, Y_subset, eval_gradient=eval_gradient)

    def diag(self, X) -> np.ndarray:
        """Return the wrapped-kernel diagonal on selected columns."""
        return self.kernel.diag(_select_columns(X, self.columns, "X"))

    def is_stationary(self) -> bool:
        """Return whether the wrapped kernel is stationary."""
        return self.kernel.is_stationary()

    def __repr__(self) -> str:
        return f"FeatureSubsetKernel(kernel={self.kernel!r}, columns={self.columns!r})"


def pairwise_tanimoto_similarity(
    X,
    Y=None,
    *,
    eps: float = 1e-12,
    validate_nonnegative: bool = True,
) -> np.ndarray:
    """Return pairwise generalized Tanimoto similarities.

    Parameters
    ----------
    X, Y
        Two-dimensional arrays of binary or non-negative count fingerprints. If
        ``Y`` is omitted, similarities are calculated between rows of ``X``.
    eps
        Small denominator threshold used to identify all-zero fingerprints.
    validate_nonnegative
        If ``True``, reject negative feature values.
    """
    X = _to_2d_float_array(X, "X")
    Y = X if Y is None else _to_2d_float_array(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if validate_nonnegative:
        _validate_nonnegative(X, "X")
        _validate_nonnegative(Y, "Y")

    dot = X @ Y.T
    x_norm = np.sum(X * X, axis=1)[:, None]
    y_norm = np.sum(Y * Y, axis=1)[None, :]
    denominator = x_norm + y_norm - dot

    similarity = np.divide(
        dot,
        denominator,
        out=np.zeros_like(dot, dtype=float),
        where=denominator > eps,
    )
    both_zero = denominator <= eps
    similarity[both_zero] = 1.0
    return similarity


def build_tanimoto_gpr_kernel(
    *,
    constant_value: float = 1.0,
    noise_level: float = 1.0,
    validate_nonnegative: bool = True,
):
    """Build a scikit-learn GPR kernel using Tanimoto fingerprint similarity."""
    return (
        ConstantKernel(constant_value)
        * TanimotoKernel(validate_nonnegative=validate_nonnegative)
        + WhiteKernel(noise_level=noise_level)
    )


def build_additive_kernel(*kernels: Kernel) -> Kernel:
    """Add two or more scikit-learn kernels."""
    _validate_kernel_sequence(kernels)
    return reduce(add, kernels)


def build_product_kernel(*kernels: Kernel) -> Kernel:
    """Multiply two or more scikit-learn kernels."""
    _validate_kernel_sequence(kernels)
    return reduce(mul, kernels)


def _validate_kernel_sequence(kernels: tuple[Kernel, ...]) -> None:
    if len(kernels) < 2:
        raise ValueError("Provide at least two kernels")
    if not all(isinstance(kernel, Kernel) for kernel in kernels):
        raise TypeError("All arguments must be scikit-learn Kernel instances")


def _select_columns(values, columns, name: str) -> np.ndarray:
    array = _to_2d_float_array(values, name)
    column_indices = np.asarray(columns)
    if column_indices.ndim != 1 or column_indices.size == 0:
        raise ValueError("columns must contain at least one column index")
    if not np.issubdtype(column_indices.dtype, np.integer):
        raise TypeError("columns must be integer column indices")
    if np.any(column_indices < 0) or np.any(column_indices >= array.shape[1]):
        raise ValueError(f"columns references indices outside {name}")
    return array[:, column_indices]


def _to_2d_float_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one feature")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_nonnegative(values: np.ndarray, name: str) -> None:
    if np.any(values < 0):
        raise ValueError(f"{name} must contain only non-negative values")
