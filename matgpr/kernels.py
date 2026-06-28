from __future__ import annotations

from functools import reduce
from operator import add, mul

import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process.kernels import ConstantKernel, Hyperparameter, Kernel, WhiteKernel

__all__ = [
    "ElementFractionKernel",
    "FeatureSubsetKernel",
    "StructureFeatureKernel",
    "TanimotoKernel",
    "build_additive_kernel",
    "build_element_fraction_gpr_kernel",
    "build_product_kernel",
    "build_structure_gpr_kernel",
    "build_tanimoto_gpr_kernel",
    "pairwise_composition_distance",
    "pairwise_structure_distance",
    "pairwise_tanimoto_similarity",
]


class ElementFractionKernel(Kernel):
    """Kernel for elemental composition fraction or count vectors.

    The kernel first projects non-negative element vectors onto the composition
    simplex by normalizing each row to sum to one. It then compares normalized
    compositions with either an L1 Laplacian kernel or an L2 RBF kernel.

    The default L1 form is:

    ``k(x, y) = exp(-sum_i |c_i - c'_i| / length_scale)``

    where ``c`` and ``c'`` are element-fraction vectors. This is useful when
    the physical idea of similarity is the amount of composition substitution
    needed to move from one material to another.
    """

    def __init__(
        self,
        *,
        length_scale: float = 1.0,
        length_scale_bounds: tuple[float, float] | str = (1e-5, 1e5),
        metric: str = "l1",
        normalize: bool = True,
        eps: float = 1e-12,
    ):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.metric = metric
        self.normalize = normalize
        self.eps = eps

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient: bool = False):
        """Return the element-fraction composition kernel matrix."""
        if eval_gradient and Y is not None:
            raise ValueError("Gradient can only be evaluated when Y is None")

        length_scale = _validate_length_scale(self.length_scale)
        distance = pairwise_composition_distance(
            X,
            Y,
            metric=self.metric,
            normalize=self.normalize,
            eps=self.eps,
        )

        if self.metric == "l1":
            kernel = np.exp(-distance / length_scale)
            log_gradient = kernel * distance / length_scale
        elif self.metric == "l2":
            kernel = np.exp(-0.5 * distance / length_scale**2)
            log_gradient = kernel * distance / length_scale**2
        else:
            raise ValueError("metric must be either 'l1' or 'l2'")

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                return kernel, np.empty((*kernel.shape, 0))
            return kernel, log_gradient[:, :, None]
        return kernel

    def diag(self, X) -> np.ndarray:
        """Return the diagonal of the kernel matrix."""
        X = _normalize_composition_vectors(X, normalize=self.normalize, eps=self.eps, name="X")
        return np.ones(X.shape[0])

    def is_stationary(self) -> bool:
        """Return whether the kernel is stationary."""
        return False

    def __repr__(self) -> str:
        return (
            "ElementFractionKernel("
            f"length_scale={self.length_scale}, "
            f"metric={self.metric!r}, "
            f"normalize={self.normalize})"
        )


class StructureFeatureKernel(Kernel):
    """Kernel for continuous structure descriptors.

    This kernel is intended for global crystal-structure descriptors such as
    lattice-shape, volume-per-atom, and density features from
    ``matgpr.structure_fingerprint``. It compares descriptor vectors after
    optional fixed feature scaling using either an L2 RBF kernel or an L1
    Laplacian kernel.
    """

    def __init__(
        self,
        *,
        length_scale: float = 1.0,
        length_scale_bounds: tuple[float, float] | str = (1e-5, 1e5),
        metric: str = "l2",
        feature_scales=None,
    ):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.metric = metric
        self.feature_scales = feature_scales

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient: bool = False):
        """Return the structure-feature kernel matrix."""
        if eval_gradient and Y is not None:
            raise ValueError("Gradient can only be evaluated when Y is None")

        length_scale = _validate_length_scale(self.length_scale)
        distance = pairwise_structure_distance(
            X,
            Y,
            metric=self.metric,
            feature_scales=self.feature_scales,
        )

        if self.metric == "l2":
            kernel = np.exp(-0.5 * distance / length_scale**2)
            log_gradient = kernel * distance / length_scale**2
        elif self.metric == "l1":
            kernel = np.exp(-distance / length_scale)
            log_gradient = kernel * distance / length_scale
        else:
            raise ValueError("metric must be either 'l1' or 'l2'")

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                return kernel, np.empty((*kernel.shape, 0))
            return kernel, log_gradient[:, :, None]
        return kernel

    def diag(self, X) -> np.ndarray:
        """Return the diagonal of the kernel matrix."""
        X = _scale_structure_features(X, feature_scales=self.feature_scales, name="X")
        return np.ones(X.shape[0])

    def is_stationary(self) -> bool:
        """Return whether the kernel is stationary."""
        return True

    def __repr__(self) -> str:
        return (
            "StructureFeatureKernel("
            f"length_scale={self.length_scale}, "
            f"metric={self.metric!r})"
        )


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


def pairwise_composition_distance(
    X,
    Y=None,
    *,
    metric: str = "l1",
    normalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return pairwise distances between non-negative composition vectors.

    Parameters
    ----------
    X, Y
        Two-dimensional arrays of elemental amount, count, or fraction vectors.
        If ``Y`` is omitted, distances are calculated between rows of ``X``.
    metric
        ``"l1"`` returns the sum of absolute element-fraction differences.
        ``"l2"`` returns squared Euclidean distance.
    normalize
        If ``True``, each row is normalized to sum to one before distances are
        computed.
    eps
        Small threshold used to reject all-zero composition rows.
    """
    X = _normalize_composition_vectors(X, normalize=normalize, eps=eps, name="X")
    Y = X if Y is None else _normalize_composition_vectors(Y, normalize=normalize, eps=eps, name="Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")

    difference = X[:, None, :] - Y[None, :, :]
    if metric == "l1":
        return np.sum(np.abs(difference), axis=2)
    if metric == "l2":
        return np.sum(difference * difference, axis=2)
    raise ValueError("metric must be either 'l1' or 'l2'")


def pairwise_structure_distance(
    X,
    Y=None,
    *,
    metric: str = "l2",
    feature_scales=None,
) -> np.ndarray:
    """Return pairwise distances between continuous structure descriptor rows.

    Parameters
    ----------
    X, Y
        Two-dimensional arrays of structure descriptors.
    metric
        ``"l2"`` returns squared Euclidean distance. ``"l1"`` returns summed
        absolute feature differences.
    feature_scales
        Optional positive per-feature scales. When supplied, descriptor columns
        are divided by these scales before distances are computed.
    """
    X = _scale_structure_features(X, feature_scales=feature_scales, name="X")
    Y = X if Y is None else _scale_structure_features(Y, feature_scales=feature_scales, name="Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")

    difference = X[:, None, :] - Y[None, :, :]
    if metric == "l2":
        return np.sum(difference * difference, axis=2)
    if metric == "l1":
        return np.sum(np.abs(difference), axis=2)
    raise ValueError("metric must be either 'l1' or 'l2'")


def build_element_fraction_gpr_kernel(
    *,
    constant_value: float = 1.0,
    length_scale: float = 1.0,
    noise_level: float = 1.0,
    metric: str = "l1",
    normalize: bool = True,
):
    """Build a scikit-learn GPR kernel for element-fraction composition vectors."""
    return (
        ConstantKernel(constant_value)
        * ElementFractionKernel(
            length_scale=length_scale,
            metric=metric,
            normalize=normalize,
        )
        + WhiteKernel(noise_level=noise_level)
    )


def build_structure_gpr_kernel(
    *,
    constant_value: float = 1.0,
    length_scale: float = 1.0,
    noise_level: float = 1.0,
    metric: str = "l2",
    feature_scales=None,
):
    """Build a scikit-learn GPR kernel for continuous structure descriptors."""
    return (
        ConstantKernel(constant_value)
        * StructureFeatureKernel(
            length_scale=length_scale,
            metric=metric,
            feature_scales=feature_scales,
        )
        + WhiteKernel(noise_level=noise_level)
    )


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


def _normalize_composition_vectors(
    values,
    *,
    normalize: bool,
    eps: float,
    name: str,
) -> np.ndarray:
    array = _to_2d_float_array(values, name)
    if eps <= 0:
        raise ValueError("eps must be positive")
    _validate_nonnegative(array, name)

    if not normalize:
        return array

    row_sums = np.sum(array, axis=1)
    if np.any(row_sums <= eps):
        raise ValueError(f"{name} must not contain all-zero composition rows")
    return array / row_sums[:, None]


def _scale_structure_features(values, *, feature_scales, name: str) -> np.ndarray:
    array = _to_2d_float_array(values, name)
    if feature_scales is None:
        return array

    scales = np.asarray(feature_scales, dtype=float)
    if scales.ndim != 1:
        raise ValueError("feature_scales must be a one-dimensional array")
    if scales.shape[0] != array.shape[1]:
        raise ValueError("feature_scales must match the number of structure features")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("feature_scales must contain only positive finite values")
    return array / scales


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


def _validate_length_scale(length_scale: float) -> float:
    value = float(length_scale)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("length_scale must be a positive finite number")
    return value
