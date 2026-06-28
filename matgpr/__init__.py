"""Utilities for data preparation, Gaussian Process Regression, and analysis."""

from .data_cleaning import (
    drop_columns_by_missing_fraction,
    drop_duplicate_rows,
    filter_iqr_outliers,
    impute_missing_values,
    normalize_column_names,
    replace_missing_placeholders,
)
from .data_splitting import separate_features_target, split_train_test
from .estimators import MatGPRRegressor, PhysicsInformedGPRRegressor
from .featurizers import CompositionFeaturizer, PolymerSmilesFeaturizer, SmilesFeaturizer
from .fingerprint_cache import FINGERPRINT_CACHE_SCHEMA_VERSION, fingerprint_cache_key
from .io_utils import load_artifact, log_experiment_result, save_artifact
from .kernels import (
    FeatureSubsetKernel,
    TanimotoKernel,
    build_additive_kernel,
    build_product_kernel,
    build_tanimoto_gpr_kernel,
    pairwise_tanimoto_similarity,
)
from .metrics import regression_metrics, train_test_regression_metrics
from .inorganic_fingerprints import (
    CompositionFingerprintResult,
    append_composition_fingerprints,
    clean_formula,
    composition_fingerprint,
    featurize_compositions,
)
from .organic_fingerprints import (
    DEFAULT_RDKIT_DESCRIPTORS,
    SmilesFingerprintResult,
    append_smiles_features,
    canonicalize_molecule_smiles,
    canonicalize_polymer_smiles,
    featurize_smiles,
    fingerprint_smiles,
)
from .pca import fit_pca, summarize_pca, transform_pca
from .preprocessing import build_preprocessor, build_scaler, identify_feature_types
from .reporting import summarize_missingness, summarize_numeric_columns
from .sklearn_gpr import (
    build_sklearn_gpr_grid_search,
    build_sklearn_gpr_kernel,
    build_sklearn_gpr_model,
)
from .target_transforms import (
    IdentityTargetTransform,
    LogTargetTransform,
    PhysicsResidualTransform,
    StandardizedTargetTransform,
    make_target_transform,
)
from .uncertainty import (
    calibration_curve,
    gaussian_nlpd,
    interval_coverage,
    prediction_interval_bounds,
    standardized_residuals,
    uncertainty_diagnostics,
    uncertainty_error_correlation,
)
from .visualization import (
    plot_correlation_matrix,
    plot_distribution,
    plot_learning_curve,
    plot_parity,
    plot_pca_scores,
    plot_pca_scree,
    plot_uncertainty_calibration,
    plot_uncertainty_vs_error,
)

try:
    from .gpytorch_gpr import (
        EquationMeanFunction,
        ExactGPRModel,
        GPyTorchGPRResult,
        GPyTorchPrediction,
        PhysicsInformedMean,
        fit_gpytorch_gpr,
        predict_gpytorch_gpr,
        train_gpytorch_gpr,
    )
except ImportError:
    EquationMeanFunction = None
    ExactGPRModel = None
    GPyTorchGPRResult = None
    GPyTorchPrediction = None
    PhysicsInformedMean = None
    fit_gpytorch_gpr = None
    predict_gpytorch_gpr = None
    train_gpytorch_gpr = None

__all__ = [
    "EquationMeanFunction",
    "ExactGPRModel",
    "FeatureSubsetKernel",
    "GPyTorchGPRResult",
    "GPyTorchPrediction",
    "IdentityTargetTransform",
    "LogTargetTransform",
    "MatGPRRegressor",
    "PhysicsInformedMean",
    "PhysicsInformedGPRRegressor",
    "PhysicsResidualTransform",
    "CompositionFingerprintResult",
    "CompositionFeaturizer",
    "DEFAULT_RDKIT_DESCRIPTORS",
    "FINGERPRINT_CACHE_SCHEMA_VERSION",
    "PolymerSmilesFeaturizer",
    "SmilesFingerprintResult",
    "SmilesFeaturizer",
    "StandardizedTargetTransform",
    "TanimotoKernel",
    "append_composition_fingerprints",
    "append_smiles_features",
    "build_additive_kernel",
    "build_product_kernel",
    "build_preprocessor",
    "build_scaler",
    "build_sklearn_gpr_grid_search",
    "build_sklearn_gpr_kernel",
    "build_sklearn_gpr_model",
    "build_tanimoto_gpr_kernel",
    "canonicalize_molecule_smiles",
    "canonicalize_polymer_smiles",
    "calibration_curve",
    "clean_formula",
    "composition_fingerprint",
    "drop_columns_by_missing_fraction",
    "drop_duplicate_rows",
    "featurize_compositions",
    "featurize_smiles",
    "filter_iqr_outliers",
    "fingerprint_smiles",
    "fingerprint_cache_key",
    "fit_gpytorch_gpr",
    "fit_pca",
    "gaussian_nlpd",
    "identify_feature_types",
    "impute_missing_values",
    "interval_coverage",
    "load_artifact",
    "log_experiment_result",
    "make_target_transform",
    "normalize_column_names",
    "pairwise_tanimoto_similarity",
    "plot_correlation_matrix",
    "plot_distribution",
    "plot_learning_curve",
    "plot_parity",
    "plot_pca_scores",
    "plot_pca_scree",
    "plot_uncertainty_calibration",
    "plot_uncertainty_vs_error",
    "prediction_interval_bounds",
    "predict_gpytorch_gpr",
    "regression_metrics",
    "replace_missing_placeholders",
    "save_artifact",
    "separate_features_target",
    "split_train_test",
    "standardized_residuals",
    "summarize_missingness",
    "summarize_numeric_columns",
    "summarize_pca",
    "train_gpytorch_gpr",
    "train_test_regression_metrics",
    "transform_pca",
    "uncertainty_diagnostics",
    "uncertainty_error_correlation",
]
