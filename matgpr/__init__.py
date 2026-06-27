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
from .io_utils import load_artifact, log_experiment_result, save_artifact
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
from .visualization import (
    plot_correlation_matrix,
    plot_distribution,
    plot_learning_curve,
    plot_parity,
    plot_pca_scores,
    plot_pca_scree,
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
    "GPyTorchGPRResult",
    "GPyTorchPrediction",
    "PhysicsInformedMean",
    "CompositionFingerprintResult",
    "DEFAULT_RDKIT_DESCRIPTORS",
    "SmilesFingerprintResult",
    "append_composition_fingerprints",
    "append_smiles_features",
    "build_preprocessor",
    "build_scaler",
    "build_sklearn_gpr_grid_search",
    "build_sklearn_gpr_kernel",
    "build_sklearn_gpr_model",
    "canonicalize_molecule_smiles",
    "canonicalize_polymer_smiles",
    "clean_formula",
    "composition_fingerprint",
    "drop_columns_by_missing_fraction",
    "drop_duplicate_rows",
    "featurize_compositions",
    "featurize_smiles",
    "filter_iqr_outliers",
    "fingerprint_smiles",
    "fit_gpytorch_gpr",
    "fit_pca",
    "identify_feature_types",
    "impute_missing_values",
    "load_artifact",
    "log_experiment_result",
    "normalize_column_names",
    "plot_correlation_matrix",
    "plot_distribution",
    "plot_learning_curve",
    "plot_parity",
    "plot_pca_scores",
    "plot_pca_scree",
    "predict_gpytorch_gpr",
    "regression_metrics",
    "replace_missing_placeholders",
    "save_artifact",
    "separate_features_target",
    "split_train_test",
    "summarize_missingness",
    "summarize_numeric_columns",
    "summarize_pca",
    "train_gpytorch_gpr",
    "train_test_regression_metrics",
    "transform_pca",
]
