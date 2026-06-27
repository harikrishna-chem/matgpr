"""Execute reduced versions of example notebooks for CI smoke testing.

The full notebooks are scientific benchmark workflows and can be expensive.
This script modifies notebooks in memory only, reducing repeats, optimizer
iterations, cross-validation folds, and SHAP sample sizes. It verifies that the
notebook code paths execute without committing generated outputs.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = [
    PROJECT_ROOT / "examples" / "opv" / "opv_gpr_modeling.ipynb",
    PROJECT_ROOT / "examples" / "hardness" / "hardness_gpr_modeling.ipynb",
    PROJECT_ROOT / "examples" / "gas_transport" / "gas_transport_gpr_modeling.ipynb",
    PROJECT_ROOT
    / "examples"
    / "solvent_diffusivity"
    / "solvent_diffusivity_gpr_modeling.ipynb",
    PROJECT_ROOT / "examples" / "spall_strength" / "spall_strength_gpr_modeling.ipynb",
]

REPLACEMENTS = {
    "LEARNING_CURVE_REPEATS = 5": "LEARNING_CURVE_REPEATS = 1",
    "LEARNING_CURVE_PERCENTS = np.arange(10, 71, 10)": "LEARNING_CURVE_PERCENTS = np.array([10])",
    "TRAINING_PERCENTS = np.arange(10, 71, 10)": "TRAINING_PERCENTS = np.array([10])",
    "TRAIN_PERCENTS = list(range(10, MAX_TRAIN_PERCENT + 1, 10))": "TRAIN_PERCENTS = [10]",
    "TRAINING_PERCENTS = np.array([10, 30, 50, 70, 90, 100])": "TRAINING_PERCENTS = np.array([10])",
    "N_REPEATS = 20": "N_REPEATS = 1",
    "N_RANDOM_SPLITS = 10": "N_RANDOM_SPLITS = 1",
    "N_CV_SPLITS = 10": "N_CV_SPLITS = 2",
    "CV_SPLITS = 10": "CV_SPLITS = 2",
    "SELECTION_PERCENT = 20": "SELECTION_PERCENT = 10",
    "TRAINING_ITER = 150": "TRAINING_ITER = 2",
    "TRAINING_ITER = 120": "TRAINING_ITER = 2",
    "TRAINING_ITER = 100": "TRAINING_ITER = 2",
    "PRODUCTION_TRAINING_ITER = 300": "PRODUCTION_TRAINING_ITER = 2",
    "PRODUCTION_TRAINING_ITER = 250": "PRODUCTION_TRAINING_ITER = 2",
    "PRODUCTION_TRAINING_ITER = 200": "PRODUCTION_TRAINING_ITER = 2",
    "PRODUCTION_TRAINING_ITER = 120": "PRODUCTION_TRAINING_ITER = 2",
    "LOW_DATA_TRAINING_ITER = 200": "LOW_DATA_TRAINING_ITER = 2",
    "FINAL_TRAINING_ITER = 300": "FINAL_TRAINING_ITER = 2",
    "SHAP_BACKGROUND_SIZE = min(60, len(X))": "SHAP_BACKGROUND_SIZE = min(5, len(X))",
    "SHAP_EXPLAIN_SIZE = min(120, len(X))": "SHAP_EXPLAIN_SIZE = min(5, len(X))",
    "SHAP_MAX_FEATURES = min(30, len(production_feature_columns))": "SHAP_MAX_FEATURES = min(3, len(production_feature_columns))",
    "SHAP_MAX_FEATURES = min(20, len(production_feature_columns))": "SHAP_MAX_FEATURES = min(3, len(production_feature_columns))",
    "SHAP_BACKGROUND_SIZE = min(50, len(model_data))": "SHAP_BACKGROUND_SIZE = min(5, len(model_data))",
    "SHAP_BACKGROUND_SIZE = min(40, len(model_data))": "SHAP_BACKGROUND_SIZE = min(5, len(model_data))",
    "SHAP_BACKGROUND_SIZE = min(35, len(model_data))": "SHAP_BACKGROUND_SIZE = min(5, len(model_data))",
    "SHAP_EXPLAIN_SIZE = min(80, len(model_data))": "SHAP_EXPLAIN_SIZE = min(5, len(model_data))",
    "SHAP_EXPLAIN_SIZE = min(60, len(model_data))": "SHAP_EXPLAIN_SIZE = min(5, len(model_data))",
    "SHAP_EXPLAIN_SIZE = min(50, len(model_data))": "SHAP_EXPLAIN_SIZE = min(5, len(model_data))",
}

SAMPLE_LINES = [
    "model_data = model_data.dropna().reset_index(drop=True)",
    "model_data = raw_data.dropna(subset=[TARGET_COLUMN, *FEATURE_COLUMNS]).reset_index(drop=True)",
]


def main() -> None:
    for path in NOTEBOOKS:
        execute_reduced_notebook(path)
        print(f"smoke passed: {path.relative_to(PROJECT_ROOT)}")


def execute_reduced_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    sampled = False

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        for old, new in REPLACEMENTS.items():
            source = source.replace(old, new)

        if not sampled:
            for line in SAMPLE_LINES:
                if line in source and "sample(n=min(120, len(model_data))" not in source:
                    source = source.replace(
                        line,
                        line
                        + "\nmodel_data = model_data.sample(n=min(120, len(model_data)), "
                        "random_state=RANDOM_STATE).reset_index(drop=True)",
                        1,
                    )
                    sampled = True
                    break
        cell.source = source

    processor = ExecutePreprocessor(timeout=900, kernel_name="python3")
    processor.preprocess(notebook, {"metadata": {"path": str(path.parent)}})


if __name__ == "__main__":
    main()
