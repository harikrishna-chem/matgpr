# Data, Metrics, And Utilities

## Data Cleaning

::: matgpr.data_cleaning

## Data Splitting

::: matgpr.data_splitting

## Preprocessing

::: matgpr.preprocessing

## Metrics

The numerical metric helpers keep scientifically undefined values as `NaN`.
Use the JSON-safe companion helpers when metrics must be serialized through a
strict JSON encoder or API response.

::: matgpr.metrics

## PCA

::: matgpr.pca

## Reporting

::: matgpr.reporting

## Artifact IO

`log_experiment_result` appends experiment rows to a schema-safe CSV log. When
new runs add or omit metric columns, the existing file is realigned to the
union of columns and remains readable by `pandas.read_csv`.

::: matgpr.io_utils

## Optional Dependencies

::: matgpr.optional_dependencies

## Scikit-Learn GPR Helpers

::: matgpr.sklearn_gpr
