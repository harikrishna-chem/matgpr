from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from matgpr import (
    drop_columns_by_missing_fraction,
    drop_duplicate_rows,
    filter_iqr_outliers,
    impute_missing_values,
    replace_missing_placeholders,
)


class DataCleaningTests(unittest.TestCase):
    def test_replace_missing_placeholders_is_vectorized_and_preserves_numeric_columns(self):
        data = pd.DataFrame(
            {
                "label": ["valid", " NA ", "None", pd.NA],
                "value": [0, 1, 2, 3],
            }
        )

        cleaned = replace_missing_placeholders(data)

        self.assertEqual(cleaned["value"].tolist(), [0, 1, 2, 3])
        self.assertFalse(pd.isna(cleaned.loc[0, "label"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "label"]))
        self.assertTrue(pd.isna(cleaned.loc[2, "label"]))
        self.assertTrue(pd.isna(cleaned.loc[3, "label"]))

    def test_drop_duplicate_rows_can_preserve_material_index(self):
        data = pd.DataFrame({"formula": ["B4C", "B4C", "BN"]}, index=["m1", "m2", "m3"])

        cleaned = drop_duplicate_rows(data, subset=["formula"], reset_index=False)

        self.assertEqual(cleaned.index.tolist(), ["m1", "m3"])

    def test_drop_columns_by_missing_fraction_can_return_dropped_columns(self):
        data = pd.DataFrame({"good": [1.0, 2.0, np.nan], "bad": [1.0, np.nan, np.nan]})

        cleaned, dropped = drop_columns_by_missing_fraction(
            data,
            max_missing_fraction=0.5,
            return_dropped=True,
        )

        self.assertEqual(cleaned.columns.tolist(), ["good"])
        self.assertEqual(dropped, ["bad"])

    def test_impute_missing_values_can_return_fitted_imputer(self):
        data = pd.DataFrame({"x": [1.0, np.nan, 3.0]})

        cleaned, imputer = impute_missing_values(data, return_imputer=True)

        self.assertEqual(cleaned["x"].tolist(), [1.0, 2.0, 3.0])
        self.assertTrue(hasattr(imputer, "statistics_"))

    def test_filter_iqr_outliers_supports_any_and_all_policies(self):
        data = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 100.0],
                "y": [1.0, 2.0, 3.0, 4.0],
            },
            index=["a", "b", "c", "d"],
        )

        drop_any = filter_iqr_outliers(data, ["x", "y"], drop_if="any", reset_index=False)
        drop_all = filter_iqr_outliers(data, ["x", "y"], drop_if="all", reset_index=False)

        self.assertEqual(drop_any.index.tolist(), ["a", "b", "c"])
        self.assertEqual(drop_all.index.tolist(), ["a", "b", "c", "d"])

    def test_filter_iqr_outliers_skips_constant_and_all_missing_columns(self):
        data = pd.DataFrame({"constant": [1.0, 1.0, 1.0], "missing": [np.nan, np.nan, np.nan]})

        cleaned = filter_iqr_outliers(data, ["constant", "missing"])

        self.assertEqual(cleaned.shape, data.shape)


if __name__ == "__main__":
    unittest.main()
