from __future__ import annotations

import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from matgpr import (
    ElementFractionKernel,
    FeatureSubsetKernel,
    StructureFeatureKernel,
    TanimotoKernel,
    build_additive_kernel,
    build_element_fraction_gpr_kernel,
    build_product_kernel,
    build_sklearn_gpr_kernel,
    build_sklearn_gpr_model,
    build_structure_gpr_kernel,
    build_tanimoto_gpr_kernel,
    pairwise_composition_distance,
    pairwise_structure_distance,
    pairwise_tanimoto_similarity,
)


class TanimotoKernelTests(unittest.TestCase):
    def test_pairwise_tanimoto_similarity_matches_manual_values(self):
        x = np.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        y = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        similarity = pairwise_tanimoto_similarity(x, y)

        expected = np.array(
            [
                [0.5, 0.0],
                [0.5, 0.0],
                [0.0, 1.0],
            ]
        )
        self.assertTrue(np.allclose(similarity, expected))

    def test_tanimoto_kernel_diag_gradient_and_validation(self):
        x = np.array([[1.0, 0.0], [1.0, 1.0]])
        kernel = TanimotoKernel()

        value, gradient = kernel(x, eval_gradient=True)

        self.assertTrue(np.allclose(np.diag(value), [1.0, 1.0]))
        self.assertEqual(gradient.shape, (2, 2, 0))
        self.assertTrue(np.allclose(kernel.diag(x), [1.0, 1.0]))
        self.assertFalse(kernel.is_stationary())

        with self.assertRaises(ValueError):
            kernel([[1.0, -1.0]])
        with self.assertRaises(ValueError):
            pairwise_tanimoto_similarity([[1.0, 0.0]], [[1.0, 0.0, 1.0]])

    def test_tanimoto_kernel_can_fit_sklearn_gpr(self):
        x = np.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        y = np.array([1.0, 1.2, 0.8, 1.5])
        kernel = (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * TanimotoKernel()
            + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed")
        )

        model = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
        model.fit(x, y)
        mean, std = model.predict(x[:2], return_std=True)

        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(std >= 0.0))

    def test_build_sklearn_tanimoto_kernel(self):
        kernel = build_sklearn_gpr_kernel("tanimoto", noise_level=0.1)
        direct = build_tanimoto_gpr_kernel(noise_level=0.1)

        x = np.array([[1.0, 0.0], [1.0, 1.0]])
        self.assertTrue(np.allclose(kernel(x), direct(x)))


class KernelCompositionTests(unittest.TestCase):
    def test_pairwise_composition_distance_normalizes_element_counts(self):
        x = np.array(
            [
                [4.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        y = np.array([[0.0, 1.0, 1.0]])

        distance = pairwise_composition_distance(x, y, metric="l1")

        expected = np.array([[1.6], [1.0]])
        self.assertTrue(np.allclose(distance, expected))

    def test_element_fraction_kernel_l1_gradient_and_validation(self):
        x = np.array(
            [
                [4.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        kernel = ElementFractionKernel(length_scale=2.0, metric="l1")

        value, gradient = kernel(x, eval_gradient=True)

        expected_off_diagonal = np.exp(-0.6 / 2.0)
        self.assertTrue(np.allclose(np.diag(value), [1.0, 1.0]))
        self.assertAlmostEqual(value[0, 1], expected_off_diagonal)
        self.assertEqual(gradient.shape, (2, 2, 1))
        self.assertTrue(np.allclose(kernel.diag(x), [1.0, 1.0]))
        self.assertFalse(kernel.is_stationary())

        with self.assertRaises(ValueError):
            kernel([[0.0, 0.0]])
        with self.assertRaises(ValueError):
            ElementFractionKernel(metric="cosine")(x)

    def test_element_fraction_kernel_can_fit_sklearn_gpr(self):
        x = np.array(
            [
                [4.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )
        y = np.array([28.0, 32.0, 45.0, 40.0])
        kernel = build_element_fraction_gpr_kernel(noise_level=1e-4)

        model = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
        model.fit(x, y)
        mean, std = model.predict(x[:2], return_std=True)

        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(std >= 0.0))

    def test_build_sklearn_composition_kernel(self):
        kernel = build_sklearn_gpr_kernel("composition", noise_level=0.1)
        direct = build_element_fraction_gpr_kernel(noise_level=0.1)

        x = np.array([[4.0, 1.0], [1.0, 1.0]])
        self.assertTrue(np.allclose(kernel(x), direct(x)))

    def test_build_sklearn_element_fraction_model_optimizes(self):
        x = np.array(
            [
                [4.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )
        y = np.array([28.0, 32.0, 45.0, 40.0])
        model = build_sklearn_gpr_model(
            kernel="element_fraction",
            n_restarts_optimizer=0,
            random_state=0,
        )

        model.fit(x, y)
        mean, std = model.predict(x[:2], return_std=True)

        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(std >= 0.0))

    def test_feature_subset_kernel_applies_kernel_to_selected_columns(self):
        x = np.array(
            [
                [1.0, 0.0, 0.1],
                [1.0, 1.0, 0.2],
            ]
        )
        tanimoto_subset = FeatureSubsetKernel(TanimotoKernel(), columns=[0, 1])
        rbf_subset = FeatureSubsetKernel(RBF(length_scale=1.0), columns=[2])

        additive = build_additive_kernel(tanimoto_subset, rbf_subset)
        expected = TanimotoKernel()(x[:, [0, 1]]) + RBF(length_scale=1.0)(x[:, [2]])

        self.assertTrue(np.allclose(additive(x), expected))
        self.assertTrue(np.allclose(tanimoto_subset.diag(x), [1.0, 1.0]))

        with self.assertRaises(ValueError):
            FeatureSubsetKernel(TanimotoKernel(), columns=[3])(x)

    def test_product_kernel_helper_and_validation(self):
        x = np.array([[1.0, 0.0], [1.0, 1.0]])
        kernel = build_product_kernel(TanimotoKernel(), ConstantKernel(2.0))

        self.assertTrue(np.allclose(kernel(x), 2.0 * TanimotoKernel()(x)))
        with self.assertRaises(ValueError):
            build_additive_kernel(TanimotoKernel())
        with self.assertRaises(TypeError):
            build_product_kernel(TanimotoKernel(), "not a kernel")


class StructureKernelTests(unittest.TestCase):
    def test_pairwise_structure_distance_uses_feature_scales(self):
        x = np.array([[1.0, 2.0], [2.0, 2.0]])
        y = np.array([[3.0, 6.0]])

        distance = pairwise_structure_distance(
            x,
            y,
            metric="l2",
            feature_scales=[2.0, 4.0],
        )

        expected = np.array([[2.0], [1.25]])
        self.assertTrue(np.allclose(distance, expected))

    def test_structure_feature_kernel_l2_gradient_and_validation(self):
        x = np.array([[1.0, 2.0], [2.0, 2.0]])
        kernel = StructureFeatureKernel(length_scale=2.0, metric="l2")

        value, gradient = kernel(x, eval_gradient=True)

        expected_off_diagonal = np.exp(-0.5 * 1.0 / 4.0)
        self.assertTrue(np.allclose(np.diag(value), [1.0, 1.0]))
        self.assertAlmostEqual(value[0, 1], expected_off_diagonal)
        self.assertEqual(gradient.shape, (2, 2, 1))
        self.assertTrue(np.allclose(kernel.diag(x), [1.0, 1.0]))
        self.assertTrue(kernel.is_stationary())

        with self.assertRaises(ValueError):
            StructureFeatureKernel(metric="cosine")(x)
        with self.assertRaises(ValueError):
            StructureFeatureKernel(feature_scales=[1.0, 0.0])(x)

    def test_structure_feature_kernel_can_fit_sklearn_gpr(self):
        x = np.array(
            [
                [1.2, 0.0, 2.1],
                [1.3, 0.0, 2.0],
                [1.7, 0.1, 2.8],
                [1.8, 0.1, 2.9],
            ]
        )
        y = np.array([80.0, 78.0, 25.0, 24.0])
        kernel = build_structure_gpr_kernel(noise_level=1e-4)

        model = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
        model.fit(x, y)
        mean, std = model.predict(x[:2], return_std=True)

        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(std >= 0.0))

    def test_build_sklearn_structure_model_optimizes(self):
        x = np.array(
            [
                [1.2, 0.0, 2.1],
                [1.3, 0.0, 2.0],
                [1.7, 0.1, 2.8],
                [1.8, 0.1, 2.9],
            ]
        )
        y = np.array([80.0, 78.0, 25.0, 24.0])
        model = build_sklearn_gpr_model(
            kernel="structure",
            n_restarts_optimizer=0,
            random_state=0,
        )

        model.fit(x, y)
        mean, std = model.predict(x[:2], return_std=True)

        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(std >= 0.0))


if __name__ == "__main__":
    unittest.main()
