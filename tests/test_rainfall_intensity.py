
"""Synthetic tests for A1 left-truncated Gamma intensity model.

No real calibration or holdout rainfall files are accessed.
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import gamma

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location(
    "rainfall_intensity",
    SRC / "rainfall_intensity.py",
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class RainfallIntensityTests(unittest.TestCase):

    def test_valid_wet_amounts(self):
        values = [1.0, 2.5, 10.0]

        result = model.validate_wet_amounts(values)

        np.testing.assert_array_equal(result, values)

    def test_below_threshold_rejected(self):
        with self.assertRaises(ValueError):
            model.validate_wet_amounts([0.99, 2.0])

    def test_invalid_amounts_rejected(self):
        invalid = [
            [1.0, float("nan")],
            [1.0, float("inf")],
            [1.0, -2.0],
            [1.0],
            [],
        ]

        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    model.validate_wet_amounts(values)

    def test_invalid_parameters_rejected(self):
        for shape, scale in [
            (0, 2),
            (-1, 2),
            (1, 0),
            (1, -2),
            (float("nan"), 2),
        ]:
            with self.subTest(shape=shape, scale=scale):
                with self.assertRaises(ValueError):
                    model.truncated_gamma_logpdf(
                        [1.0, 2.0],
                        shape,
                        scale,
                    )

    def test_logpdf_matches_conditional_formula(self):
        values = np.array([1.0, 3.0, 12.0])
        shape = 0.8
        scale = 10.0
        threshold = 1.0

        actual = model.truncated_gamma_logpdf(
            values, shape, scale, threshold
        )

        expected = (
            gamma.logpdf(values, a=shape, scale=scale)
            - gamma.logsf(threshold, a=shape, scale=scale)
        )

        np.testing.assert_allclose(
            actual, expected, rtol=1e-12, atol=1e-12
        )

    def test_truncated_density_normalizes(self):
        from scipy.integrate import quad

        shape = 0.8
        scale = 10.0

        def density(x):
            return math.exp(
                model.truncated_gamma_logpdf(
                    [x, x], shape, scale
                )[0]
            )

        integral, error = quad(
            density, 1.0, math.inf
        )

        self.assertAlmostEqual(integral, 1.0, places=6)
        self.assertLess(error, 1e-5)

    def test_synthetic_parameter_recovery(self):
        """Fitting should approximately recover known parameters."""
        rng = np.random.default_rng(20261123)

        true_shape = 1.5
        true_scale = 8.0
        threshold = 1.0

        survival = gamma.sf(
            threshold,
            a=true_shape,
            scale=true_scale,
        )

        # Inverse-survival sampling conditional on X >= threshold.
        u = rng.uniform(0.0, 1.0, size=5000)

        values = gamma.isf(
            u * survival,
            a=true_shape,
            scale=true_scale,
        )

        fitted = model.fit_truncated_gamma(values)

        self.assertTrue(fitted["optimization_success"])
        self.assertEqual(fitted["n_wet_days"], 5000)

        self.assertAlmostEqual(
            fitted["shape"], true_shape, delta=0.25
        )
        self.assertAlmostEqual(
            fitted["scale"], true_scale, delta=1.5
        )


if __name__ == "__main__":
    unittest.main()
