
"""Synthetic tests for A2 shifted mixed-exponential model.

No calibration or reserved holdout observations are accessed.
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location(
    "rainfall_intensity",
    SRC / "rainfall_intensity.py",
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class MixedExponentialTests(unittest.TestCase):

    def test_known_mixture_density(self):
        values = np.array([1.0, 2.0, 10.0])

        weight = 0.2
        small = 5.0
        large = 20.0

        actual = model.mixed_exponential_logpdf(
            values, weight, small, large
        )

        excess = values - 1.0

        expected_density = (
            (1 - weight) / small
            * np.exp(-excess / small)
            + weight / large
            * np.exp(-excess / large)
        )

        np.testing.assert_allclose(
            actual,
            np.log(expected_density),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_density_integrates_to_one(self):
        def density(x):
            return math.exp(
                model.mixed_exponential_logpdf(
                    [x, x],
                    0.2,
                    5.0,
                    20.0,
                )[0]
            )

        integral, error = quad(
            density,
            1.0,
            math.inf,
        )

        self.assertAlmostEqual(
            integral, 1.0, places=6
        )
        self.assertLess(error, 1e-5)

    def test_threshold_density(self):
        actual = model.mixed_exponential_logpdf(
            [1.0, 1.0],
            0.25,
            4.0,
            16.0,
        )

        expected = math.log(
            0.75 / 4.0 + 0.25 / 16.0
        )

        self.assertAlmostEqual(
            actual[0], expected
        )

    def test_below_threshold_rejected(self):
        with self.assertRaises(ValueError):
            model.mixed_exponential_logpdf(
                [0.9, 2.0],
                0.2,
                5.0,
                20.0,
            )

    def test_invalid_parameters_rejected(self):
        invalid = [
            (0.0, 5.0, 20.0),
            (1.0, 5.0, 20.0),
            (-0.1, 5.0, 20.0),
            (1.1, 5.0, 20.0),
            (0.2, 0.0, 20.0),
            (0.2, -1.0, 20.0),
            (0.2, 20.0, 5.0),
            (0.2, 5.0, 5.0),
            (float("nan"), 5.0, 20.0),
            (0.2, 5.0, float("inf")),
        ]

        for weight, small, large in invalid:
            with self.subTest(
                weight=weight,
                small=small,
                large=large,
            ):
                with self.assertRaises(ValueError):
                    model.mixed_exponential_logpdf(
                        [1.0, 2.0],
                        weight,
                        small,
                        large,
                    )

    def test_synthetic_parameter_recovery(self):
        rng = np.random.default_rng(20261123)

        n = 12000
        true_weight = 0.25
        true_small = 4.0
        true_large = 25.0

        is_large = rng.random(n) < true_weight

        scales = np.where(
            is_large,
            true_large,
            true_small,
        )

        values = 1.0 + rng.exponential(
            scale=scales
        )

        fitted = model.fit_mixed_exponential(
            values
        )

        self.assertTrue(
            fitted["optimization_success"]
        )
        self.assertEqual(
            fitted["n_wet_days"], n
        )

        self.assertAlmostEqual(
            fitted["weight_large"],
            true_weight,
            delta=0.10,
        )

        self.assertAlmostEqual(
            fitted["scale_small"],
            true_small,
            delta=1.5,
        )

        self.assertAlmostEqual(
            fitted["scale_large"],
            true_large,
            delta=5.0,
        )

    def test_positive_excess_required(self):
        with self.assertRaises(ValueError):
            model.fit_mixed_exponential(
                [1.0, 1.0, 1.0]
            )

    def test_inputs_not_modified(self):
        values = np.array(
            [1.0, 2.0, 5.0, 10.0, 20.0]
        )
        original = values.copy()

        model.mixed_exponential_logpdf(
            values,
            0.2,
            5.0,
            20.0,
        )

        np.testing.assert_array_equal(
            values, original
        )


if __name__ == "__main__":
    unittest.main()
