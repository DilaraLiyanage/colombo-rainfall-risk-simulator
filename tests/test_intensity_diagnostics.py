
"""Synthetic tests for A1/A2 intensity diagnostics.

No calibration or holdout datasets are accessed.
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
    "intensity_diagnostics",
    SRC / "intensity_diagnostics.py",
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def example_parameters(n=6):
    a1 = {
        "shape": 1.5,
        "scale": 8.0,
        "threshold_mm": 1.0,
        "n_wet_days": n,
        "log_likelihood": -20.0,
    }

    a2 = {
        "weight_large": 0.2,
        "scale_small": 5.0,
        "scale_large": 20.0,
        "threshold_mm": 1.0,
        "n_wet_days": n,
        "log_likelihood": -18.0,
    }

    return a1, a2


class IntensityDiagnosticsTests(unittest.TestCase):

    def test_a1_survival_matches_conditional_formula(self):
        a1, _ = example_parameters()

        actual = model.a1_survival(20.0, a1)

        expected = (
            gamma.sf(20.0, a=1.5, scale=8.0)
            / gamma.sf(1.0, a=1.5, scale=8.0)
        )

        self.assertAlmostEqual(actual, expected)

    def test_a2_survival_matches_mixture_formula(self):
        _, a2 = example_parameters()

        actual = model.a2_survival(11.0, a2)

        expected = (
            0.8 * math.exp(-10.0 / 5.0)
            + 0.2 * math.exp(-10.0 / 20.0)
        )

        self.assertAlmostEqual(actual, expected)

    def test_survival_at_or_below_threshold(self):
        a1, a2 = example_parameters()

        for x in (0.0, 1.0):
            with self.subTest(x=x):
                self.assertEqual(model.a1_survival(x, a1), 1.0)
                self.assertEqual(model.a2_survival(x, a2), 1.0)

    def test_quantile_survival_consistency(self):
        a1, a2 = example_parameters()

        for probability in (0.1, 0.5, 0.9, 0.99, 0.995):
            with self.subTest(p=probability):
                x1 = model.a1_quantile(probability, a1)
                x2 = model.a2_quantile(probability, a2)

                self.assertAlmostEqual(
                    model.a1_survival(x1, a1),
                    1.0 - probability,
                    places=9,
                )

                self.assertAlmostEqual(
                    model.a2_survival(x2, a2),
                    1.0 - probability,
                    places=9,
                )

    def test_quantile_boundaries(self):
        a1, a2 = example_parameters()

        self.assertEqual(model.a1_quantile(0, a1), 1.0)
        self.assertEqual(model.a2_quantile(0, a2), 1.0)

        self.assertTrue(math.isinf(model.a1_quantile(1, a1)))
        self.assertTrue(math.isinf(model.a2_quantile(1, a2)))

    def test_invalid_quantiles_rejected(self):
        a1, a2 = example_parameters()

        for probability in (-0.1, 1.1, float("nan")):
            with self.subTest(p=probability):
                with self.assertRaises(ValueError):
                    model.a1_quantile(probability, a1)

                with self.assertRaises(ValueError):
                    model.a2_quantile(probability, a2)

    def test_information_criteria_and_counts(self):
        amounts = np.array([
            1.0, 2.0, 5.0, 10.0, 50.0, 150.0,
        ])

        a1, a2 = example_parameters(n=len(amounts))

        results = model.intensity_diagnostics(
            amounts, a1, a2
        )

        comparison = results["comparison"]

        self.assertEqual(results["n_wet_days"], 6)

        self.assertAlmostEqual(
            comparison["A1"]["aic"],
            2 * 2 - 2 * (-20.0),
        )

        self.assertAlmostEqual(
            comparison["A2"]["bic"],
            3 * math.log(6) - 2 * (-18.0),
        )

        exceedances = {
            row["threshold_mm"]: row
            for row in results["exceedances"]
        }

        self.assertEqual(
            exceedances[50.0]["observed_count"], 2
        )
        self.assertEqual(
            exceedances[100.0]["observed_count"], 1
        )
        self.assertEqual(
            exceedances[150.0]["observed_count"], 1
        )

        self.assertAlmostEqual(
            exceedances[50.0]["a2_expected_count"],
            6 * model.a2_survival(50.0, a2),
        )

    def test_invalid_observations_rejected(self):
        a1, a2 = example_parameters(n=2)

        invalid = [
            [0.5, 2.0],
            [1.0, float("nan")],
            [1.0, float("inf")],
            [1.0],
            [],
        ]

        for amounts in invalid:
            with self.subTest(amounts=amounts):
                with self.assertRaises(ValueError):
                    model.intensity_diagnostics(
                        amounts, a1, a2
                    )

    def test_sample_size_mismatch_rejected(self):
        amounts = [1.0, 2.0, 3.0]

        a1, a2 = example_parameters(n=6)

        with self.assertRaises(ValueError):
            model.intensity_diagnostics(
                amounts, a1, a2
            )

    def test_input_observations_not_modified(self):
        amounts = np.array([
            1.0, 2.0, 5.0, 10.0, 50.0, 150.0,
        ])

        original = amounts.copy()
        a1, a2 = example_parameters(n=len(amounts))

        model.intensity_diagnostics(amounts, a1, a2)

        np.testing.assert_array_equal(amounts, original)


if __name__ == "__main__":
    unittest.main()
