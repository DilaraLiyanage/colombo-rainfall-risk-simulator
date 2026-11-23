
"""Synthetic tests for Markov occurrence-model comparison.

Does not load calibration or holdout rainfall datasets.
"""

import importlib.util
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Make sibling source modules importable.
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location(
    "model_validation",
    SRC / "model_validation.py",
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class ModelValidationTests(unittest.TestCase):

    def test_known_log_likelihood(self):
        counts = {
            "N00": 3,
            "N01": 1,
            "N10": 2,
            "N11": 4,
        }

        probabilities = {
            "P00": 0.75,
            "P01": 0.25,
            "P10": 1 / 3,
            "P11": 2 / 3,
        }

        expected = (
            3 * math.log(0.75)
            + math.log(0.25)
            + 2 * math.log(1 / 3)
            + 4 * math.log(2 / 3)
        )

        actual = model.transition_log_likelihood(
            counts, probabilities
        )

        self.assertAlmostEqual(actual, expected)

    def test_zero_count_contributes_zero(self):
        counts = {
            "N00": 5,
            "N01": 0,
            "N10": 1,
            "N11": 4,
        }

        probabilities = {
            "P00": 1.0,
            "P01": 0.0,
            "P10": 0.2,
            "P11": 0.8,
        }

        actual = model.transition_log_likelihood(
            counts, probabilities
        )

        expected = math.log(0.2) + 4 * math.log(0.8)

        self.assertAlmostEqual(actual, expected)

    def test_positive_count_with_zero_probability(self):
        counts = {
            "N00": 1,
            "N01": 1,
            "N10": 1,
            "N11": 1,
        }

        probabilities = {
            "P00": 1.0,
            "P01": 0.0,
            "P10": 0.5,
            "P11": 0.5,
        }

        actual = model.transition_log_likelihood(
            counts, probabilities
        )

        self.assertEqual(actual, float("-inf"))

    def test_information_criteria(self):
        group = {
            "counts": {
                "N00": 3,
                "N01": 1,
                "N10": 2,
                "N11": 4,
            },
            "probabilities": {
                "P00": 0.75,
                "P01": 0.25,
                "P10": 1 / 3,
                "P11": 2 / 3,
            },
        }

        result = model.evaluate_model(
            "TEST",
            {"ALL": group},
        )

        ll = model.transition_log_likelihood(
            group["counts"],
            group["probabilities"],
        )

        self.assertEqual(result["parameters"], 2)
        self.assertEqual(result["transitions"], 10)
        self.assertEqual(result["minimum_denominator"], 4)

        self.assertAlmostEqual(
            result["aic"],
            4 - 2 * ll,
        )

        self.assertAlmostEqual(
            result["bic"],
            2 * math.log(10) - 2 * ll,
        )

    def test_parameter_counts(self):
        group = {
            "counts": {
                "N00": 3,
                "N01": 1,
                "N10": 2,
                "N11": 4,
            },
            "probabilities": {
                "P00": 0.75,
                "P01": 0.25,
                "P10": 1 / 3,
                "P11": 2 / 3,
            },
        }

        for n_groups, expected_k in [
            (1, 2),
            (4, 8),
            (12, 24),
        ]:
            with self.subTest(groups=n_groups):
                groups = {
                    str(i): group
                    for i in range(n_groups)
                }

                result = model.evaluate_model(
                    "TEST",
                    groups,
                )

                self.assertEqual(
                    result["parameters"],
                    expected_k,
                )

    def test_invalid_probability_rejected(self):
        counts = {
            "N00": 1,
            "N01": 1,
            "N10": 1,
            "N11": 1,
        }

        probabilities = {
            "P00": 1.2,
            "P01": -0.2,
            "P10": 0.5,
            "P11": 0.5,
        }

        with self.assertRaises(ValueError):
            model.transition_log_likelihood(
                counts, probabilities,
            )

    def test_synthetic_model_comparison(self):
        # Full leap year with both states represented
        # in every monthly and seasonal group.
        start = date(2020, 1, 1)

        dates = [
            start + timedelta(days=i)
            for i in range(366)
        ]

        states = [
            (i // 2) % 2
            for i in range(366)
        ]

        results = model.compare_occurrence_models(
            dates, states,
        )

        self.assertEqual(
            [r["name"] for r in results],
            ["B1", "B2", "B3"],
        )

        self.assertEqual(
            [r["parameters"] for r in results],
            [2, 8, 24],
        )

        for result in results:
            self.assertEqual(result["transitions"], 365)
            self.assertTrue(
                math.isfinite(result["log_likelihood"])
            )
            self.assertTrue(
                math.isfinite(result["aic"])
            )
            self.assertTrue(
                math.isfinite(result["bic"])
            )


if __name__ == "__main__":
    unittest.main()
