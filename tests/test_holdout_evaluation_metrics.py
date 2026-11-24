
"""Tests for holdout evaluation metrics using artificial values only."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from holdout_evaluation_metrics import (
    validate_simulations,
    predictive_summary,
    compare_observed,
    compare_event_count,
)


class HoldoutEvaluationMetricsTests(unittest.TestCase):

    def test_valid_simulations(self):
        values = validate_simulations([0.0, 5.0, 10.0])
        np.testing.assert_array_equal(values, [0.0, 5.0, 10.0])

    def test_rejects_empty_simulations(self):
        with self.assertRaises(ValueError):
            validate_simulations([])

    def test_rejects_single_simulation(self):
        with self.assertRaises(ValueError):
            validate_simulations([10.0])

    def test_rejects_multidimensional_simulations(self):
        with self.assertRaises(ValueError):
            validate_simulations([[1.0, 2.0], [3.0, 4.0]])

    def test_rejects_nonfinite_simulations(self):
        for values in ([1.0, np.nan], [1.0, np.inf]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_simulations(values)

    def test_rejects_negative_simulations(self):
        with self.assertRaises(ValueError):
            validate_simulations([1.0, -0.1])

    def test_predictive_summary(self):
        result = predictive_summary([0.0, 10.0, 20.0, 30.0, 40.0])

        self.assertEqual(result["n_simulations"], 5)
        self.assertAlmostEqual(result["mean"], 20.0)
        self.assertAlmostEqual(result["median"], 20.0)
        self.assertAlmostEqual(result["predictive_low"], 1.0)
        self.assertAlmostEqual(result["predictive_high"], 39.0)
        self.assertEqual(result["minimum"], 0.0)
        self.assertEqual(result["maximum"], 40.0)

    def test_observation_inside_interval(self):
        result = compare_observed(
            20.0,
            [0.0, 10.0, 20.0, 30.0, 40.0],
        )

        self.assertEqual(result["predictive_position"], "inside")
        self.assertAlmostEqual(result["absolute_error"], 0.0)
        self.assertAlmostEqual(result["percentile_midrank"], 0.5)

    def test_observation_below_interval(self):
        result = compare_observed(
            0.0,
            [10.0, 20.0, 30.0, 40.0, 50.0],
        )
        self.assertEqual(result["predictive_position"], "below")
        self.assertAlmostEqual(result["percentile_midrank"], 0.0)

    def test_observation_above_interval(self):
        result = compare_observed(
            100.0,
            [10.0, 20.0, 30.0, 40.0, 50.0],
        )
        self.assertEqual(result["predictive_position"], "above")
        self.assertAlmostEqual(result["percentile_midrank"], 1.0)

    def test_observation_on_interval_boundary(self):
        values = [0.0, 10.0, 20.0, 30.0, 40.0]
        lower = float(np.quantile(values, 0.025))
        upper = float(np.quantile(values, 0.975))

        for observed in (lower, upper):
            with self.subTest(observed=observed):
                result = compare_observed(observed, values)
                self.assertEqual(
                    result["predictive_position"],
                    "inside",
                )

    def test_tied_discrete_values(self):
        result = compare_observed(
            1.0,
            [0.0, 1.0, 1.0, 1.0, 2.0],
        )

        self.assertAlmostEqual(
            result["fraction_simulations_below"],
            0.2,
        )
        self.assertAlmostEqual(
            result["fraction_simulations_equal"],
            0.6,
        )
        self.assertAlmostEqual(
            result["fraction_simulations_above"],
            0.2,
        )
        self.assertAlmostEqual(
            result["percentile_midrank"],
            0.5,
        )

    def test_rejects_invalid_observation(self):
        for value in (-1.0, np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    compare_observed(value, [1.0, 2.0, 3.0])

    def test_input_values_not_modified(self):
        original = np.array([10.0, 20.0, 30.0])
        before = original.copy()

        compare_observed(25.0, original)

        np.testing.assert_array_equal(original, before)


    def test_event_count_probabilities(self):
        result = compare_event_count(
            2,
            [0, 0, 1, 2, 2, 3, 4, 0, 1, 2],
        )

        self.assertEqual(result["observed_count"], 2)
        self.assertAlmostEqual(result["mean_simulated_count"], 1.5)
        self.assertAlmostEqual(result["probability_zero_events"], 0.3)
        self.assertAlmostEqual(result["probability_at_least_one"], 0.7)
        self.assertAlmostEqual(
            result["probability_count_at_least_observed"], 0.5
        )
        self.assertAlmostEqual(
            result["probability_count_greater_than_observed"], 0.2
        )
        self.assertAlmostEqual(
            result["probability_exact_observed_count"], 0.3
        )

    def test_all_zero_event_counts(self):
        result = compare_event_count(0, [0, 0, 0, 0])

        self.assertEqual(result["mean_simulated_count"], 0.0)
        self.assertEqual(result["probability_zero_events"], 1.0)
        self.assertEqual(result["probability_at_least_one"], 0.0)
        self.assertEqual(
            result["probability_count_at_least_observed"], 1.0
        )
        self.assertEqual(result["predictive_position"], "inside")

    def test_observation_exceeds_all_simulations(self):
        result = compare_event_count(10, [0, 1, 1, 2, 3])

        self.assertEqual(result["predictive_position"], "above")
        self.assertEqual(
            result["probability_count_at_least_observed"], 0.0
        )
        self.assertEqual(result["percentile_midrank"], 1.0)

    def test_rejects_negative_event_counts(self):
        with self.assertRaises(ValueError):
            compare_event_count(-1, [0, 1, 2])

        with self.assertRaises(ValueError):
            compare_event_count(1, [0, -1, 2])

    def test_rejects_fractional_event_counts(self):
        with self.assertRaises(ValueError):
            compare_event_count(1.5, [0, 1, 2])

        with self.assertRaises(ValueError):
            compare_event_count(1, [0, 1.5, 2])

    def test_rejects_nonfinite_event_counts(self):
        for observed in (np.nan, np.inf):
            with self.subTest(observed=observed):
                with self.assertRaises(ValueError):
                    compare_event_count(observed, [0, 1, 2])

        with self.assertRaises(ValueError):
            compare_event_count(1, [0, np.nan, 2])

    def test_event_count_input_unchanged(self):
        original = np.array([0, 1, 2, 3])
        before = original.copy()

        compare_event_count(2, original)

        np.testing.assert_array_equal(original, before)


    def test_discrete_interval_has_integer_bounds(self):
        result = compare_event_count(
            2,
            [0, 0, 1, 2, 2, 3, 4, 0, 1, 2],
        )

        self.assertEqual(result["predictive_low"], 0)
        self.assertEqual(result["predictive_high"], 4)

        self.assertIsInstance(result["predictive_low"], int)
        self.assertIsInstance(result["predictive_high"], int)

    def test_discrete_interval_all_zero(self):
        result = compare_event_count(
            0,
            [0, 0, 0, 0, 0],
        )

        self.assertEqual(result["predictive_low"], 0)
        self.assertEqual(result["predictive_high"], 0)
        self.assertEqual(
            result["predictive_position"],
            "inside",
        )

    def test_discrete_interval_observation_above(self):
        result = compare_event_count(
            5,
            [0, 0, 1, 1, 2],
        )

        self.assertEqual(result["predictive_low"], 0)
        self.assertEqual(result["predictive_high"], 2)
        self.assertEqual(
            result["predictive_position"],
            "above",
        )

    def test_discrete_interval_observation_below(self):
        result = compare_event_count(
            0,
            [1, 1, 2, 2, 3],
        )

        self.assertEqual(result["predictive_low"], 1)
        self.assertEqual(result["predictive_high"], 3)
        self.assertEqual(
            result["predictive_position"],
            "below",
        )




if __name__ == "__main__":
    unittest.main()
