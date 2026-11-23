
"""Synthetic tests for repeated Monte Carlo rainfall simulation.

No calibration or holdout rainfall observations are accessed.
"""

import importlib.util
import math
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import gamma

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location(
    "repeated_monte_carlo",
    SRC / "repeated_monte_carlo.py",
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def occurrence_group(p01, p11):
    return {
        "probabilities": {
            "P00": 1.0 - p01,
            "P01": p01,
            "P10": 1.0 - p11,
            "P11": p11,
        }
    }


def example_parameters():
    b1 = {
        "ALL": occurrence_group(0.3, 0.9)
    }

    b2 = {
        "NE": occurrence_group(0.2, 0.7),
        "FIM": occurrence_group(0.4, 0.8),
        "SW": occurrence_group(0.6, 0.95),
        "SIM": occurrence_group(0.5, 0.9),
    }

    a1 = {
        "shape": 1.5,
        "scale": 8.0,
        "threshold_mm": 1.0,
    }

    a2 = {
        "weight_large": 0.2,
        "scale_small": 5.0,
        "scale_large": 20.0,
        "threshold_mm": 1.0,
    }

    return b1, b2, a1, a2


def run_example(
    occurrence_name="B1",
    intensity_name="A1",
    n=100,
    days=30,
    seed=12345,
    occurrence_parameters=None,
    initial_wet_probability=0.8,
    dry_amounts=None,
):
    b1, b2, a1, a2 = example_parameters()

    occurrences = {"B1": b1, "B2": b2}
    intensities = {"A1": a1, "A2": a2}

    if occurrence_parameters is None:
        occurrence_parameters = occurrences[occurrence_name]

    if dry_amounts is None:
        dry_amounts = [0.0, 0.25, 0.75]

    return model.simulate_many(
        dates=model.synthetic_dates(date(2027, 1, 1), days),
        occurrence_name=occurrence_name,
        occurrence_parameters=occurrence_parameters,
        intensity_name=intensity_name,
        intensity_parameters=intensities[intensity_name],
        dry_amounts=dry_amounts,
        initial_wet_probability=initial_wet_probability,
        n_replications=n,
        seed=seed,
    )


class RepeatedMonteCarloTests(unittest.TestCase):

    def test_all_model_combinations(self):
        for name, (occurrence, intensity) in (
            model.MODEL_CONFIGS.items()
        ):
            with self.subTest(model=name):
                result = run_example(
                    occurrence_name=occurrence,
                    intensity_name=intensity,
                )

                self.assertEqual(result["n_replications"], 100)
                self.assertEqual(result["days_per_path"], 30)

                self.assertEqual(
                    result["annual_total_mm"].shape, (100,)
                )
                self.assertEqual(
                    result["annual_maximum_mm"].shape, (100,)
                )
                self.assertEqual(
                    result["annual_wet_days"].shape, (100,)
                )

                self.assertTrue(
                    np.all(np.isfinite(result["annual_total_mm"]))
                )
                self.assertTrue(
                    np.all(result["annual_total_mm"] >= 0)
                )
                self.assertTrue(
                    np.all(result["annual_maximum_mm"] >= 0)
                )
                self.assertTrue(
                    np.all(result["annual_wet_days"] <= 30)
                )

    def test_reproducibility(self):
        first = run_example(seed=2026111902)
        second = run_example(seed=2026111902)

        for key in (
            "annual_total_mm",
            "annual_maximum_mm",
            "annual_wet_days",
        ):
            np.testing.assert_array_equal(first[key], second[key])

        for threshold in model.EXTREME_THRESHOLDS:
            np.testing.assert_array_equal(
                first["extreme_counts"][threshold],
                second["extreme_counts"][threshold],
            )

    def test_different_seeds_change_results(self):
        first = run_example(seed=1)
        second = run_example(seed=2)

        self.assertFalse(
            np.array_equal(
                first["annual_total_mm"],
                second["annual_total_mm"],
            )
        )

    def test_forced_dry_paths(self):
        occurrence = {
            "ALL": occurrence_group(0.0, 0.0)
        }

        result = run_example(
            occurrence_parameters=occurrence,
            initial_wet_probability=0.0,
            dry_amounts=[0.25],
            n=20,
            days=10,
        )

        np.testing.assert_array_equal(
            result["annual_wet_days"],
            np.zeros(20),
        )

        np.testing.assert_allclose(
            result["annual_total_mm"],
            np.full(20, 2.5),
        )

        np.testing.assert_allclose(
            result["annual_maximum_mm"],
            np.full(20, 0.25),
        )

        for counts in result["extreme_counts"].values():
            self.assertTrue(np.all(counts == 0))

    def test_forced_wet_paths(self):
        occurrence = {
            "ALL": occurrence_group(1.0, 1.0)
        }

        result = run_example(
            occurrence_parameters=occurrence,
            initial_wet_probability=1.0,
            n=20,
            days=10,
        )

        self.assertTrue(
            np.all(result["annual_wet_days"] == 10)
        )

        self.assertTrue(
            np.all(result["annual_total_mm"] >= 10.0)
        )

        self.assertTrue(
            np.all(result["annual_maximum_mm"] >= 1.0)
        )

    def test_extreme_count_ordering(self):
        result = run_example(
            occurrence_name="B2",
            intensity_name="A2",
            n=200,
            days=365,
        )

        counts = result["extreme_counts"]

        self.assertTrue(
            np.all(counts[50.0] >= counts[100.0])
        )
        self.assertTrue(
            np.all(counts[100.0] >= counts[150.0])
        )

        for threshold in model.EXTREME_THRESHOLDS:
            self.assertTrue(
                np.all(
                    counts[threshold]
                    <= result["annual_wet_days"]
                )
            )

    def test_a1_batch_intensity(self):
        _, _, a1, _ = example_parameters()
        rng = np.random.default_rng(123)

        values = model.sample_wet_batch(
            "A1", a1, rng, 1000
        )

        self.assertEqual(len(values), 1000)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 1.0))

        survival = gamma.sf(
            1.0, a=1.5, scale=8.0
        )
        self.assertGreater(survival, 0.0)

    def test_a2_batch_intensity_mean(self):
        _, _, _, a2 = example_parameters()
        rng = np.random.default_rng(456)

        values = model.sample_wet_batch(
            "A2", a2, rng, 20000
        )

        # E[X] = threshold + (1-w)*small + w*large
        expected_mean = (
            1.0 + 0.8 * 5.0 + 0.2 * 20.0
        )

        self.assertEqual(len(values), 20000)
        self.assertTrue(np.all(values >= 1.0))
        self.assertAlmostEqual(
            float(np.mean(values)),
            expected_mean,
            delta=0.4,
        )

    def test_mean_statistics_constant_values(self):
        result = model.mean_statistics(
            [10.0, 10.0, 10.0, 10.0]
        )

        self.assertEqual(result["mean"], 10.0)
        self.assertEqual(result["sd"], 0.0)
        self.assertEqual(result["ci_low"], 10.0)
        self.assertEqual(result["ci_high"], 10.0)
        self.assertEqual(result["predictive_low"], 10.0)
        self.assertEqual(result["predictive_high"], 10.0)

    def test_mean_statistics_known_values(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = model.mean_statistics(values)

        self.assertAlmostEqual(result["mean"], 3.0)
        self.assertAlmostEqual(
            result["sd"], math.sqrt(2.5)
        )
        self.assertLess(result["ci_low"], 3.0)
        self.assertGreater(result["ci_high"], 3.0)
        self.assertLess(
            result["predictive_low"],
            result["predictive_high"],
        )

    def test_event_probability_known_values(self):
        result = model.event_probability_statistics(
            [0, 1, 2, 0, 3]
        )

        self.assertEqual(result["successes"], 3)
        self.assertEqual(result["n"], 5)
        self.assertAlmostEqual(
            result["probability"], 0.6
        )
        self.assertLess(result["ci_low"], 0.6)
        self.assertGreater(result["ci_high"], 0.6)

    def test_wilson_interval_boundary_cases(self):
        for counts, expected_probability in (
            ([0] * 20, 0.0),
            ([1] * 20, 1.0),
        ):
            with self.subTest(p=expected_probability):
                result = model.event_probability_statistics(
                    counts
                )

                self.assertEqual(
                    result["probability"],
                    expected_probability,
                )
                self.assertTrue(
                    0.0 <= result["ci_low"]
                    <= result["ci_high"] <= 1.0
                )

    def test_invalid_replication_count(self):
        b1, _, a1, _ = example_parameters()

        with self.assertRaises(ValueError):
            model.simulate_many(
                dates=model.synthetic_dates(
                    date(2027, 1, 1), 5
                ),
                occurrence_name="B1",
                occurrence_parameters=b1,
                intensity_name="A1",
                intensity_parameters=a1,
                dry_amounts=[0.0, 0.5],
                initial_wet_probability=0.8,
                n_replications=1,
                seed=123,
            )

    def test_invalid_dry_amounts(self):
        for dry in ([], [-0.1], [1.0], [float("nan")]):
            with self.subTest(dry=dry):
                with self.assertRaises(ValueError):
                    run_example(dry_amounts=dry)

    def test_invalid_initial_probability(self):
        for probability in (
            -0.1, 1.1, float("nan")
        ):
            with self.subTest(p=probability):
                with self.assertRaises(ValueError):
                    run_example(
                        initial_wet_probability=probability
                    )

    def test_invalid_summary_inputs(self):
        for values in (
            [],
            [float("nan"), 1.0],
            [float("inf"), 1.0],
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    model.mean_statistics(values)

        for values in (
            [],
            [-1, 0],
            [float("nan"), 0],
        ):
            with self.subTest(events=values):
                with self.assertRaises(ValueError):
                    model.event_probability_statistics(values)


    def test_b1_stationary_wet_frequency(self):
        """Simulated wet frequency should match Markov theory."""
        p01 = 0.3
        p11 = 0.9

        theoretical_wet_probability = (
            p01 / (p01 + 1.0 - p11)
        )

        occurrence = {
            "ALL": occurrence_group(p01, p11)
        }

        result = run_example(
            occurrence_name="B1",
            occurrence_parameters=occurrence,
            initial_wet_probability=theoretical_wet_probability,
            n=4000,
            days=365,
            seed=2026111902,
        )

        simulated_wet_probability = (
            np.mean(result["annual_wet_days"]) / 365.0
        )

        self.assertAlmostEqual(
            simulated_wet_probability,
            theoretical_wet_probability,
            delta=0.015,
        )


    def test_b2_seasonal_wet_frequency(self):
        """Seasonal simulated frequencies match expected probabilities.

        P01 and P11 are equal within each season, making
        the current-day state independent of the previous state.
        This provides a controlled test of seasonal routing.
        """

        season_probabilities = {
            "NE": 0.2,
            "FIM": 0.5,
            "SW": 0.9,
            "SIM": 0.6,
        }

        occurrence = {
            season: occurrence_group(p, p)
            for season, p in season_probabilities.items()
        }

        result = run_example(
            occurrence_name="B2",
            occurrence_parameters=occurrence,
            initial_wet_probability=0.5,
            n=3000,
            days=365,
            seed=2026111903,
        )

        dates = model.synthetic_dates(
            date(2027, 1, 1), 365
        )

        # Day 1 uses the explicitly supplied initial
        # wet probability, not the seasonal transition.
        expected_wet_days = 0.5

        # For subsequent days, P01 = P11 within each season.
        # Therefore, the expected wet-day probability equals
        # the corresponding seasonal transition probability.
        for day in dates[1:]:
            expected_wet_days += (
                model.wet_transition_probability(
                    "B2",
                    occurrence,
                    day,
                    0,
                )
            )

        # Compare the mean count across 3000 simulated years,
        # rather than relying on a single simulated path.
        simulated_wet_days = float(
            np.mean(result["annual_wet_days"])
        )

        self.assertAlmostEqual(
            simulated_wet_days,
            expected_wet_days,
            delta=3.0,
        )



if __name__ == "__main__":
    unittest.main()
