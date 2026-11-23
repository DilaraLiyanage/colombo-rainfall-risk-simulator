
"""Synthetic tests for the Monte Carlo rainfall simulator.

Tests M1, M2 and M3 without accessing real rainfall datasets.
"""

import importlib.util
import math
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location(
    "monte_carlo_simulator",
    SRC / "monte_carlo_simulator.py",
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def occurrence_group(p01, p11):
    """Create a synthetic fitted Markov group."""
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


class MonteCarloSimulatorTests(unittest.TestCase):

    def test_synthetic_dates(self):
        dates = model.synthetic_dates(
            date(2027, 1, 1), 365
        )

        self.assertEqual(len(dates), 365)
        self.assertEqual(dates[0], date(2027, 1, 1))
        self.assertEqual(dates[-1], date(2027, 12, 31))

        model.validate_simulation_dates(dates)

    def test_leap_year_dates(self):
        dates = model.synthetic_dates(
            date(2028, 1, 1), 366
        )

        self.assertEqual(len(dates), 366)
        self.assertIn(date(2028, 2, 29), dates)
        self.assertEqual(dates[-1], date(2028, 12, 31))

    def test_nonconsecutive_dates_rejected(self):
        dates = [
            date(2027, 1, 1),
            date(2027, 1, 3),
        ]

        with self.assertRaises(ValueError):
            model.validate_simulation_dates(dates)

    def test_b1_transition_probabilities(self):
        b1, _, _, _ = example_parameters()

        self.assertAlmostEqual(
            model.wet_transition_probability(
                "B1", b1, date(2027, 1, 5), 0
            ),
            0.3,
        )

        self.assertAlmostEqual(
            model.wet_transition_probability(
                "B1", b1, date(2027, 1, 5), 1
            ),
            0.9,
        )

    def test_b2_seasonal_routing(self):
        _, b2, _, _ = example_parameters()

        expected = [
            (date(2027, 1, 15), 0.2),
            (date(2027, 3, 15), 0.4),
            (date(2027, 7, 15), 0.6),
            (date(2027, 11, 15), 0.5),
        ]

        for day, probability in expected:
            with self.subTest(day=day):
                actual = model.wet_transition_probability(
                    "B2", b2, day, 0
                )

                self.assertAlmostEqual(
                    actual, probability
                )

    def test_a1_wet_amounts_meet_threshold(self):
        _, _, a1, _ = example_parameters()
        rng = np.random.default_rng(123)

        amounts = [
            model.sample_wet_intensity("A1", a1, rng)
            for _ in range(100)
        ]

        self.assertTrue(
            all(math.isfinite(x) and x >= 1.0 for x in amounts)
        )

    def test_a2_wet_amounts_meet_threshold(self):
        _, _, _, a2 = example_parameters()
        rng = np.random.default_rng(456)

        amounts = [
            model.sample_wet_intensity("A2", a2, rng)
            for _ in range(100)
        ]

        self.assertTrue(
            all(math.isfinite(x) and x >= 1.0 for x in amounts)
        )

    def test_same_seed_produces_same_path(self):
        b1, _, a1, _ = example_parameters()
        dates = model.synthetic_dates(date(2027, 1, 1), 100)

        options = {
            "dates": dates,
            "occurrence_name": "B1",
            "occurrence_parameters": b1,
            "intensity_name": "A1",
            "intensity_parameters": a1,
            "dry_amounts": [0.0, 0.2, 0.8],
            "initial_wet_probability": 0.8,
            "seed": 20261123,
        }

        first = model.simulate_path(**options)
        second = model.simulate_path(**options)

        self.assertEqual(first, second)
        self.assertEqual(len(first["rainfall_mm"]), 100)

    def test_forced_dry_days_preserve_small_amounts(self):
        b1 = {
            "ALL": occurrence_group(0.0, 0.0)
        }
        _, _, a1, _ = example_parameters()

        dates = model.synthetic_dates(date(2027, 1, 1), 30)

        path = model.simulate_path(
            dates=dates,
            occurrence_name="B1",
            occurrence_parameters=b1,
            intensity_name="A1",
            intensity_parameters=a1,
            dry_amounts=[0.25],
            initial_wet_probability=0.0,
            seed=12,
        )

        self.assertEqual(path["states"], [0] * 30)
        self.assertEqual(path["rainfall_mm"], [0.25] * 30)

    def test_forced_wet_days(self):
        b1 = {
            "ALL": occurrence_group(1.0, 1.0)
        }
        _, _, _, a2 = example_parameters()

        dates = model.synthetic_dates(date(2027, 1, 1), 30)

        path = model.simulate_path(
            dates=dates,
            occurrence_name="B1",
            occurrence_parameters=b1,
            intensity_name="A2",
            intensity_parameters=a2,
            dry_amounts=[0.0],
            initial_wet_probability=1.0,
            seed=13,
        )

        self.assertEqual(path["states"], [1] * 30)

        self.assertTrue(
            all(x >= 1.0 for x in path["rainfall_mm"])
        )

    def test_invalid_dry_amounts_rejected(self):
        b1, _, a1, _ = example_parameters()
        dates = model.synthetic_dates(date(2027, 1, 1), 5)

        for invalid in (
            [],
            [-0.1],
            [1.0],
            [float("nan")],
        ):
            with self.subTest(dry_amounts=invalid):
                with self.assertRaises(ValueError):
                    model.simulate_path(
                        dates=dates,
                        occurrence_name="B1",
                        occurrence_parameters=b1,
                        intensity_name="A1",
                        intensity_parameters=a1,
                        dry_amounts=invalid,
                        initial_wet_probability=0.5,
                        seed=1,
                    )

    def test_summary_statistics(self):
        path = {
            "dates": model.synthetic_dates(
                date(2027, 1, 1), 5
            ),
            "states": [0, 1, 1, 1, 0],
            "rainfall_mm": [0.2, 10.0, 55.0, 160.0, 0.5],
        }

        result = model.summarize_path(path)

        self.assertEqual(result["days"], 5)
        self.assertEqual(result["wet_days"], 3)
        self.assertAlmostEqual(result["total_mm"], 225.7)
        self.assertEqual(result["maximum_mm"], 160.0)
        self.assertEqual(result["events_ge_50"], 2)
        self.assertEqual(result["events_ge_100"], 1)
        self.assertEqual(result["events_ge_150"], 1)

    def test_all_three_model_combinations(self):
        b1, b2, a1, a2 = example_parameters()

        occurrence_models = {
            "B1": b1,
            "B2": b2,
        }

        intensity_models = {
            "A1": a1,
            "A2": a2,
        }

        dates = model.synthetic_dates(
            date(2027, 1, 1), 60
        )

        for name, combination in model.MODEL_CONFIGS.items():
            with self.subTest(model=name):
                occurrence_name, intensity_name = combination

                path = model.simulate_path(
                    dates=dates,
                    occurrence_name=occurrence_name,
                    occurrence_parameters=occurrence_models[
                        occurrence_name
                    ],
                    intensity_name=intensity_name,
                    intensity_parameters=intensity_models[
                        intensity_name
                    ],
                    dry_amounts=[0.0, 0.4, 0.8],
                    initial_wet_probability=0.8,
                    seed=12345,
                )

                summary = model.summarize_path(path)

                self.assertEqual(summary["days"], 60)
                self.assertEqual(len(path["states"]), 60)

                for state, rainfall in zip(
                    path["states"], path["rainfall_mm"]
                ):
                    self.assertEqual(
                        state,
                        int(rainfall >= 1.0),
                    )


    def test_summary_rejects_state_rainfall_mismatch(self):
        path = {
            "dates": model.synthetic_dates(
                date(2027, 1, 1), 3
            ),
            "states": [0, 0, 1],
            "rainfall_mm": [0.2, 15.0, 20.0],
        }

        with self.assertRaisesRegex(
            ValueError,
            "Rainfall amounts and occurrence states disagree",
        ):
            model.summarize_path(path)

    def test_summary_rejects_negative_rainfall(self):
        path = {
            "dates": model.synthetic_dates(
                date(2027, 1, 1), 3
            ),
            "states": [0, 0, 1],
            "rainfall_mm": [0.2, -0.5, 20.0],
        }

        with self.assertRaises(ValueError):
            model.summarize_path(path)

    def test_summary_rejects_invalid_states(self):
        path = {
            "dates": model.synthetic_dates(
                date(2027, 1, 1), 3
            ),
            "states": [0, 2, 1],
            "rainfall_mm": [0.2, 0.5, 20.0],
        }

        with self.assertRaisesRegex(
            ValueError,
            "Invalid simulated occurrence states",
        ):
            model.summarize_path(path)


if __name__ == "__main__":
    unittest.main()
