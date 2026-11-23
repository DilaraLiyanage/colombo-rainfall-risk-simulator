
"""Tests for chronological Markov-model validation.

Synthetic observations only; no real rainfall files are accessed.
"""

import importlib.util
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location(
    "chronological_validation",
    SRC / "chronological_validation.py",
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class ChronologicalValidationTests(unittest.TestCase):

    def test_smoothed_probability(self):
        counts = {
            "N00": 3,
            "N01": 1,
            "N10": 2,
            "N11": 4,
        }

        self.assertAlmostEqual(
            model.smoothed_wet_probability(counts, 0),
            1.5 / 5,
        )

        self.assertAlmostEqual(
            model.smoothed_wet_probability(counts, 1),
            4.5 / 7,
        )

    def test_zero_observations_smoothed(self):
        counts = {
            "N00": 0,
            "N01": 0,
            "N10": 0,
            "N11": 0,
        }

        self.assertEqual(
            model.smoothed_wet_probability(counts, 0),
            0.5,
        )

    def test_known_log_loss_and_brier(self):
        result = model.score_predictions(
            [0, 1],
            [0.5, 0.5],
        )

        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(
            result["log_loss"],
            math.log(2),
        )
        self.assertAlmostEqual(
            result["brier_score"],
            0.25,
        )
        self.assertAlmostEqual(
            result["accuracy"],
            0.5,
        )

    def test_perfect_direction_predictions(self):
        result = model.score_predictions(
            [0, 1],
            [0.1, 0.9],
        )

        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["brier_score"], 0.01)
        self.assertAlmostEqual(
            result["log_loss"],
            -math.log(0.9),
        )

    def test_invalid_predictions_rejected(self):
        cases = [
            ([0, 1], [0.5]),
            ([], []),
            ([0], [0.0]),
            ([1], [1.0]),
            ([1], [float("nan")]),
            ([2], [0.5]),
        ]

        for actual, predicted in cases:
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    model.score_predictions(actual, predicted)

    def test_model_probability_routing(self):
        counts = {
            "N00": 3,
            "N01": 1,
            "N10": 2,
            "N11": 4,
        }
        fitted_group = {"counts": counts}

        expected = 1.5 / 5

        self.assertAlmostEqual(
            model.predict_probability(
                "B1", {"ALL": fitted_group},
                date(2020, 3, 1), 0
            ),
            expected,
        )

        self.assertAlmostEqual(
            model.predict_probability(
                "B2", {"FIM": fitted_group},
                date(2020, 3, 1), 0
            ),
            expected,
        )

        self.assertAlmostEqual(
            model.predict_probability(
                "B3", {3: fitted_group},
                date(2020, 3, 1), 0
            ),
            expected,
        )

    def test_synthetic_chronological_split(self):
        start = date(2001, 1, 1)

        dates = [
            start + timedelta(days=i)
            for i in range(7305)
        ]

        states = [
            (i // 2) % 2
            for i in range(7305)
        ]

        results = model.chronological_validate(dates, states)

        self.assertEqual(set(results), {"B1", "B2", "B3"})

        for result in results.values():
            self.assertEqual(result["n"], 1827)
            self.assertTrue(math.isfinite(result["log_loss"]))
            self.assertTrue(math.isfinite(result["brier_score"]))

    def test_incorrect_split_rejected(self):
        start = date(2001, 1, 1)

        dates = [
            start + timedelta(days=i)
            for i in range(7304)
        ]
        states = [(i // 2) % 2 for i in range(7304)]

        with self.assertRaises(ValueError):
            model.chronological_validate(dates, states)


    def test_duplicate_date_rejected(self):
        start = date(2001, 1, 1)
        dates = [
            start + timedelta(days=i)
            for i in range(7305)
        ]
        states = [(i // 2) % 2 for i in range(7305)]

        dates[100] = dates[99]

        with self.assertRaisesRegex(
            ValueError, "Nonconsecutive calibration dates"
        ):
            model.chronological_validate(dates, states)

    def test_invalid_calibration_state_rejected(self):
        start = date(2001, 1, 1)
        dates = [
            start + timedelta(days=i)
            for i in range(7305)
        ]
        states = [(i // 2) % 2 for i in range(7305)]

        states[100] = 2

        with self.assertRaisesRegex(
            ValueError, "Invalid rainfall occurrence state"
        ):
            model.chronological_validate(dates, states)



if __name__ == "__main__":
    unittest.main()
