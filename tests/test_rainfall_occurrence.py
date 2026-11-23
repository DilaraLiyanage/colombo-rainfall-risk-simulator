
"""Unit tests for first-order Markov rainfall occurrence."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "rainfall_occurrence",
    ROOT / "src" / "rainfall_occurrence.py",
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class RainfallOccurrenceTests(unittest.TestCase):

    def test_wet_dry_classification(self):
        rainfall = [0.0, 0.5, 0.999, 1.0, 5.0, 20.0]
        expected = [0, 0, 0, 1, 1, 1]

        self.assertEqual(
            model.classify_days(rainfall),
            expected,
        )

    def test_known_transition_counts(self):
        # Transitions: 00, 01, 11, 10, 01
        states = [0, 0, 1, 1, 0, 1]

        result = model.fit_b1(states)

        self.assertEqual(
            result["counts"],
            {
                "N00": 1,
                "N01": 2,
                "N10": 1,
                "N11": 1,
            },
        )

    def test_known_transition_probabilities(self):
        states = [0, 0, 1, 1, 0, 1]

        p = model.fit_b1(states)["probabilities"]

        self.assertAlmostEqual(p["P00"], 1 / 3)
        self.assertAlmostEqual(p["P01"], 2 / 3)
        self.assertAlmostEqual(p["P10"], 1 / 2)
        self.assertAlmostEqual(p["P11"], 1 / 2)

    def test_transition_rows_sum_to_one(self):
        states = [0, 0, 1, 1, 0, 1]

        p = model.fit_b1(states)["probabilities"]

        self.assertAlmostEqual(p["P00"] + p["P01"], 1.0)
        self.assertAlmostEqual(p["P10"] + p["P11"], 1.0)

    def test_insufficient_states_rejected(self):
        with self.assertRaises(ValueError):
            model.fit_b1([0])

    def test_invalid_states_rejected(self):
        invalid_cases = [
            [0, 2, 1],
            [0, -1, 1],
            [0, 0.5, 1],
            [0, True, 1],
        ]

        for states in invalid_cases:
            with self.subTest(states=states):
                with self.assertRaises(ValueError):
                    model.fit_b1(states)

    def test_missing_previous_state_rejected(self):
        for states in ([0, 0, 0], [1, 1, 1]):
            with self.subTest(states=states):
                with self.assertRaises(ValueError):
                    model.fit_b1(states)

    def test_invalid_rainfall_rejected(self):
        for rainfall in ([1.0, -0.1], [1.0, float("nan")],
                         [1.0, float("inf")]):
            with self.subTest(rainfall=rainfall):
                with self.assertRaises(ValueError):
                    model.classify_days(rainfall)

    def test_invalid_threshold_rejected(self):
        for threshold in (0, -1, float("nan")):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    model.classify_days([0.5, 1.0], threshold)

    def test_input_not_modified(self):
        states = [0, 0, 1, 1, 0, 1]
        original = states.copy()

        model.fit_b1(states)

        self.assertEqual(states, original)


if __name__ == "__main__":
    unittest.main()