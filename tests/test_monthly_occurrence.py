
"""Synthetic unit tests for B3 monthly Markov occurrence."""

import importlib.util
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "rainfall_occurrence",
    ROOT / "src" / "rainfall_occurrence.py",
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def synthetic_year():
    """A full leap year with both previous-day states in every month."""
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(366)]
    states = [(i // 2) % 2 for i in range(366)]
    return dates, states


class MonthlyOccurrenceTests(unittest.TestCase):

    def test_all_twelve_months_present(self):
        dates, states = synthetic_year()
        result = model.fit_b3(dates, states)

        self.assertEqual(set(result), set(range(1, 13)))

    def test_total_transition_count(self):
        dates, states = synthetic_year()
        result = model.fit_b3(dates, states)

        self.assertEqual(
            sum(x["transitions"] for x in result.values()),
            365,
        )

    def test_monthly_calendar_counts(self):
        dates, states = synthetic_year()
        result = model.fit_b3(dates, states)

        expected = {
            1: 30, 2: 29, 3: 31, 4: 30,
            5: 31, 6: 30, 7: 31, 8: 31,
            9: 30, 10: 31, 11: 30, 12: 31,
        }

        for month, count in expected.items():
            with self.subTest(month=month):
                self.assertEqual(
                    result[month]["transitions"],
                    count,
                )

    def test_probabilities_sum_to_one(self):
        dates, states = synthetic_year()
        result = model.fit_b3(dates, states)

        for month, values in result.items():
            with self.subTest(month=month):
                p = values["probabilities"]
                self.assertAlmostEqual(p["P00"] + p["P01"], 1)
                self.assertAlmostEqual(p["P10"] + p["P11"], 1)

    def test_b1_b3_transition_consistency(self):
        dates, states = synthetic_year()
        b1 = model.fit_b1(states)
        b3 = model.fit_b3(dates, states)

        for transition in ("N00", "N01", "N10", "N11"):
            self.assertEqual(
                sum(
                    result["counts"][transition]
                    for result in b3.values()
                ),
                b1["counts"][transition],
            )

    def test_b2_b3_transition_consistency(self):
        dates, states = synthetic_year()
        b2 = model.fit_b2(dates, states)
        b3 = model.fit_b3(dates, states)

        for transition in ("N00", "N01", "N10", "N11"):
            self.assertEqual(
                sum(
                    result["counts"][transition]
                    for result in b2.values()
                ),
                sum(
                    result["counts"][transition]
                    for result in b3.values()
                ),
            )

    def test_nonconsecutive_dates_rejected(self):
        dates, states = synthetic_year()
        dates[10] += timedelta(days=2)

        with self.assertRaises(ValueError):
            model.fit_b3(dates, states)

    def test_mismatched_lengths_rejected(self):
        dates, states = synthetic_year()

        with self.assertRaises(ValueError):
            model.fit_b3(dates, states[:-1])

    def test_invalid_states_rejected(self):
        dates, states = synthetic_year()
        states[5] = 2

        with self.assertRaises(ValueError):
            model.fit_b3(dates, states)

    def test_missing_monthly_state_rejected(self):
        dates, states = synthetic_year()

        # No dry-origin transitions in March.
        for i, day in enumerate(dates):
            if day.month in (2, 3, 4):
                states[i] = 1

        with self.assertRaisesRegex(
            ValueError, "Insufficient transition observations"
        ):
            model.fit_b3(dates, states)

    def test_inputs_not_modified(self):
        dates, states = synthetic_year()
        original_dates = dates.copy()
        original_states = states.copy()

        model.fit_b3(dates, states)

        self.assertEqual(dates, original_dates)
        self.assertEqual(states, original_states)


if __name__ == "__main__":
    unittest.main()
