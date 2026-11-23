
"""Synthetic tests for B2 seasonal Markov rainfall occurrence."""

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


class SeasonalOccurrenceTests(unittest.TestCase):

    def test_season_assignments(self):
        expected = {
            1: "NE", 2: "NE", 3: "FIM", 4: "FIM",
            5: "SW", 6: "SW", 7: "SW", 8: "SW",
            9: "SW", 10: "SIM", 11: "SIM", 12: "NE",
        }

        for month, season in expected.items():
            with self.subTest(month=month):
                self.assertEqual(
                    model.season_for_date(date(2020, month, 15)),
                    season,
                )

    def test_invalid_season_date(self):
        with self.assertRaises(ValueError):
            model.season_for_date("2020-01-01")

    def test_transition_boundary_assignment(self):
        # March 1 belongs to FIM, even if February 29 was NE.
        dates = [
            date(2020, 2, 29),
            date(2020, 3, 1),
        ]

        self.assertEqual(model.season_for_date(dates[0]), "NE")
        self.assertEqual(model.season_for_date(dates[1]), "FIM")

    def test_synthetic_seasonal_transitions(self):
        # Construct a continuous synthetic year with both
        # previous-day states represented within each season.
        start = date(2020, 1, 1)
        dates = [start + timedelta(days=i) for i in range(366)]

        states = [i % 4 // 2 for i in range(366)]

        results = model.fit_b2(dates, states)

        self.assertEqual(set(results), {"NE", "FIM", "SW", "SIM"})

        total = sum(
            item["transitions"] for item in results.values()
        )
        self.assertEqual(total, 365)

        for season, result in results.items():
            with self.subTest(season=season):
                p = result["probabilities"]
                self.assertAlmostEqual(p["P00"] + p["P01"], 1)
                self.assertAlmostEqual(p["P10"] + p["P11"], 1)

    def test_b1_b2_count_consistency(self):
        start = date(2020, 1, 1)
        dates = [start + timedelta(days=i) for i in range(366)]
        states = [i % 4 // 2 for i in range(366)]

        b1 = model.fit_b1(states)
        b2 = model.fit_b2(dates, states)

        for transition in ("N00", "N01", "N10", "N11"):
            seasonal_total = sum(
                result["counts"][transition]
                for result in b2.values()
            )
            self.assertEqual(
                seasonal_total,
                b1["counts"][transition],
            )

    def test_nonconsecutive_dates_rejected(self):
        dates = [
            date(2020, 1, 1),
            date(2020, 1, 3),
            date(2020, 1, 4),
        ]

        with self.assertRaises(ValueError):
            model.fit_b2(dates, [0, 1, 0])

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            model.fit_b2(
                [date(2020, 1, 1), date(2020, 1, 2)],
                [0],
            )

    def test_invalid_states_rejected(self):
        dates = [
            date(2020, 1, 1) + timedelta(days=i)
            for i in range(3)
        ]

        with self.assertRaises(ValueError):
            model.fit_b2(dates, [0, 2, 1])

    def test_insufficient_seasonal_observations_rejected(self):
        dates = [
            date(2020, 1, 1) + timedelta(days=i)
            for i in range(10)
        ]

        with self.assertRaises(ValueError):
            model.fit_b2(dates, [i % 2 for i in range(10)])


if __name__ == "__main__":
    unittest.main()
