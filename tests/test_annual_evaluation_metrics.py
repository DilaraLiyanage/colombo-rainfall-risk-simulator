
"""Tests for annual rainfall metrics using synthetic data only."""

import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annual_evaluation_metrics import (
    evaluation_calendar,
    validate_daily_paths,
    calculate_annual_metrics,
)


class AnnualEvaluationMetricsTests(unittest.TestCase):

    def setUp(self):
        self.dates = evaluation_calendar()
        self.paths = np.zeros(
            (2, len(self.dates)),
            dtype=float,
        )

    def test_calendar_length_and_boundaries(self):
        self.assertEqual(len(self.dates), 1826)
        self.assertEqual(self.dates[0], date(2021, 1, 1))
        self.assertEqual(self.dates[-1], date(2025, 12, 31))

    def test_leap_day_present_once(self):
        self.assertEqual(
            self.dates.count(date(2024, 2, 29)),
            1,
        )

    def test_days_per_year(self):
        result = calculate_annual_metrics(
            self.dates,
            self.paths,
        )

        self.assertEqual(
            result["days_per_year"],
            {
                2021: 365,
                2022: 365,
                2023: 365,
                2024: 366,
                2025: 365,
            },
        )

    def test_annual_totals_and_five_year_total(self):
        expected = [10, 20, 30, 40, 50]

        for year, amount in zip(range(2021, 2026), expected):
            index = self.dates.index(date(year, 1, 1))
            self.paths[0, index] = amount

        result = calculate_annual_metrics(
            self.dates,
            self.paths,
        )

        np.testing.assert_array_equal(
            result["annual_total_mm"][0],
            expected,
        )
        self.assertEqual(
            result["five_year_total_mm"][0],
            150.0,
        )

        self.assertAlmostEqual(
            result["annual_total_sd_mm"][0],
            np.std(expected, ddof=1),
        )

    def test_wet_day_threshold_inclusive(self):
        self.paths[0, 0] = 0.999
        self.paths[0, 1] = 1.0
        self.paths[0, 2] = 1.001

        result = calculate_annual_metrics(
            self.dates,
            self.paths,
        )

        self.assertEqual(
            result["annual_wet_days"][0, 0],
            2,
        )
        self.assertEqual(
            result["five_year_wet_days"][0],
            2,
        )

    def test_extreme_thresholds_inclusive(self):
        amounts = [49.999, 50.0, 100.0, 150.0]

        self.paths[0, :4] = amounts

        result = calculate_annual_metrics(
            self.dates,
            self.paths,
        )

        self.assertEqual(
            result["annual_extreme_counts"][50.0][0, 0],
            3,
        )
        self.assertEqual(
            result["annual_extreme_counts"][100.0][0, 0],
            2,
        )
        self.assertEqual(
            result["annual_extreme_counts"][150.0][0, 0],
            1,
        )

    def test_yearly_maximum_and_five_year_maximum(self):
        for year, amount in zip(
            range(2021, 2026),
            [5, 10, 20, 15, 8],
        ):
            index = self.dates.index(date(year, 1, 1))
            self.paths[0, index] = amount

        result = calculate_annual_metrics(
            self.dates,
            self.paths,
        )

        np.testing.assert_array_equal(
            result["annual_maximum_mm"][0],
            [5, 10, 20, 15, 8],
        )
        self.assertEqual(
            result["five_year_maximum_mm"][0],
            20.0,
        )

    def test_zero_rainfall_paths(self):
        result = calculate_annual_metrics(
            self.dates,
            self.paths,
        )

        self.assertTrue(
            np.all(result["five_year_total_mm"] == 0)
        )
        self.assertTrue(
            np.all(result["five_year_wet_days"] == 0)
        )
        self.assertTrue(
            np.all(result["annual_total_sd_mm"] == 0)
        )

    def test_rejects_incomplete_calendar(self):
        with self.assertRaises(ValueError):
            validate_daily_paths(
                self.dates[:-1],
                self.paths[:, :-1],
            )

    def test_rejects_wrong_calendar_order(self):
        altered_dates = self.dates.copy()
        altered_dates[0], altered_dates[1] = (
            altered_dates[1],
            altered_dates[0],
        )

        with self.assertRaises(ValueError):
            validate_daily_paths(
                altered_dates,
                self.paths,
            )

    def test_rejects_wrong_number_of_daily_values(self):
        with self.assertRaises(ValueError):
            validate_daily_paths(
                self.dates,
                self.paths[:, :-1],
            )

    def test_rejects_one_dimensional_input(self):
        with self.assertRaises(ValueError):
            validate_daily_paths(
                self.dates,
                np.zeros(len(self.dates)),
            )

    def test_rejects_negative_rainfall(self):
        self.paths[0, 0] = -0.01

        with self.assertRaises(ValueError):
            validate_daily_paths(
                self.dates,
                self.paths,
            )

    def test_rejects_nonfinite_rainfall(self):
        for invalid in (np.nan, np.inf):
            with self.subTest(invalid=invalid):
                altered = self.paths.copy()
                altered[0, 0] = invalid

                with self.assertRaises(ValueError):
                    validate_daily_paths(
                        self.dates,
                        altered,
                    )

    def test_input_paths_unchanged(self):
        before = self.paths.copy()

        calculate_annual_metrics(
            self.dates,
            self.paths,
        )

        np.testing.assert_array_equal(
            self.paths,
            before,
        )


if __name__ == "__main__":
    unittest.main()
