
"""Tests for synthetic pre-holdout simulation safeguards.

No observed 2021-2025 rainfall data is accessed.
"""

import sys
import unittest
import copy
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from holdout_simulation_preflight import (
    evaluation_dates,
    validate_five_year_results,
    EXPECTED_DAYS,
)
from repeated_monte_carlo import (
    N_REPLICATIONS,
    EXTREME_THRESHOLDS,
)


def valid_synthetic_result():
    """Construct internally consistent synthetic summary arrays."""
    n = N_REPLICATIONS

    return {
        "n_replications": n,
        "days_per_path": EXPECTED_DAYS,
        "annual_total_mm": np.full(n, 14000.0),
        "annual_maximum_mm": np.full(n, 120.0),
        "annual_wet_days": np.full(n, 1450),
        "extreme_counts": {
            threshold: np.full(n, max(0, 4 - index))
            for index, threshold in enumerate(
                EXTREME_THRESHOLDS
            )
        },
    }


def valid_annual_synthetic_result():
    """Create internally consistent artificial annual summaries."""
    result = valid_synthetic_result()
    n = N_REPLICATIONS

    result["years"] = (2021, 2022, 2023, 2024, 2025)

    result["yearly_total_mm"] = np.full(
        (n, 5), 2800.0, dtype=float
    )

    result["yearly_wet_days"] = np.full(
        (n, 5), 290, dtype=np.int32
    )

    result["yearly_maximum_mm"] = np.full(
        (n, 5), 120.0, dtype=float
    )

    result["yearly_extreme_counts"] = {
        threshold: np.full(
            (n, 5),
            max(0, 4 - index),
            dtype=np.int32,
        )
        for index, threshold in enumerate(
            EXTREME_THRESHOLDS
        )
    }

    # Reconcile whole-path values with yearly arrays.
    result["annual_total_mm"] = np.sum(
        result["yearly_total_mm"], axis=1
    )

    result["annual_wet_days"] = np.sum(
        result["yearly_wet_days"], axis=1
    )

    result["annual_maximum_mm"] = np.max(
        result["yearly_maximum_mm"], axis=1
    )

    result["extreme_counts"] = {
        threshold: np.sum(counts, axis=1)
        for threshold, counts
        in result["yearly_extreme_counts"].items()
    }

    result["annual_total_sd_mm"] = np.std(
        result["yearly_total_mm"], axis=1, ddof=1
    )

    return result


def assert_invalid_annual_result(test_case, altered_result):
    """Verify that a corrupted annual result is rejected."""
    with test_case.assertRaises((RuntimeError, AssertionError)):
        validate_five_year_results(altered_result)


class HoldoutSimulationPreflightTests(unittest.TestCase):

    def test_evaluation_calendar(self):
        dates = evaluation_dates()

        self.assertEqual(len(dates), 1826)
        self.assertEqual(dates[0], date(2021, 1, 1))
        self.assertEqual(dates[-1], date(2025, 12, 31))
        self.assertEqual(
            dates.count(date(2024, 2, 29)),
            1,
        )

    def test_valid_simulation_result(self):
        result = valid_synthetic_result()

        validate_five_year_results(result)

    def test_rejects_incorrect_dimensions(self):
        result = valid_synthetic_result()
        result["days_per_path"] = 1825

        with self.assertRaises(RuntimeError):
            validate_five_year_results(result)

    def test_rejects_negative_rainfall(self):
        result = valid_synthetic_result()
        result["annual_total_mm"][0] = -10.0

        with self.assertRaises(RuntimeError):
            validate_five_year_results(result)

    def test_rejects_nonfinite_rainfall(self):
        result = valid_synthetic_result()
        result["annual_total_mm"][0] = np.nan

        with self.assertRaises(RuntimeError):
            validate_five_year_results(result)

    def test_rejects_impossible_extreme_counts(self):
        result = valid_synthetic_result()

        first, second = EXTREME_THRESHOLDS[:2]

        result["extreme_counts"][first][0] = 1
        result["extreme_counts"][second][0] = 2

        with self.assertRaises(RuntimeError):
            validate_five_year_results(result)


    def test_valid_annual_results(self):
        result = valid_annual_synthetic_result()
        validate_five_year_results(result)

    def test_rejects_wrong_annual_years(self):
        result = valid_annual_synthetic_result()
        result["years"] = (2021, 2022, 2023, 2024, 2026)
        assert_invalid_annual_result(self, result)

    def test_rejects_annual_total_mismatch(self):
        result = valid_annual_synthetic_result()
        result["yearly_total_mm"][0, 0] += 10.0
        assert_invalid_annual_result(self, result)

    def test_rejects_annual_wet_day_mismatch(self):
        result = valid_annual_synthetic_result()
        result["yearly_wet_days"][0, 0] += 1
        assert_invalid_annual_result(self, result)

    def test_rejects_annual_maximum_mismatch(self):
        result = valid_annual_synthetic_result()
        result["yearly_maximum_mm"][0, :] = 50.0
        assert_invalid_annual_result(self, result)

    def test_rejects_annual_extreme_count_mismatch(self):
        result = valid_annual_synthetic_result()
        threshold = EXTREME_THRESHOLDS[0]

        result["yearly_extreme_counts"][threshold][0, 0] += 1

        assert_invalid_annual_result(self, result)

    def test_rejects_annual_sd_mismatch(self):
        result = valid_annual_synthetic_result()
        result["annual_total_sd_mm"][0] = 999.0
        assert_invalid_annual_result(self, result)

    def test_rejects_negative_annual_total(self):
        result = valid_annual_synthetic_result()
        result["yearly_total_mm"][0, 0] = -1.0
        assert_invalid_annual_result(self, result)

    def test_rejects_nonfinite_annual_total(self):
        result = valid_annual_synthetic_result()
        result["yearly_total_mm"][0, 0] = np.nan
        assert_invalid_annual_result(self, result)


if __name__ == "__main__":
    unittest.main()
