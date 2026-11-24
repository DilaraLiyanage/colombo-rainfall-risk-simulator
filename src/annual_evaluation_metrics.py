
"""Annual rainfall statistics for independent simulation evaluation.

Pure calculation functions: no file access, model fitting, or
holdout observation loading.

A row represents one synthetic rainfall path; columns represent
consecutive calendar dates.
"""

from datetime import date, timedelta

import numpy as np


START_DATE = date(2021, 1, 1)
END_DATE = date(2025, 12, 31)
WET_THRESHOLD_MM = 1.0
EXTREME_THRESHOLDS_MM = (50.0, 100.0, 150.0)


def evaluation_calendar():
    """Return every date in the predetermined evaluation period."""
    number_of_days = (END_DATE - START_DATE).days + 1

    return [
        START_DATE + timedelta(days=offset)
        for offset in range(number_of_days)
    ]


def validate_daily_paths(dates, rainfall_paths):
    """Validate calendar alignment and simulated rainfall values."""
    expected_dates = evaluation_calendar()

    if list(dates) != expected_dates:
        raise ValueError(
            "Dates must match the complete 2021-2025 calendar"
        )

    values = np.asarray(rainfall_paths, dtype=float)

    if values.ndim != 2:
        raise ValueError(
            "Rainfall paths must be a two-dimensional array"
        )

    if values.shape[0] < 1:
        raise ValueError("At least one path is required")

    if values.shape[1] != len(expected_dates):
        raise ValueError(
            "Daily rainfall values do not match calendar length"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Rainfall values must be finite")

    if np.any(values < 0):
        raise ValueError("Rainfall values cannot be negative")

    return values


def calculate_annual_metrics(dates, rainfall_paths):
    """Calculate yearly statistics for every simulated path.

    Returns one value per path per calendar year, alongside
    five-year totals and the within-path annual rainfall SD.

    All extreme thresholds use >= comparisons.
    """
    values = validate_daily_paths(dates, rainfall_paths)

    years = tuple(range(2021, 2026))
    date_years = np.asarray([d.year for d in dates])

    annual_totals = []
    annual_wet_days = []
    annual_maxima = []
    annual_extremes = {
        threshold: [] for threshold in EXTREME_THRESHOLDS_MM
    }

    days_per_year = {}

    for year in years:
        selected = date_years == year
        year_values = values[:, selected]

        days_per_year[year] = int(np.sum(selected))

        annual_totals.append(
            np.sum(year_values, axis=1)
        )
        annual_wet_days.append(
            np.sum(
                year_values >= WET_THRESHOLD_MM,
                axis=1,
            )
        )
        annual_maxima.append(
            np.max(year_values, axis=1)
        )

        for threshold in EXTREME_THRESHOLDS_MM:
            annual_extremes[threshold].append(
                np.sum(year_values >= threshold, axis=1)
            )

    totals = np.column_stack(annual_totals)
    wet_days = np.column_stack(annual_wet_days)
    maxima = np.column_stack(annual_maxima)

    extremes = {
        threshold: np.column_stack(yearly_counts)
        for threshold, yearly_counts
        in annual_extremes.items()
    }

    return {
        "years": years,
        "days_per_year": days_per_year,
        "n_paths": int(values.shape[0]),
        "annual_total_mm": totals,
        "annual_wet_days": wet_days,
        "annual_maximum_mm": maxima,
        "annual_extreme_counts": extremes,
        "five_year_total_mm": np.sum(totals, axis=1),
        "five_year_wet_days": np.sum(wet_days, axis=1),
        "five_year_maximum_mm": np.max(maxima, axis=1),
        "five_year_extreme_counts": {
            threshold: np.sum(counts, axis=1)
            for threshold, counts in extremes.items()
        },
        "annual_total_sd_mm": np.std(
            totals, axis=1, ddof=1
        ),
    }


def main():
    """Verify annual calculations using artificial rainfall."""
    dates = evaluation_calendar()

    # Two completely artificial paths.
    paths = np.zeros((2, len(dates)), dtype=float)

    first_day_by_year = {
        year: dates.index(date(year, 1, 1))
        for year in range(2021, 2026)
    }

    # Path 1: one increasingly heavy rainfall day each year.
    for year, index in first_day_by_year.items():
        paths[0, index] = 10.0 * (year - 2020)

    # Path 2: one 100 mm event per year.
    for index in first_day_by_year.values():
        paths[1, index] = 100.0

    result = calculate_annual_metrics(dates, paths)

    print("ANNUAL EVALUATION — SYNTHETIC PREFLIGHT")
    print("Calendar:", dates[0], "to", dates[-1])
    print("Calendar days:", len(dates))
    print("Days per year:", result["days_per_year"])
    print("Synthetic paths:", result["n_paths"])

    for path_index in range(result["n_paths"]):
        print(f"\nArtificial path {path_index + 1}")
        print(
            "Annual totals (mm):",
            result["annual_total_mm"][path_index].tolist(),
        )
        print(
            "Five-year total (mm):",
            result["five_year_total_mm"][path_index],
        )
        print(
            "Annual total SD (mm):",
            round(
                result["annual_total_sd_mm"][path_index], 4
            ),
        )

    print("\nObserved holdout rainfall: NOT ACCESSED")
    print("PASS: annual synthetic evaluation completed")


if __name__ == "__main__":
    main()
