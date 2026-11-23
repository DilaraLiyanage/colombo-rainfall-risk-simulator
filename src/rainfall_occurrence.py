
"""B1: Homogeneous first-order Markov rainfall occurrence model.

Calibration period: 2001-01-01 to 2020-12-31.
Wet-day threshold: rainfall >= 1.0 mm.
No holdout access or output modification.
"""

import csv
import hashlib
import math
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = (
    ROOT / "data/processed/colombo_calibration_2001_2020.csv"
)

EXPECTED_SHA256 = (
    "0fcaa7daca76d1cfbd7543e2b3a3a6cb3aa4509d2c8aab532bf5ddb0b65c89a0"
)

WET_THRESHOLD_MM = 1.0
START = date(2001, 1, 1)
EXPECTED_DAYS = 7305


def load_calibration(path=CALIBRATION_FILE):
    """Load and validate the calibration-only daily series."""
    path = Path(path)

    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Calibration checksum mismatch")

    dates = []
    rainfall = []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != ["date", "rainfall_mm"]:
            raise ValueError("Unexpected calibration CSV columns")

        for row in reader:
            day = date.fromisoformat(row["date"])
            expected = START + timedelta(days=len(dates))

            if day != expected:
                raise ValueError("Unexpected calibration date")

            amount = float(row["rainfall_mm"])

            if not math.isfinite(amount) or amount < 0:
                raise ValueError("Invalid rainfall value")

            dates.append(day)
            rainfall.append(amount)

    if len(dates) != EXPECTED_DAYS:
        raise ValueError("Incorrect calibration record count")

    if dates[-1] != date(2020, 12, 31):
        raise ValueError("Calibration end date mismatch")

    return dates, rainfall


def classify_days(rainfall, threshold=WET_THRESHOLD_MM):
    """Return 0 for dry days and 1 for wet days."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Wet-day threshold must be positive")

    if any(not math.isfinite(x) or x < 0 for x in rainfall):
        raise ValueError("Rainfall must be finite and non-negative")

    return [int(amount >= threshold) for amount in rainfall]


def fit_b1(states):
    """Estimate the four transition probabilities from observed states."""
    if len(states) < 2:
        raise ValueError("At least two daily states are required")

    if any(type(s) is not int or s not in (0, 1) for s in states):
        raise ValueError("States must be integers 0 or 1")

    counts = Counter(zip(states[:-1], states[1:]))

    n00 = counts[(0, 0)]
    n01 = counts[(0, 1)]
    n10 = counts[(1, 0)]
    n11 = counts[(1, 1)]

    from_dry = n00 + n01
    from_wet = n10 + n11

    if from_dry == 0 or from_wet == 0:
        raise ValueError("Both previous-day states must occur")

    probabilities = {
        "P00": n00 / from_dry,
        "P01": n01 / from_dry,
        "P10": n10 / from_wet,
        "P11": n11 / from_wet,
    }

    return {
        "counts": {
            "N00": n00,
            "N01": n01,
            "N10": n10,
            "N11": n11,
        },
        "probabilities": probabilities,
    }


# ------------------------------------------------------------
# B2: Four-season first-order Markov occurrence model
# ------------------------------------------------------------

SEASON_MONTHS = {
    "NE": (12, 1, 2),
    "FIM": (3, 4),
    "SW": (5, 6, 7, 8, 9),
    "SIM": (10, 11),
}


def season_for_date(day):
    """Return the predefined season for a calendar date."""
    if not isinstance(day, date):
        raise ValueError("Expected a datetime.date")

    for season, months in SEASON_MONTHS.items():
        if day.month in months:
            return season

    raise ValueError("Invalid season assignment")


def fit_b2(dates, states):
    """Fit seasonal Markov probabilities using current-day seasons.

    Transition (states[t-1], states[t]) belongs to the season
    of dates[t]. Transitions crossing season boundaries are retained.
    """
    if len(dates) != len(states) or len(states) < 2:
        raise ValueError("Dates and states must have matching lengths >= 2")

    if any(type(s) is not int or s not in (0, 1) for s in states):
        raise ValueError("States must be integers 0 or 1")

    if any(not isinstance(day, date) for day in dates):
        raise ValueError("Dates must be datetime.date objects")

    if any(
        current != previous + timedelta(days=1)
        for previous, current in zip(dates[:-1], dates[1:])
    ):
        raise ValueError("Dates must be consecutive daily observations")

    counts = {
        season: Counter()
        for season in SEASON_MONTHS
    }

    for i in range(1, len(states)):
        season = season_for_date(dates[i])
        transition = (states[i - 1], states[i])
        counts[season][transition] += 1

    results = {}

    for season in SEASON_MONTHS:
        c = counts[season]

        n00 = c[(0, 0)]
        n01 = c[(0, 1)]
        n10 = c[(1, 0)]
        n11 = c[(1, 1)]

        from_dry = n00 + n01
        from_wet = n10 + n11

        if from_dry == 0 or from_wet == 0:
            raise ValueError(
                f"Insufficient transition observations in {season}"
            )

        results[season] = {
            "counts": {
                "N00": n00,
                "N01": n01,
                "N10": n10,
                "N11": n11,
            },
            "probabilities": {
                "P00": n00 / from_dry,
                "P01": n01 / from_dry,
                "P10": n10 / from_wet,
                "P11": n11 / from_wet,
            },
            "transitions": sum(c.values()),
        }

    total_transitions = sum(
        result["transitions"]
        for result in results.values()
    )

    assert total_transitions == len(states) - 1

    return results


def print_b2_results(dates, states):
    """Display B2 fitted transition probabilities."""
    results = fit_b2(dates, states)

    print("\nMODEL: B2 — Four-Season First-Order Markov Chain")

    for season, result in results.items():
        print(f"\nSeason: {season}")
        print("Transitions:", result["transitions"])

        for key, value in result["counts"].items():
            print(f"  {key}: {value}")

        for key, value in result["probabilities"].items():
            print(f"  {key}: {value:.10f}")

    print("\nPASS: B2 seasonal Markov model fitted successfully")


# ------------------------------------------------------------
# B3: Monthly first-order Markov occurrence model
# ------------------------------------------------------------

def fit_b3(dates, states):
    """Fit 12 monthly Markov transition matrices.

    Each transition is assigned to the month of the current day.
    All adjacent calendar-day transitions are retained.

    No smoothing is applied. Missing state denominators cause
    explicit failure instead of silently inventing probabilities.
    """
    if len(dates) != len(states) or len(states) < 2:
        raise ValueError(
            "Dates and states must have matching lengths >= 2"
        )

    if any(type(s) is not int or s not in (0, 1) for s in states):
        raise ValueError("States must be integers 0 or 1")

    if any(type(day) is not date for day in dates):
        raise ValueError("Dates must be datetime.date objects")

    if any(
        current != previous + timedelta(days=1)
        for previous, current in zip(dates[:-1], dates[1:])
    ):
        raise ValueError("Dates must be consecutive daily observations")

    counts = {month: Counter() for month in range(1, 13)}

    for i in range(1, len(states)):
        month = dates[i].month
        transition = (states[i - 1], states[i])
        counts[month][transition] += 1

    results = {}

    for month in range(1, 13):
        c = counts[month]

        n00 = c[(0, 0)]
        n01 = c[(0, 1)]
        n10 = c[(1, 0)]
        n11 = c[(1, 1)]

        dry_total = n00 + n01
        wet_total = n10 + n11

        if dry_total == 0 or wet_total == 0:
            raise ValueError(
                f"Insufficient transition observations in month {month}"
            )

        results[month] = {
            "counts": {
                "N00": n00,
                "N01": n01,
                "N10": n10,
                "N11": n11,
            },
            "probabilities": {
                "P00": n00 / dry_total,
                "P01": n01 / dry_total,
                "P10": n10 / wet_total,
                "P11": n11 / wet_total,
            },
            "transitions": sum(c.values()),
        }

    assert sum(
        result["transitions"] for result in results.values()
    ) == len(states) - 1

    return results


def print_b3_results(dates, states):
    """Display the 12 monthly Markov probability estimates."""
    results = fit_b3(dates, states)

    print("\nMODEL: B3 — Monthly First-Order Markov Chain")

    print(
        f"{'Month':<7} {'N':>6} {'P01':>12} {'P11':>12}"
    )

    for month, result in results.items():
        p = result["probabilities"]

        print(
            f"{month:<7} "
            f"{result['transitions']:>6} "
            f"{p['P01']:>12.10f} "
            f"{p['P11']:>12.10f}"
        )

    print("\nPASS: B3 monthly Markov model fitted successfully")


def main():
    dates, rainfall = load_calibration()
    states = classify_days(rainfall)
    result = fit_b1(states)

    print("MODEL: B1 — Homogeneous First-Order Markov Chain")
    print("Calibration:", dates[0], "to", dates[-1])
    print("Wet threshold:", WET_THRESHOLD_MM, "mm")
    print("Total days:", len(states))
    print("Dry days:", states.count(0))
    print("Wet days:", states.count(1))

    print("\nTransition counts:")
    for key, value in result["counts"].items():
        print(f"{key}: {value}")

    print("\nTransition probabilities:")
    for key, value in result["probabilities"].items():
        print(f"{key}: {value:.10f}")

    assert sum(result["counts"].values()) == len(states) - 1

    p = result["probabilities"]
    assert math.isclose(p["P00"] + p["P01"], 1.0)
    assert math.isclose(p["P10"] + p["P11"], 1.0)

    print("\nPASS: B1 Markov model fitted successfully")
    print_b2_results(dates, states)
    print_b3_results(dates, states)


if __name__ == "__main__":
    main()