
"""Monte Carlo daily rainfall simulator: M1, M2 and M3.

M1 = B1 homogeneous occurrence + A1 truncated Gamma intensity
M2 = B2 seasonal occurrence + A1 truncated Gamma intensity
M3 = B2 seasonal occurrence + A2 shifted mixture intensity

Fits using 2001-2020 calibration data only.
Simulates an illustrative synthetic year (2027).
Does not access the 2021-2025 holdout.
"""

from datetime import date, timedelta
import math

import numpy as np
from scipy.stats import gamma

from rainfall_occurrence import (
    load_calibration,
    classify_days,
    fit_b1,
    fit_b2,
    season_for_date,
)
from rainfall_intensity import (
    fit_truncated_gamma,
    fit_mixed_exponential,
)

WET_THRESHOLD_MM = 1.0

MODEL_CONFIGS = {
    "M1": ("B1", "A1"),
    "M2": ("B2", "A1"),
    "M3": ("B2", "A2"),
}


def synthetic_dates(start, number_of_days):
    """Generate consecutive dates for a synthetic experiment."""
    if type(start) is not date:
        raise ValueError("Start must be a date")

    if type(number_of_days) is not int or number_of_days < 1:
        raise ValueError("Number of days must be positive")

    return [
        start + timedelta(days=i)
        for i in range(number_of_days)
    ]


def validate_simulation_dates(dates):
    """Ensure the simulation has a valid consecutive date sequence."""
    if not dates or any(type(d) is not date for d in dates):
        raise ValueError("Invalid simulation dates")

    for previous, current in zip(dates[:-1], dates[1:]):
        if current != previous + timedelta(days=1):
            raise ValueError("Simulation dates must be consecutive")


def wet_transition_probability(model_name, fitted, day, previous_state):
    """Return P(current day wet | previous day's simulated state)."""
    if type(previous_state) is not int or previous_state not in (0, 1):
        raise ValueError("Previous state must be 0 or 1")

    if model_name == "B1":
        group = fitted["ALL"]
    elif model_name == "B2":
        group = fitted[season_for_date(day)]
    else:
        raise ValueError("Unknown occurrence model")

    probability = (
        group["probabilities"]["P01"]
        if previous_state == 0
        else group["probabilities"]["P11"]
    )

    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid transition probability")

    return float(probability)


def sample_wet_intensity(intensity_name, parameters, rng):
    """Draw one wet-day amount from A1 or A2."""
    if intensity_name == "A1":
        threshold = parameters["threshold_mm"]
        shape = parameters["shape"]
        scale = parameters["scale"]

        survival_at_threshold = gamma.sf(
            threshold, a=shape, scale=scale
        )

        if (
            shape <= 0
            or scale <= 0
            or not math.isfinite(survival_at_threshold)
            or survival_at_threshold <= 0
        ):
            raise ValueError("Invalid A1 parameters")

        # Inverse-survival sampling from the truncated Gamma.
        u = max(float(rng.random()), np.finfo(float).tiny)

        amount = float(
            gamma.isf(
                u * survival_at_threshold,
                a=shape,
                scale=scale,
            )
        )

    elif intensity_name == "A2":
        threshold = parameters["threshold_mm"]
        weight = parameters["weight_large"]
        small = parameters["scale_small"]
        large = parameters["scale_large"]

        if (
            not all(
                math.isfinite(x)
                for x in (threshold, weight, small, large)
            )
            or not 0 < weight < 1
            or not 0 < small < large
        ):
            raise ValueError("Invalid A2 parameters")

        selected_scale = (
            large if rng.random() < weight else small
        )

        amount = float(
            threshold + rng.exponential(selected_scale)
        )

    else:
        raise ValueError("Unknown intensity model")

    if not math.isfinite(amount) or amount < WET_THRESHOLD_MM:
        raise RuntimeError("Invalid simulated wet-day rainfall")

    return amount


def simulate_path(
    dates,
    occurrence_name,
    occurrence_parameters,
    intensity_name,
    intensity_parameters,
    dry_amounts,
    initial_wet_probability,
    seed,
):
    """Generate one entirely synthetic daily rainfall sequence.

    Each day's occurrence depends on the previous *simulated*
    state, never an observed future-day state.
    """
    validate_simulation_dates(dates)

    dry = np.asarray(dry_amounts, dtype=float)

    if (
        dry.ndim != 1
        or dry.size == 0
        or not np.all(np.isfinite(dry))
        or np.any(dry < 0)
        or np.any(dry >= WET_THRESHOLD_MM)
    ):
        raise ValueError("Invalid calibration dry-day amounts")

    if (
        not math.isfinite(initial_wet_probability)
        or not 0 <= initial_wet_probability <= 1
    ):
        raise ValueError("Invalid initial wet-day probability")

    rng = np.random.default_rng(seed)

    # Initial state is sampled from the calibration wet-day
    # frequency. Subsequent states follow the Markov model.
    previous_state = int(
        rng.random() < initial_wet_probability
    )

    states = []
    rainfall = []

    for i, day in enumerate(dates):
        if i == 0:
            state = previous_state
        else:
            probability = wet_transition_probability(
                occurrence_name,
                occurrence_parameters,
                day,
                previous_state,
            )
            state = int(rng.random() < probability)

        if state == 1:
            amount = sample_wet_intensity(
                intensity_name,
                intensity_parameters,
                rng,
            )
        else:
            amount = float(rng.choice(dry))

        states.append(state)
        rainfall.append(amount)
        previous_state = state

    return {
        "dates": dates,
        "states": states,
        "rainfall_mm": rainfall,
    }



def summarize_path(path):
    """Validate and summarize one synthetic rainfall path."""
    dates = path["dates"]
    states = np.asarray(path["states"])
    rainfall = np.asarray(path["rainfall_mm"], dtype=float)

    validate_simulation_dates(dates)

    if (
        rainfall.ndim != 1
        or states.ndim != 1
        or rainfall.size != len(dates)
        or states.size != len(dates)
        or not np.all(np.isfinite(rainfall))
        or np.any(rainfall < 0)
    ):
        raise ValueError("Invalid synthetic path")

    if any(
        type(state) not in (int, np.int64, np.int32)
        or state not in (0, 1)
        for state in states
    ):
        raise ValueError("Invalid simulated occurrence states")

    expected_states = (
        rainfall >= WET_THRESHOLD_MM
    ).astype(int)

    if not np.array_equal(
        states.astype(int), expected_states
    ):
        raise ValueError(
            "Rainfall amounts and occurrence states disagree"
        )

    return {
        "days": len(rainfall),
        "wet_days": int(np.count_nonzero(states)),
        "total_mm": float(np.sum(rainfall)),
        "maximum_mm": float(np.max(rainfall)),
        "events_ge_50": int(np.count_nonzero(rainfall >= 50)),
        "events_ge_100": int(np.count_nonzero(rainfall >= 100)),
        "events_ge_150": int(np.count_nonzero(rainfall >= 150)),
    }


def main():
    calibration_dates, calibration_rainfall = load_calibration()
    calibration_states = classify_days(calibration_rainfall)

    occurrence_models = {
        "B1": {"ALL": fit_b1(calibration_states)},
        "B2": fit_b2(calibration_dates, calibration_states),
    }

    wet_amounts = [
        amount for amount in calibration_rainfall
        if amount >= WET_THRESHOLD_MM
    ]
    dry_amounts = [
        amount for amount in calibration_rainfall
        if amount < WET_THRESHOLD_MM
    ]

    intensity_models = {
        "A1": fit_truncated_gamma(wet_amounts),
        "A2": fit_mixed_exponential(wet_amounts),
    }

    initial_wet_probability = (
        sum(calibration_states) / len(calibration_states)
    )

    dates = synthetic_dates(date(2027, 1, 1), 365)

    print("MONTE CARLO SIMULATOR — INITIAL SMOKE TEST")
    print(
        "Calibration:",
        calibration_dates[0], "to", calibration_dates[-1]
    )
    print("Simulation:", dates[0], "to", dates[-1])
    print("Calibration wet days:", len(wet_amounts))
    print("Calibration dry days:", len(dry_amounts))
    print("Initial wet probability:",
          f"{initial_wet_probability:.6f}")
    print("Holdout 2021-2025: NOT ACCESSED")

    print(
        f"\n{'Model':<7} {'Wet days':>9}"
        f" {'Total mm':>13} {'Max mm':>11}"
        f" {'>=50':>7} {'>=100':>7} {'>=150':>7}"
    )

    for index, (model_name, configuration) in enumerate(
        MODEL_CONFIGS.items()
    ):
        occurrence_name, intensity_name = configuration

        path = simulate_path(
            dates=dates,
            occurrence_name=occurrence_name,
            occurrence_parameters=occurrence_models[
                occurrence_name
            ],
            intensity_name=intensity_name,
            intensity_parameters=intensity_models[
                intensity_name
            ],
            dry_amounts=dry_amounts,
            initial_wet_probability=initial_wet_probability,
            seed=20261123 + index,
        )

        summary = summarize_path(path)

        print(
            f"{model_name:<7}"
            f" {summary['wet_days']:>9}"
            f" {summary['total_mm']:>13.2f}"
            f" {summary['maximum_mm']:>11.2f}"
            f" {summary['events_ge_50']:>7}"
            f" {summary['events_ge_100']:>7}"
            f" {summary['events_ge_150']:>7}"
        )

    print("\nPASS: initial simulation completed")
    print(
        "NOTE: One random path per model is a software smoke "
        "test, not a model-performance comparison."
    )


if __name__ == "__main__":
    main()
