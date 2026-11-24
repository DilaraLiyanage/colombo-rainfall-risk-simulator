
"""Repeated Monte Carlo rainfall simulation.

2,000 synthetic annual paths per model (M1, M2, M3).
Fits all parameters using 2001-2020 calibration data only.
Does not read the reserved 2021-2025 holdout observations.
"""

import math
from datetime import date

import numpy as np
from scipy.stats import gamma, norm, t

from rainfall_occurrence import (
    load_calibration,
    classify_days,
    fit_b1,
    fit_b2,
)
from rainfall_intensity import (
    fit_truncated_gamma,
    fit_mixed_exponential,
)
from monte_carlo_simulator import (
    MODEL_CONFIGS,
    synthetic_dates,
    validate_simulation_dates,
    wet_transition_probability,
)

N_REPLICATIONS = 2000
BASE_SEED = 2026111902
EXPERIMENT_YEAR = 2027
EXTREME_THRESHOLDS = (50.0, 100.0, 150.0)
WET_THRESHOLD_MM = 1.0


def sample_wet_batch(name, parameters, rng, count):
    """Generate a batch of independent wet-day intensities."""
    if count == 0:
        return np.empty(0, dtype=float)

    threshold = parameters["threshold_mm"]

    if name == "A1":
        shape = parameters["shape"]
        scale = parameters["scale"]

        survival = gamma.sf(
            threshold, a=shape, scale=scale
        )

        if not 0 < survival <= 1:
            raise ValueError("Invalid Gamma survival probability")

        uniforms = rng.random(count)
        uniforms = np.maximum(
            uniforms, np.finfo(float).tiny
        )

        values = gamma.isf(
            uniforms * survival,
            a=shape,
            scale=scale,
        )

    elif name == "A2":
        weight = parameters["weight_large"]
        small = parameters["scale_small"]
        large = parameters["scale_large"]

        choices = rng.random(count) < weight
        scales = np.where(choices, large, small)

        values = threshold + rng.exponential(scales)

    else:
        raise ValueError("Unknown intensity model")

    values = np.asarray(values, dtype=float)

    if (
        not np.all(np.isfinite(values))
        or np.any(values < WET_THRESHOLD_MM)
    ):
        raise RuntimeError("Invalid simulated wet-day intensity")

    return values



def simulate_many(
    dates,
    occurrence_name,
    occurrence_parameters,
    intensity_name,
    intensity_parameters,
    dry_amounts,
    initial_wet_probability,
    n_replications,
    seed,
    include_annual_metrics=False,
    extreme_thresholds=None,
):


    """Simulate independent synthetic paths in vectorized batches.

    Markov states are updated sequentially by day, but across all
    replications simultaneously. No observed future states are used.
    """
    validate_simulation_dates(dates)


    # Default to the three previously verified thresholds.
    if extreme_thresholds is None:
        thresholds = EXTREME_THRESHOLDS
    else:
        try:
            thresholds = tuple(
                float(value) for value in extreme_thresholds
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Extreme thresholds must be numeric"
            ) from exc

        if (
            not thresholds
            or not all(
                math.isfinite(value) and value >= WET_THRESHOLD_MM
                for value in thresholds
            )
            or len(set(thresholds)) != len(thresholds)
            or tuple(sorted(thresholds)) != thresholds
        ):
            raise ValueError(
                "Extreme thresholds must be finite, unique, "
                "sorted, and >= 1.0 mm"
            )


    if type(n_replications) is not int or n_replications < 2:
        raise ValueError("At least two replications are required")

    if not (
        math.isfinite(initial_wet_probability)
        and 0 <= initial_wet_probability <= 1
    ):
        raise ValueError("Invalid initial wet-day probability")

    dry = np.asarray(dry_amounts, dtype=float)

    if (
        dry.ndim != 1
        or dry.size == 0
        or not np.all(np.isfinite(dry))
        or np.any(dry < 0)
        or np.any(dry >= WET_THRESHOLD_MM)
    ):
        raise ValueError("Invalid dry-day calibration amounts")

    rng = np.random.default_rng(seed)
    n = n_replications

    previous = (
        rng.random(n) < initial_wet_probability
    ).astype(np.int8)

    totals = np.zeros(n, dtype=float)
    maxima = np.zeros(n, dtype=float)
    wet_days = np.zeros(n, dtype=np.int32)

    extreme_counts = {
        threshold: np.zeros(n, dtype=np.int32)
        for threshold in thresholds
    }


    # Track annual statistics without storing daily rainfall paths.
    if include_annual_metrics:
        years = tuple(sorted({day.year for day in dates}))

        year_to_index = {
            year: index
            for index, year in enumerate(years)
        }

        annual_shape = (n, len(years))

        annual_totals = np.zeros(
            annual_shape, dtype=float
        )
        annual_maxima = np.zeros(
            annual_shape, dtype=float
        )
        annual_wet_days = np.zeros(
            annual_shape, dtype=np.int32
        )

        annual_extreme_counts = {
            threshold: np.zeros(
                annual_shape, dtype=np.int32
            )
            for threshold in thresholds
        }

    for day_index, day in enumerate(dates):
        if day_index == 0:
            states = previous.copy()
        else:
            p_dry = wet_transition_probability(
                occurrence_name,
                occurrence_parameters,
                day,
                0,
            )
            p_wet = wet_transition_probability(
                occurrence_name,
                occurrence_parameters,
                day,
                1,
            )

            probabilities = np.where(
                previous == 1,
                p_wet,
                p_dry,
            )

            states = (
                rng.random(n) < probabilities
            ).astype(np.int8)

        wet_mask = states == 1
        n_wet = int(np.count_nonzero(wet_mask))
        n_dry = n - n_wet

        amounts = np.empty(n, dtype=float)

        if n_wet:
            amounts[wet_mask] = sample_wet_batch(
                intensity_name,
                intensity_parameters,
                rng,
                n_wet,
            )

        if n_dry:
            amounts[~wet_mask] = rng.choice(
                dry, size=n_dry, replace=True
            )

        if not np.array_equal(
            (amounts >= WET_THRESHOLD_MM).astype(np.int8),
            states,
        ):
            raise RuntimeError("Occurrence/intensity mismatch")

        totals += amounts
        maxima = np.maximum(maxima, amounts)
        wet_days += states

        for threshold in thresholds:
            extreme_counts[threshold] += (
                amounts >= threshold
            )

        if include_annual_metrics:
            year_index = year_to_index[day.year]

            annual_totals[:, year_index] += amounts

            annual_maxima[:, year_index] = np.maximum(
                annual_maxima[:, year_index],
                amounts,
            )

            annual_wet_days[:, year_index] += states

            for threshold in thresholds:
                annual_extreme_counts[threshold][
                    :, year_index
                ] += (amounts >= threshold)

        previous = states


    result = {
        "annual_total_mm": totals,
        "annual_maximum_mm": maxima,
        "annual_wet_days": wet_days,
        "extreme_counts": extreme_counts,
        "n_replications": n,
        "days_per_path": len(dates),
    }

    if include_annual_metrics:
        result["years"] = years
        result["yearly_total_mm"] = annual_totals
        result["yearly_maximum_mm"] = annual_maxima
        result["yearly_wet_days"] = annual_wet_days
        result["yearly_extreme_counts"] = annual_extreme_counts

        result["annual_total_sd_mm"] = (
            np.std(annual_totals, axis=1, ddof=1)
            if len(years) >= 2
            else np.full(n, np.nan)
        )

    return result


def mean_statistics(values):
    """Mean, 95% Monte Carlo CI, and 95% predictive interval."""
    x = np.asarray(values, dtype=float)

    if x.ndim != 1 or len(x) < 2:
        raise ValueError("Insufficient simulation results")

    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite simulation results")

    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))

    critical = float(t.ppf(0.975, df=n - 1))
    margin = critical * sd / math.sqrt(n)

    lower, upper = np.quantile(x, [0.025, 0.975])

    return {
        "mean": mean,
        "sd": sd,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
        "predictive_low": float(lower),
        "predictive_high": float(upper),
    }


def event_probability_statistics(event_counts):
    """Probability of >=1 event with a 95% Wilson interval."""
    x = np.asarray(event_counts)

    if x.ndim != 1 or len(x) < 2:
        raise ValueError("Insufficient event results")

    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("Invalid event counts")

    successes = int(np.count_nonzero(x >= 1))
    n = len(x)
    probability = successes / n

    z = float(norm.ppf(0.975))
    denominator = 1 + z * z / n

    center = (
        probability + z * z / (2 * n)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            probability * (1 - probability) / n
            + z * z / (4 * n * n)
        )
        / denominator
    )

    return {
        "probability": probability,
        "ci_low": max(0.0, center - half_width),
        "ci_high": min(1.0, center + half_width),
        "successes": successes,
        "n": n,
    }


def main():
    dates_cal, rainfall_cal = load_calibration()
    states_cal = classify_days(rainfall_cal)

    if (
        dates_cal[0] != date(2001, 1, 1)
        or dates_cal[-1] != date(2020, 12, 31)
        or len(dates_cal) != 7305
    ):
        raise RuntimeError("Unexpected calibration period")

    occurrences = {
        "B1": {"ALL": fit_b1(states_cal)},
        "B2": fit_b2(dates_cal, states_cal),
    }

    wet = [
        amount for amount in rainfall_cal
        if amount >= WET_THRESHOLD_MM
    ]
    dry = [
        amount for amount in rainfall_cal
        if amount < WET_THRESHOLD_MM
    ]

    intensities = {
        "A1": fit_truncated_gamma(wet),
        "A2": fit_mixed_exponential(wet),
    }

    initial_probability = sum(states_cal) / len(states_cal)

    dates = synthetic_dates(
        date(EXPERIMENT_YEAR, 1, 1), 365
    )

    print("REPEATED MONTE CARLO RAINFALL EXPERIMENT")
    print("Calibration:", dates_cal[0], "to", dates_cal[-1])
    print("Synthetic period:", dates[0], "to", dates[-1])
    print("Replications per model:", N_REPLICATIONS)
    print("Base seed:", BASE_SEED)
    print("Holdout 2021-2025: NOT ACCESSED")

    for index, (model_name, configuration) in enumerate(
        MODEL_CONFIGS.items()
    ):
        occurrence_name, intensity_name = configuration

        # Reproducible, distinct stream for each model.
        seed_sequence = np.random.SeedSequence(
            [BASE_SEED, index]
        )

        results = simulate_many(
            dates=dates,
            occurrence_name=occurrence_name,
            occurrence_parameters=occurrences[occurrence_name],
            intensity_name=intensity_name,
            intensity_parameters=intensities[intensity_name],
            dry_amounts=dry,
            initial_wet_probability=initial_probability,
            n_replications=N_REPLICATIONS,
            seed=seed_sequence,
        )

        totals = mean_statistics(results["annual_total_mm"])
        maxima = mean_statistics(results["annual_maximum_mm"])
        wet_days = mean_statistics(results["annual_wet_days"])

        print(f"\n{'=' * 50}")
        print(f"MODEL {model_name}: {occurrence_name} + {intensity_name}")
        print(f"{'=' * 50}")

        print(f"Mean annual rainfall: {totals['mean']:.2f} mm")
        print(
            "95% CI for mean annual rainfall: "
            f"[{totals['ci_low']:.2f}, "
            f"{totals['ci_high']:.2f}] mm"
        )
        print(
            "95% annual rainfall predictive interval: "
            f"[{totals['predictive_low']:.2f}, "
            f"{totals['predictive_high']:.2f}] mm"
        )
        print(f"SD annual rainfall: {totals['sd']:.2f} mm")
        print(f"Mean wet days: {wet_days['mean']:.2f}")

        print(f"Mean annual maximum: {maxima['mean']:.2f} mm")

        print("\nExtreme rainfall events per synthetic year:")

        for threshold in EXTREME_THRESHOLDS:
            counts = results["extreme_counts"][threshold]


            expected_count = mean_statistics(counts)
            event_probability = event_probability_statistics(counts)

            print(
                f"  >= {threshold:.0f} mm: "
                f"mean count = {expected_count['mean']:.4f}, "
                f"P(at least one) = "
                f"{event_probability['probability']:.4%}, "
                f"95% CI = "
                f"[{event_probability['ci_low']:.4%}, "
                f"{event_probability['ci_high']:.4%}]"
            )

    print("\nPASS: repeated Monte Carlo experiment completed")
    print(
        "NOTE: Confidence intervals reflect Monte Carlo "
        "estimation uncertainty conditional on fitted parameters."
    )
    print(
        "NOTE: Predictive intervals describe simulated "
        "year-to-year variation, not observed forecast accuracy."
    )


if __name__ == "__main__":
    main()
