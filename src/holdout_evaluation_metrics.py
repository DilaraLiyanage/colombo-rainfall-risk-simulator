
"""Pure statistical metrics for independent rainfall evaluation.

This module performs no file reading and contains no data loading.
Observed values must be supplied explicitly by the caller.

All metrics are conditional on the fitted simulation model.
"""

import numpy as np


def validate_simulations(simulated_values):
    """Require a finite, nonnegative, one-dimensional simulation sample."""
    values = np.asarray(simulated_values, dtype=float)

    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two simulated values are required")

    if not np.all(np.isfinite(values)):
        raise ValueError("Simulations must contain finite values")

    if np.any(values < 0):
        raise ValueError("Rainfall metrics cannot be negative")

    return values


def predictive_summary(simulated_values):
    """Summarize a distribution of comparable simulated outcomes."""
    values = validate_simulations(simulated_values)

    low, median, high = np.quantile(
        values,
        [0.025, 0.5, 0.975],
    )

    return {
        "n_simulations": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(median),
        "sd": float(np.std(values, ddof=1)),
        "predictive_low": float(low),
        "predictive_high": float(high),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def compare_observed(observed_value, simulated_values):
    """Compare one observed statistic with matched simulation outcomes.

    The observed value and every simulated value must describe
    the same period and metric.

    The tail fractions are descriptive Monte Carlo proportions,
    not formal hypothesis-test p-values.
    """
    values = validate_simulations(simulated_values)

    observed = float(observed_value)

    if not np.isfinite(observed) or observed < 0:
        raise ValueError("Observed metric must be finite and nonnegative")

    summary = predictive_summary(values)

    below = float(np.mean(values < observed))
    equal = float(np.mean(values == observed))
    above = float(np.mean(values > observed))

    percentile_midrank = below + 0.5 * equal

    if observed < summary["predictive_low"]:
        position = "below"
    elif observed > summary["predictive_high"]:
        position = "above"
    else:
        position = "inside"

    return {
        "observed": observed,
        "simulated_mean": summary["mean"],
        "simulated_median": summary["median"],
        "predictive_low": summary["predictive_low"],
        "predictive_high": summary["predictive_high"],
        "predictive_position": position,
        "difference_from_mean": observed - summary["mean"],
        "absolute_error": abs(observed - summary["mean"]),
        "fraction_simulations_below": below,
        "fraction_simulations_equal": equal,
        "fraction_simulations_above": above,
        "percentile_midrank": percentile_midrank,
    }


def compare_event_count(observed_count, simulated_counts):
    """Compare one observed extreme-event count with simulated counts.

    Both inputs must refer to the same rainfall threshold and
    the same observation period.

    Returned probabilities are empirical simulation frequencies,
    conditional on the fitted model, not formal p-values.
    """
    counts = validate_simulations(simulated_counts)

    if not np.all(counts == np.floor(counts)):
        raise ValueError("Simulated event counts must be integers")

    observed = float(observed_count)

    if (
        not np.isfinite(observed)
        or observed < 0
        or not observed.is_integer()
    ):
        raise ValueError(
            "Observed event count must be a nonnegative integer"
        )

    observed = int(observed)

    comparison = compare_observed(observed, counts)

    # Discrete empirical predictive interval:
    # select actual simulated count values, not interpolated counts.
    discrete_low, discrete_high = np.quantile(
        counts,
        [0.025, 0.975],
        method="inverted_cdf",
    )

    comparison["predictive_low"] = int(discrete_low)
    comparison["predictive_high"] = int(discrete_high)

    if observed < discrete_low:
        comparison["predictive_position"] = "below"
    elif observed > discrete_high:
        comparison["predictive_position"] = "above"
    else:
        comparison["predictive_position"] = "inside"

    comparison.update({
        "observed_count": observed,
        "mean_simulated_count": float(np.mean(counts)),
        "probability_zero_events": float(np.mean(counts == 0)),
        "probability_at_least_one": float(np.mean(counts >= 1)),
        "probability_count_at_least_observed": float(
            np.mean(counts >= observed)
        ),
        "probability_count_greater_than_observed": float(
            np.mean(counts > observed)
        ),
        "probability_exact_observed_count": float(
            np.mean(counts == observed)
        ),
    })

    return comparison


def main():
    """Demonstrate the functions using artificial data only."""
    rng = np.random.default_rng(62001)

    synthetic_simulations = rng.normal(
        loc=14000.0,
        scale=500.0,
        size=2000,
    )

    synthetic_simulations = np.maximum(
        synthetic_simulations,
        0.0,
    )

    artificial_observation = 14200.0

    summary = predictive_summary(synthetic_simulations)
    comparison = compare_observed(
        artificial_observation,
        synthetic_simulations,
    )

    print("INDEPENDENT EVALUATION METRICS — SYNTHETIC TEST")
    print("Simulation count:", summary["n_simulations"])
    print("Simulated mean:", round(summary["mean"], 2))
    print(
        "95% predictive interval:",
        round(summary["predictive_low"], 2),
        "to",
        round(summary["predictive_high"], 2),
    )
    print("Artificial observation:", artificial_observation)
    print("Position:", comparison["predictive_position"])
    print(
        "Percentile midrank:",
        round(100 * comparison["percentile_midrank"], 2),
        "%",
    )
    print("Observed holdout rainfall: NOT ACCESSED")
    print("PASS: synthetic evaluation metrics completed")


if __name__ == "__main__":
    main()
