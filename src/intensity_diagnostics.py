
"""Calibration-only diagnostics for rainfall intensity models.

Compares:
  A1: Left-truncated Gamma
  A2: Shifted two-component exponential mixture

Uses 2001-2020 calibration observations only.
Does not access the reserved 2021-2025 holdout.
"""

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import gamma

from rainfall_occurrence import load_calibration
from rainfall_intensity import (
    WET_THRESHOLD_MM,
    fit_truncated_gamma,
    fit_mixed_exponential,
)


QUANTILES = (0.50, 0.90, 0.95, 0.99, 0.995)
EXTREME_THRESHOLDS_MM = (50.0, 100.0, 150.0)


def validate_probability(probability):
    """Accept probabilities within the closed interval [0, 1]."""
    p = float(probability)

    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("Invalid probability")

    return p


def a1_survival(x, parameters):
    """P(X >= x | X >= wet threshold) for A1."""
    threshold = parameters["threshold_mm"]
    shape = parameters["shape"]
    scale = parameters["scale"]

    if not math.isfinite(x):
        raise ValueError("Invalid rainfall threshold")

    if x <= threshold:
        return 1.0

    log_survival = (
        gamma.logsf(x, a=shape, scale=scale)
        - gamma.logsf(threshold, a=shape, scale=scale)
    )

    return float(math.exp(log_survival))


def a2_survival(x, parameters):
    """P(X >= x) for shifted mixture A2."""
    threshold = parameters["threshold_mm"]
    weight = parameters["weight_large"]
    small = parameters["scale_small"]
    large = parameters["scale_large"]

    if not math.isfinite(x):
        raise ValueError("Invalid rainfall threshold")

    if x <= threshold:
        return 1.0

    excess = x - threshold

    return float(
        (1.0 - weight) * math.exp(-excess / small)
        + weight * math.exp(-excess / large)
    )


def a1_quantile(probability, parameters):
    """Conditional Gamma quantile on [threshold, infinity)."""
    p = validate_probability(probability)

    if p == 0.0:
        return parameters["threshold_mm"]

    if p == 1.0:
        return math.inf

    threshold = parameters["threshold_mm"]
    shape = parameters["shape"]
    scale = parameters["scale"]

    conditional_tail = (
        (1.0 - p)
        * gamma.sf(threshold, a=shape, scale=scale)
    )

    return float(
        gamma.isf(conditional_tail, a=shape, scale=scale)
    )


def a2_quantile(probability, parameters):
    """Numerical quantile of the shifted exponential mixture."""
    p = validate_probability(probability)

    threshold = parameters["threshold_mm"]

    if p == 0.0:
        return threshold

    if p == 1.0:
        return math.inf

    target_survival = 1.0 - p
    lower = threshold
    upper = threshold + max(
        parameters["scale_small"],
        parameters["scale_large"],
    )

    while a2_survival(upper, parameters) > target_survival:
        upper = threshold + 2.0 * (upper - threshold)

        if not math.isfinite(upper):
            raise RuntimeError("Failed to bracket mixture quantile")

    return float(
        brentq(
            lambda x: a2_survival(x, parameters)
            - target_survival,
            lower,
            upper,
            xtol=1e-10,
        )
    )


def intensity_diagnostics(wet_amounts, a1, a2):
    """Calculate observed and fitted intensity diagnostics."""
    amounts = np.asarray(wet_amounts, dtype=float)

    if (
        amounts.ndim != 1
        or amounts.size < 2
        or not np.all(np.isfinite(amounts))
        or np.any(amounts < WET_THRESHOLD_MM)
    ):
        raise ValueError("Invalid wet-day observations")

    if (
        a1["n_wet_days"] != len(amounts)
        or a2["n_wet_days"] != len(amounts)
    ):
        raise ValueError("Model sample sizes do not match")

    quantiles = []

    for probability in QUANTILES:
        quantiles.append({
            "percentile": probability * 100,
            "observed_mm": float(np.quantile(amounts, probability)),
            "a1_mm": a1_quantile(probability, a1),
            "a2_mm": a2_quantile(probability, a2),
        })

    exceedances = []
    n = len(amounts)

    for threshold in EXTREME_THRESHOLDS_MM:
        observed_count = int(np.count_nonzero(amounts >= threshold))
        observed_probability = observed_count / n

        p_a1 = a1_survival(threshold, a1)
        p_a2 = a2_survival(threshold, a2)

        exceedances.append({
            "threshold_mm": threshold,
            "observed_count": observed_count,
            "observed_probability": observed_probability,
            "a1_probability": p_a1,
            "a2_probability": p_a2,
            "a1_expected_count": n * p_a1,
            "a2_expected_count": n * p_a2,
        })

    comparison = {}

    for name, parameters, k in (
        ("A1", a1, 2),
        ("A2", a2, 3),
    ):
        ll = parameters["log_likelihood"]

        comparison[name] = {
            "parameters": k,
            "log_likelihood": ll,
            "aic": 2 * k - 2 * ll,
            "bic": k * math.log(n) - 2 * ll,
        }

    return {
        "n_wet_days": n,
        "quantiles": quantiles,
        "exceedances": exceedances,
        "comparison": comparison,
    }


def main():
    dates, rainfall = load_calibration()

    wet_amounts = [
        amoun
        for amount in rainfall
        if amount >= WET_THRESHOLD_MM
    ]

    a1 = fit_truncated_gamma(wet_amounts)
    a2 = fit_mixed_exponential(wet_amounts)

    results = intensity_diagnostics(wet_amounts, a1, a2)

    print("CALIBRATION-ONLY INTENSITY DIAGNOSTICS")
    print("Period:", dates[0], "to", dates[-1])
    print("Wet-day threshold:", WET_THRESHOLD_MM, "mm")
    print("Wet-day observations:", results["n_wet_days"])
    print("Holdout 2021-2025: NOT ACCESSED")

    print("\nMODEL FIT COMPARISON")
    print(
        f"{'Model':<8} {'k':>4} {'LogLik':>14}"
        f" {'AIC':>14} {'BIC':>14}"
    )

    for name, r in results["comparison"].items():
        print(
            f"{name:<8} {r['parameters']:>4}"
            f" {r['log_likelihood']:>14.3f}"
            f" {r['aic']:>14.3f}"
            f" {r['bic']:>14.3f}"
        )

    print("\nWET-DAY RAINFALL PERCENTILES (mm)")
    print(
        f"{'Percentile':<12} {'Observed':>12}"
        f" {'A1':>12} {'A2':>12}"
    )

    for r in results["quantiles"]:
        print(
            f"{r['percentile']:>9.1f}%"
            f" {r['observed_mm']:>12.3f}"
            f" {r['a1_mm']:>12.3f}"
            f" {r['a2_mm']:>12.3f}"
        )

    print("\nEXTREME RAINFALL EXCEEDANCES")
    print("Counts are among wet days in the calibration period.")
    print(
        f"{'Threshold':>11} {'Observed':>10}"
        f" {'A1 expected':>13} {'A2 expected':>13}"
    )

    for r in results["exceedances"]:
        print(
            f"{r['threshold_mm']:>9.0f} mm"
            f" {r['observed_count']:>10}"
            f" {r['a1_expected_count']:>13.3f}"
            f" {r['a2_expected_count']:>13.3f}"
        )

    print("\nEXCEEDANCE PROBABILITIES (conditional on wet day)")
    print(
        f"{'Threshold':>11} {'Observed':>12}"
        f" {'A1':>12} {'A2':>12}"
    )

    for r in results["exceedances"]:
        print(
            f"{r['threshold_mm']:>9.0f} mm"
            f" {r['observed_probability']:>12.6f}"
            f" {r['a1_probability']:>12.6f}"
            f" {r['a2_probability']:>12.6f}"
        )

    print("\nPASS: intensity diagnostics completed")
    print(
        "NOTE: In-sample diagnostics; not independent "
        "predictive validation."
    )


if __name__ == "__main__":
    main()
