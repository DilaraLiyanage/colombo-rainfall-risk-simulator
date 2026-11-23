
"""Calibration-only comparison of B1, B2 and B3 occurrence models.

Computes conditional transition log-likelihood, AIC, BIC
and denominator diagnostics.

Does not access reserved 2021–2025 holdout observations.
"""

import math

from rainfall_occurrence import (
    load_calibration,
    classify_days,
    fit_b1,
    fit_b2,
    fit_b3,
)

TRANSITIONS = (
    ("N00", "P00"),
    ("N01", "P01"),
    ("N10", "P10"),
    ("N11", "P11"),
)


def transition_log_likelihood(counts, probabilities):
    """Conditional log-likelihood for one Markov transition matrix."""
    total = 0.0

    for count_key, probability_key in TRANSITIONS:
        count = counts[count_key]
        probability = probabilities[probability_key]

        if count < 0 or not 0 <= probability <= 1:
            raise ValueError("Invalid transition count or probability")

        if count == 0:
            continue

        if probability == 0:
            return float("-inf")

        total += count * math.log(probability)

    return total


def evaluate_model(name, groups):
    """Calculate likelihood and information criteria.

    Each fitted group has two free parameters:
    P01 and P11. Their complements are P00 and P10.
    """
    log_likelihood = 0.0
    n = 0
    minimum_denominator = float("inf")

    for result in groups.values():
        counts = result["counts"]
        probabilities = result["probabilities"]

        log_likelihood += transition_log_likelihood(
            counts, probabilities
        )

        n += sum(counts.values())

        dry_total = counts["N00"] + counts["N01"]
        wet_total = counts["N10"] + counts["N11"]

        minimum_denominator = min(
            minimum_denominator, dry_total, wet_total
        )

    if n <= 0:
        raise ValueError("No transitions available")

    k = 2 * len(groups)

    aic = 2 * k - 2 * log_likelihood
    bic = k * math.log(n) - 2 * log_likelihood

    return {
        "name": name,
        "groups": len(groups),
        "parameters": k,
        "transitions": n,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "minimum_denominator": minimum_denominator,
    }


def compare_occurrence_models(dates, states):
    """Compare three models fitted to the same transitions."""
    b1 = {"ALL": fit_b1(states)}
    b2 = fit_b2(dates, states)
    b3 = fit_b3(dates, states)

    results = [
        evaluate_model("B1", b1),
        evaluate_model("B2", b2),
        evaluate_model("B3", b3),
    ]

    expected_n = len(states) - 1

    if any(r["transitions"] != expected_n for r in results):
        raise ValueError("Models used different transition counts")

    return results


def main():
    dates, rainfall = load_calibration()
    states = classify_days(rainfall)

    results = compare_occurrence_models(dates, states)

    print("CALIBRATION-ONLY OCCURRENCE MODEL COMPARISON")
    print("Period:", dates[0], "to", dates[-1])
    print("Transitions:", len(states) - 1)

    print(
        f"\n{'Model':<7} {'k':>5} {'LogLik':>14} "
        f"{'AIC':>14} {'BIC':>14} {'Min N':>8}"
    )

    for r in results:
        print(
            f"{r['name']:<7} "
            f"{r['parameters']:>5} "
            f"{r['log_likelihood']:>14.3f} "
            f"{r['aic']:>14.3f} "
            f"{r['bic']:>14.3f} "
            f"{r['minimum_denominator']:>8}"
        )

    best_aic = min(results, key=lambda r: r["aic"])
    best_bic = min(results, key=lambda r: r["bic"])

    print("\nLowest calibration AIC:", best_aic["name"])
    print("Lowest calibration BIC:", best_bic["name"])

    print("\nPASS: calibration comparison completed")
    print(
        "NOTE: These are in-sample diagnostics, "
        "not independent predictive validation."
    )


if __name__ == "__main__":
    main()
