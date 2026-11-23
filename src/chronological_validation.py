
"""Chronological occurrence-model validation within 2001–2020.

Train: 2001–2015
Validate: 2016–2020
Reserved holdout: 2021–2025, never accessed.

Predicts each validation day's wet/dry state using the
actually observed previous-day state (one-step-ahead evaluation).
"""

import math
from datetime import date

from rainfall_occurrence import (
    load_calibration,
    classify_days,
    fit_b1,
    fit_b2,
    fit_b3,
    season_for_date,
)

TRAIN_END = date(2015, 12, 31)
VALIDATION_START = date(2016, 1, 1)
VALIDATION_END = date(2020, 12, 31)
SMOOTHING_ALPHA = 0.5


def smoothed_wet_probability(counts, previous_state, alpha=SMOOTHING_ALPHA):
    """Estimate P(next day wet | previous state) with fixed smoothing."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Smoothing alpha must be positive")

    if previous_state not in (0, 1) or type(previous_state) is not int:
        raise ValueError("Invalid previous-day state")

    if previous_state == 0:
        wet = counts["N01"]
        total = counts["N00"] + counts["N01"]
    else:
        wet = counts["N11"]
        total = counts["N10"] + counts["N11"]

    if total < 0 or wet < 0 or wet > total:
        raise ValueError("Invalid transition counts")

    return (wet + alpha) / (total + 2 * alpha)


def predict_probability(model_name, fitted, day, previous_state):
    """Return a one-step-ahead wet-day probability."""
    if model_name == "B1":
        counts = fitted["ALL"]["counts"]
    elif model_name == "B2":
        counts = fitted[season_for_date(day)]["counts"]
    elif model_name == "B3":
        counts = fitted[day.month]["counts"]
    else:
        raise ValueError("Unknown model")

    return smoothed_wet_probability(counts, previous_state)


def score_predictions(actual, probabilities):
    """Calculate proper probability scores and classification accuracy."""
    if len(actual) != len(probabilities) or not actual:
        raise ValueError("Mismatched or empty predictions")

    log_losses = []
    brier_losses = []
    correct = 0

    for outcome, probability in zip(actual, probabilities):
        if type(outcome) is not int or outcome not in (0, 1):
            raise ValueError("Invalid outcome")

        if not math.isfinite(probability) or not 0 < probability < 1:
            raise ValueError("Predictions must be probabilities in (0,1)")

        loss = -(
            outcome * math.log(probability)
            + (1 - outcome) * math.log1p(-probability)
        )

        log_losses.append(loss)
        brier_losses.append((probability - outcome) ** 2)
        correct += int(int(probability >= 0.5) == outcome)

    n = len(actual)

    return {
        "n": n,
        "log_loss": math.fsum(log_losses) / n,
        "brier_score": math.fsum(brier_losses) / n,
        "accuracy": correct / n,
    }


def chronological_validate(dates, states):
    """Fit on earlier dates; evaluate later dates without refitting."""

    if len(dates) != len(states):
        raise ValueError("Dates and states have different lengths")

    if len(dates) != 7305:
        raise ValueError("Incorrect calibration dataset length")

    expected_start = date(2001, 1, 1)

    for i, (day, state) in enumerate(zip(dates, states)):
        if type(day) is not date:
            raise ValueError("Invalid date type")

        if day != expected_start.fromordinal(
            expected_start.toordinal() + i
        ):
            raise ValueError("Nonconsecutive calibration dates")

        if type(state) is not int or state not in (0, 1):
            raise ValueError("Invalid rainfall occurrence state")

    train_indices = [
        i for i, day in enumerate(dates)
        if day <= TRAIN_END
    ]

    validation_indices = [
        i for i, day in enumerate(dates)
        if VALIDATION_START <= day <= VALIDATION_END
    ]

    if (
        not train_indices
        or not validation_indices
        or train_indices[-1] + 1 != validation_indices[0]
        or dates[train_indices[0]] != date(2001, 1, 1)
        or dates[train_indices[-1]] != TRAIN_END
        or dates[validation_indices[-1]] != VALIDATION_END
        or len(train_indices) != 5478
        or len(validation_indices) != 1827
        or len(dates) != 7305
    ):
        raise ValueError("Unexpected chronological split")

    train_dates = [dates[i] for i in train_indices]
    train_states = [states[i] for i in train_indices]

    fitted = {
        "B1": {"ALL": fit_b1(train_states)},
        "B2": fit_b2(train_dates, train_states),
        "B3": fit_b3(train_dates, train_states),
    }

    actual = [states[i] for i in validation_indices]
    results = {}

    for name in ("B1", "B2", "B3"):
        probabilities = [
            predict_probability(
                name,
                fitted[name],
                dates[i],
                states[i - 1],
            )
            for i in validation_indices
        ]

        results[name] = score_predictions(actual, probabilities)

    return results


def main():
    dates, rainfall = load_calibration()
    states = classify_days(rainfall)

    results = chronological_validate(dates, states)

    print("CHRONOLOGICAL OCCURRENCE VALIDATION")
    print("Training: 2001-01-01 to 2015-12-31 (5478 days)")
    print("Validation: 2016-01-01 to 2020-12-31 (1827 days)")
    print("Smoothing alpha:", SMOOTHING_ALPHA)
    print("Holdout 2021-2025: NOT ACCESSED")

    print(
        f"\n{'Model':<8} {'N':>6} {'Log loss':>12} "
        f"{'Brier':>12} {'Accuracy':>12}"
    )

    for name, r in results.items():
        print(
            f"{name:<8} {r['n']:>6} "
            f"{r['log_loss']:>12.6f} "
            f"{r['brier_score']:>12.6f} "
            f"{r['accuracy']:>12.4%}"
        )

    print(
        "\nLowest validation log loss:",
        min(results, key=lambda name: results[name]["log_loss"]),
    )
    print(
        "Lowest validation Brier score:",
        min(results, key=lambda name: results[name]["brier_score"]),
    )
    print("\nPASS: chronological validation completed")


if __name__ == "__main__":
    main()
