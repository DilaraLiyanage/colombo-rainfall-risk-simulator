
"""Pre-holdout Monte Carlo simulation verification.

Uses the calendar dates of 2021-2025, but never reads
rainfall observations from those years.

All model parameters and dry-day amounts are obtained
exclusively from the 2001-2020 calibration dataset.
"""

from datetime import date

import numpy as np

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
)
from repeated_monte_carlo import (
    BASE_SEED,
    N_REPLICATIONS,
    EXTREME_THRESHOLDS,
    simulate_many,
    mean_statistics,
    event_probability_statistics,
)



HOLDOUT_START = date(2021, 1, 1)
HOLDOUT_END = date(2025, 12, 31)
EXPECTED_DAYS = 1826
WET_THRESHOLD_MM = 1.0

def evaluation_thresholds(calibration_rainfall):
    """Calculate the five predefined extreme-event thresholds.

    Percentile thresholds use wet-day rainfall from calibration
    exclusively, with NumPy's default linear quantile method.
    """
    rainfall = np.asarray(calibration_rainfall, dtype=float)

    if (
        rainfall.ndim != 1
        or rainfall.size == 0
        or not np.all(np.isfinite(rainfall))
        or np.any(rainfall < 0)
    ):
        raise ValueError("Invalid calibration rainfall")

    wet_amounts = rainfall[rainfall >= WET_THRESHOLD_MM]

    if wet_amounts.size < 2:
        raise ValueError(
            "At least two wet-day observations are required"
        )

    percentile_99, percentile_995 = np.quantile(
        wet_amounts,
        [0.99, 0.995],
        method="linear",
    )

    thresholds = tuple(sorted((
        50.0,
        float(percentile_99),
        float(percentile_995),
        100.0,
        150.0,
    )))

    if len(set(thresholds)) != 5:
        raise ValueError(
            "Calibration thresholds must be distinct"
        )

    return thresholds


def evaluation_dates():
    """Create and validate the predetermined five-year calendar."""
    dates = synthetic_dates(HOLDOUT_START, EXPECTED_DAYS)

    if (
        len(dates) != EXPECTED_DAYS
        or dates[0] != HOLDOUT_START
        or dates[-1] != HOLDOUT_END
        or dates.count(date(2024, 2, 29)) != 1
    ):
        raise RuntimeError("Incorrect evaluation calendar")

    return dates


def validate_five_year_results(result, thresholds=EXTREME_THRESHOLDS):
    """Check dimensions and internal consistency of simulation output."""
    n = result["n_replications"]

    if (
        n != N_REPLICATIONS
        or result["days_per_path"] != EXPECTED_DAYS
    ):
        raise RuntimeError("Incorrect experiment dimensions")

    totals = np.asarray(result["annual_total_mm"])
    maxima = np.asarray(result["annual_maximum_mm"])
    wet_days = np.asarray(result["annual_wet_days"])

    for values in (totals, maxima, wet_days):
        if values.shape != (n,) or not np.all(np.isfinite(values)):
            raise RuntimeError("Invalid simulation output shape or values")

    if (
        np.any(totals < 0)
        or np.any(maxima < 0)
        or np.any(wet_days < 0)
        or np.any(wet_days > EXPECTED_DAYS)
        or np.any(totals < maxima)
    ):
        raise RuntimeError("Invalid simulated rainfall statistics")

    previous_counts = None

    for threshold in thresholds:
        counts = np.asarray(result["extreme_counts"][threshold])

        if (
            counts.shape != (n,)
            or not np.all(np.isfinite(counts))
            or np.any(counts < 0)
            or np.any(counts > wet_days)
        ):
            raise RuntimeError("Invalid extreme-event counts")

        if previous_counts is not None and np.any(
            counts > previous_counts
        ):
            raise RuntimeError("Extreme-event counts not ordered")

        previous_counts = counts


        # Validate optional annual tracking outputs.
        if "yearly_total_mm" in result:
            expected_years = (2021, 2022, 2023, 2024, 2025)

            if result["years"] != expected_years:
                raise RuntimeError("Incorrect annual year labels")

            yearly_totals = np.asarray(
                result["yearly_total_mm"]
            )
            yearly_wet_days = np.asarray(
                result["yearly_wet_days"]
            )
            yearly_maxima = np.asarray(
                result["yearly_maximum_mm"]
            )

            for values in (
                yearly_totals,
                yearly_wet_days,
                yearly_maxima,
            ):
                if (
                    values.shape != (n, 5)
                    or not np.all(np.isfinite(values))
                    or np.any(values < 0)
                ):
                    raise RuntimeError("Invalid yearly statistics")

            # Five-year totals must equal summed annual totals.
            np.testing.assert_allclose(
                yearly_totals.sum(axis=1),
                totals,
                rtol=1e-12,
                atol=1e-8,
            )

            np.testing.assert_array_equal(
                yearly_wet_days.sum(axis=1),
                wet_days,
            )

            np.testing.assert_array_equal(
                yearly_maxima.max(axis=1),
                maxima,
            )

            yearly_counts = result["yearly_extreme_counts"]

            for threshold in thresholds:
                values = np.asarray(yearly_counts[threshold])

                if (
                    values.shape != (n, 5)
                    or not np.all(np.isfinite(values))
                    or np.any(values < 0)
                ):
                    raise RuntimeError(
                        "Invalid yearly extreme-event counts"
                    )

                np.testing.assert_array_equal(
                    values.sum(axis=1),
                    result["extreme_counts"][threshold],
                )

            expected_sd = np.std(
                yearly_totals,
                axis=1,
                ddof=1,
            )

            np.testing.assert_allclose(
                result["annual_total_sd_mm"],
                expected_sd,
                rtol=1e-12,
                atol=1e-8,
            )

def main():
    dates = evaluation_dates()

    calibration_dates, rainfall = load_calibration()

    if (
        len(calibration_dates) != 7305
        or calibration_dates[0] != date(2001, 1, 1)
        or calibration_dates[-1] != date(2020, 12, 31)
    ):
        raise RuntimeError("Incorrect calibration period")

    # Predefined thresholds using calibration rainfall only.
    thresholds = evaluation_thresholds(rainfall)

    print("\nFive predefined extreme-rainfall thresholds (mm):")
    for threshold in thresholds:
        print(f"  {threshold:.12f}")

    states = classify_days(rainfall)

    wet_amounts = [
        value for value in rainfall
        if value >= WET_THRESHOLD_MM
    ]

    dry_amounts = [
        value for value in rainfall
        if value < WET_THRESHOLD_MM
    ]

    occurrence_models = {
        "B1": {"ALL": fit_b1(states)},
        "B2": fit_b2(calibration_dates, states),
    }

    intensity_models = {
        "A1": fit_truncated_gamma(wet_amounts),
        "A2": fit_mixed_exponential(wet_amounts),
    }

    initial_probability = sum(states) / len(states)

    print("FIVE-YEAR MONTE CARLO — PRE-HOLDOUT CHECK")
    print("Calibration:", calibration_dates[0], "to", calibration_dates[-1])
    print("Evaluation calendar:", dates[0], "to", dates[-1])
    print("Calendar days:", len(dates))
    print("Leap day included:", date(2024, 2, 29) in dates)
    print("Replications per model:", N_REPLICATIONS)
    print("Base seed:", BASE_SEED)
    print("Observed holdout rainfall: NOT ACCESSED")

    for index, (name, combination) in enumerate(
        MODEL_CONFIGS.items()
    ):
        occurrence_name, intensity_name = combination

        results = simulate_many(
            dates=dates,
            occurrence_name=occurrence_name,
            occurrence_parameters=occurrence_models[occurrence_name],
            intensity_name=intensity_name,
            intensity_parameters=intensity_models[intensity_name],
            dry_amounts=dry_amounts,
            initial_wet_probability=initial_probability,
            n_replications=N_REPLICATIONS,
            seed=np.random.SeedSequence([BASE_SEED, index]),
            include_annual_metrics=True,
            extreme_thresholds=thresholds,
        )

        validate_five_year_results(results, thresholds=thresholds)

        totals = mean_statistics(results["annual_total_mm"])
        maxima = mean_statistics(results["annual_maximum_mm"])
        wet_days = mean_statistics(results["annual_wet_days"])

        print(f"\nMODEL {name}: {occurrence_name} + {intensity_name}")
        print("Simulated five-year paths:", N_REPLICATIONS)
        print(f"Mean five-year rainfall: {totals['mean']:.2f} mm")
        print(
            "95% simulated five-year rainfall interval: "
            f"[{totals['predictive_low']:.2f}, "
            f"{totals['predictive_high']:.2f}] mm"
        )
        print(f"Mean five-year wet days: {wet_days['mean']:.2f}")
        print(f"Mean five-year maximum daily rainfall: {maxima['mean']:.2f} mm")

        for threshold in thresholds:
            counts = results["extreme_counts"][threshold]
            count_stats = mean_statistics(counts)
            probability = event_probability_statistics(counts)

            print(
                f"  >= {threshold:.6f} mm: "
                f"mean five-year count={count_stats['mean']:.3f}, "
                f"P(at least one in five years)="
                f"{probability['probability']:.2%}"
            )



        print("\nAnnual synthetic rainfall summary:")

        header = (
            f"{'Year':<8}"
            f"{'Mean total (mm)':>18}"
            f"{'Mean wet days':>17}"
            f"{'Mean max (mm)':>17}"
        )

        for threshold in thresholds:
            header += f"{'>=' + format(threshold, '.6f'):>16}"

        print(header)

        for year_index, year in enumerate(results["years"]):
            mean_total = np.mean(
                results["yearly_total_mm"][:, year_index]
            )
            mean_wet = np.mean(
                results["yearly_wet_days"][:, year_index]
            )
            mean_max = np.mean(
                results["yearly_maximum_mm"][:, year_index]
            )

            mean_extremes = {
                threshold: np.mean(
                    results["yearly_extreme_counts"][threshold][
                        :, year_index
                    ]
                )
                for threshold in thresholds
            }

            row = (
                f"{year:<8}"
                f"{mean_total:>18.2f}"
                f"{mean_wet:>17.2f}"
                f"{mean_max:>17.2f}"
            )

            for threshold in thresholds:
                row += f"{mean_extremes[threshold]:>16.3f}"

            print(row)


        mean_annual_sd = np.mean(
            results["annual_total_sd_mm"]
        )

        print(
            "Mean within-path annual rainfall SD: "
            f"{mean_annual_sd:.2f} mm"
        )


    print("\nPASS: five-year pre-holdout simulation completed")
    print("NOTE: All results are simulated, not observed.")


if __name__ == "__main__":
    main()
