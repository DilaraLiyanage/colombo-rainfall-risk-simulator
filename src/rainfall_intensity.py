
"""A1: Left-truncated Gamma rainfall intensity model.

Fits wet-day rainfall amounts (>= 1 mm) from the
2001–2020 calibration dataset only.

No access to the 2021–2025 reserved holdout observations.
"""

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma

from rainfall_occurrence import load_calibration

WET_THRESHOLD_MM = 1.0


def validate_wet_amounts(amounts, threshold=WET_THRESHOLD_MM):
    """Validate rainfall observations used for intensity fitting."""
    values = np.asarray(amounts, dtype=float)

    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two wet-day amounts are required")

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid wet-day threshold")

    if not np.all(np.isfinite(values)):
        raise ValueError("Rainfall values must be finite")

    if np.any(values < threshold):
        raise ValueError("Wet-day amounts must meet the threshold")

    return values


def truncated_gamma_logpdf(values, shape, scale,
                           threshold=WET_THRESHOLD_MM):
    """Conditional Gamma log-density above the wet-day threshold."""
    values = validate_wet_amounts(values, threshold)

    if not np.isfinite(shape) or shape <= 0:
        raise ValueError("Gamma shape must be positive")

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Gamma scale must be positive")

    log_survival = gamma.logsf(
        threshold,
        a=shape,
        loc=0,
        scale=scale,
    )

    if not math.isfinite(log_survival):
        raise ValueError("Invalid Gamma truncation probability")

    return (
        gamma.logpdf(values, a=shape, loc=0, scale=scale)
        - log_survival
    )


def fit_truncated_gamma(amounts, threshold=WET_THRESHOLD_MM):
    """Fit a left-truncated Gamma by numerical maximum likelihood."""
    values = validate_wet_amounts(amounts, threshold)

    mean = float(np.mean(values))
    variance = float(np.var(values))

    initial_shape = max(mean * mean / max(variance, 1e-8), 0.1)
    initial_scale = max(variance / max(mean, 1e-8), 0.1)

    def negative_log_likelihood(log_parameters):
        shape, scale = np.exp(log_parameters)

        try:
            log_density = truncated_gamma_logpdf(
                values,
                shape,
                scale,
                threshold,
            )
        except ValueError:
            return float("inf")

        total = -float(np.sum(log_density))

        return total if math.isfinite(total) else float("inf")

    result = minimize(
        negative_log_likelihood,
        x0=np.log([initial_shape, initial_scale]),
        method="Nelder-Mead",
        options={
            "maxiter": 10000,
            "xatol": 1e-10,
            "fatol": 1e-8,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Gamma optimization failed: {result.message}"
        )

    shape, scale = np.exp(result.x)

    if not np.all(np.isfinite([shape, scale])):
        raise RuntimeError("Non-finite fitted Gamma parameters")

    return {
        "shape": float(shape),
        "scale": float(scale),
        "threshold_mm": float(threshold),
        "n_wet_days": len(values),
        "log_likelihood": -float(result.fun),
        "optimization_success": bool(result.success),
    }


# ------------------------------------------------------------
# A2: Shifted two-component exponential mixture
# ------------------------------------------------------------

from scipy.special import expit, logsumexp


def mixed_exponential_logpdf(
    amounts,
    weight_large,
    scale_small,
    scale_large,
    threshold=WET_THRESHOLD_MM,
):
    """Log-density of the shifted exponential mixture.

    Rainfall X = threshold + Y, where Y is drawn from
    a mixture of two exponential distributions.

    weight_large is the probability of choosing the
    larger-scale exponential component.
    """
    values = validate_wet_amounts(amounts, threshold)

    parameters = (
        weight_large,
        scale_small,
        scale_large,
    )

    if not all(math.isfinite(float(p)) for p in parameters):
        raise ValueError("Mixture parameters must be finite")

    if not 0 < weight_large < 1:
        raise ValueError("Mixture weight must lie between 0 and 1")

    if not 0 < scale_small < scale_large:
        raise ValueError("Expected 0 < small scale < large scale")

    excess = values - threshold

    log_small = (
        math.log1p(-weight_large)
        - math.log(scale_small)
        - excess / scale_small
    )

    log_large = (
        math.log(weight_large)
        - math.log(scale_large)
        - excess / scale_large
    )

    return logsumexp(
        np.stack([log_small, log_large]),
        axis=0,
    )


def fit_mixed_exponential(
    amounts,
    threshold=WET_THRESHOLD_MM,
):
    """Fit A2 using multistart maximum likelihood.

    Scales are kept ordered to prevent label switching.
    Several deterministic initializations reduce dependence
    on a single optimizer starting point.
    """
    values = validate_wet_amounts(amounts, threshold)
    excess = values - threshold

    mean_excess = float(np.mean(excess))

    if mean_excess <= 0:
        raise ValueError("Positive rainfall excess is required")

    # Parameters: log(small scale), log(large-small gap),
    # logit(large-component weight).
    def unpack(parameters):
        small = math.exp(float(parameters[0]))
        large = small + math.exp(float(parameters[1]))
        weight = float(expit(parameters[2]))
        return weight, small, large

    def objective(parameters):
        weight, small, large = unpack(parameters)

        log_density = mixed_exponential_logpdf(
            values,
            weight,
            small,
            large,
            threshold,
        )

        nll = -float(np.sum(log_density))

        return nll if math.isfinite(nll) else 1e300

    starts = [
        (0.5, 2.0, 0.05),
        (0.75, 3.0, 0.10),
        (1.0, 4.0, 0.20),
        (0.35, 5.0, 0.30),
        (1.0, 8.0, 0.50),
    ]

    bounds = [
        (-12, 12),   # log small scale
        (-12, 12),   # log positive scale gap
        (-12, 12),   # logit mixture weigh
    ]

    best = None

    for small_factor, large_factor, weight in starts:
        small = max(mean_excess * small_factor, 1e-4)
        large = max(mean_excess * large_factor, small + 1e-4)

        start = np.array([
            math.log(small),
            math.log(large - small),
            math.log(weight / (1 - weight)),
        ])

        result = minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-12},
        )

        if result.success and math.isfinite(float(result.fun)):
            if best is None or result.fun < best.fun:
                best = result

    if best is None:
        raise RuntimeError("A2 optimization failed")

    weight, small, large = unpack(best.x)

    return {
        "weight_large": weight,
        "scale_small": small,
        "scale_large": large,
        "threshold_mm": float(threshold),
        "n_wet_days": len(values),
        "log_likelihood": -float(best.fun),
        "optimization_success": True,
    }


def print_a2_results(wet_amounts):
    """Fit and display the A2 model."""
    result = fit_mixed_exponential(wet_amounts)

    print("\nMODEL: A2 — Shifted Mixed Exponential")
    print("Wet-day observations:", result["n_wet_days"])
    print("Threshold:", result["threshold_mm"], "mm")
    print("Large-component weight:",
          f"{result['weight_large']:.12f}")
    print("Small-component scale:",
          f"{result['scale_small']:.12f}")
    print("Large-component scale:",
          f"{result['scale_large']:.12f}")
    print("Log-likelihood:",
          f"{result['log_likelihood']:.6f}")

    print("\nPASS: A2 intensity model fitted successfully")


def main():
    dates, rainfall = load_calibration()

    wet_amounts = [
        amount for amount in rainfall
        if amount >= WET_THRESHOLD_MM
    ]

    result = fit_truncated_gamma(wet_amounts)

    print("MODEL: A1 — Left-Truncated Gamma")
    print("Calibration:", dates[0], "to", dates[-1])
    print("Wet-day threshold:", WET_THRESHOLD_MM, "mm")
    print("Wet-day observations:", result["n_wet_days"])

    print("Shape:", f"{result['shape']:.12f}")
    print("Scale:", f"{result['scale']:.12f}")
    print(
        "Log-likelihood:",
        f"{result['log_likelihood']:.6f}",
    )

    print("\nPASS: A1 intensity model fitted successfully")
    print_a2_results(wet_amounts)


if __name__ == "__main__":
    main()
