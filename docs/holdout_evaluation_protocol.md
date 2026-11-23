
# SE4042 — Frozen Independent Evaluation Protocol

## Projec

Stochastic Simulation of Monsoon Rainfall and Extreme Rainfall Risk in Colombo

Protocol status: PRE-HOLDOUT — PROPOSED FOR FREEZING
Protocol date: 2026-11-23

## 1. Objective

Evaluate whether the calibration-fitted stochastic models reproduce
daily rainfall occurrence, cumulative rainfall, annual variability,
and extreme rainfall behavior during an independent 2021–2025 period.

The holdout observations must not be used for model fitting,
hyperparameter selection, or retrospective modification of the
candidate models.

## 2. Dataset partitions

Calibration:
- Start: 2001-01-01
- End: 2020-12-31
- Records: 7,305 complete daily observations
- Purpose: model fitting, development, and model diagnostics

Internal chronological validation:
- Training: 2001–2015
- Validation: 2016–2020
- Purpose: evaluate occurrence-model predictive behavior

Independent holdout:
- Start: 2021-01-01
- End: 2025-12-31
- Expected calendar days: 1,826
- Purpose: final independent evaluation only

Calendar-day completeness and the final holdout record count mus
be verified before evaluation. No missing days may be silently
discarded.

The established UTC daily aggregation and precipitation cleaning
rules must remain unchanged.

## 3. Frozen candidate models

M1 = B1 + A1
- B1: homogeneous first-order wet/dry Markov chain
- A1: left-truncated Gamma wet-day rainfall intensity

M2 = B2 + A1
- B2: four-season first-order wet/dry Markov chain
- A1: left-truncated Gamma wet-day rainfall intensity

M3 = B2 + A2
- B2: four-season first-order wet/dry Markov chain
- A2: shifted two-component exponential wet-day intensity

B3, the monthly occurrence model, is an occurrence-model challenger
already assessed during internal calibration validation. It is no
a member of the three predefined full-simulation candidates.

Wet-day classification:
- Wet if daily rainfall >= 1.0 mm
- Dry if daily rainfall < 1.0 mm

Dry-day rainfall:
- Resample observed calibration amounts below 1.0 mm
- Do not automatically replace all dry amounts with zero

Seasonal transition convention:
- Use the season of the current simulated day
- At season boundaries, retain the previous simulated state

Initial state:
- Sample using the wet-day frequency from calibration

All fitted parameters must be estimated exclusively using
2001–2020 observations.

## 4. Simulation design

Replications: 2,000 per model
Base random seed: 2026111902

Random streams:
- Use the existing deterministic NumPy SeedSequence scheme
- M1: SeedSequence([2026111902, 0])
- M2: SeedSequence([2026111902, 1])
- M3: SeedSequence([2026111902, 2])

Each replication must simulate the complete sequence of calendar
dates from 2021-01-01 through 2025-12-31 (1,826 days).

The simulation must not use observed holdout rainfall to
initialize states, fit parameters, or drive daily transitions.

Each synthetic day must depend on the previous simulated state.

The earlier 2027 single-year and repeated-year experiments are
development diagnostics, not the independent holdout evaluation.

## 5. Predefined evaluation metrics

Daily occurrence:
- Wet-day proportion
- Total wet days
- Optional descriptive monthly and seasonal wet-day frequencies

Rainfall amount:
- Total rainfall over the holdout period
- Mean annual rainfall using complete calendar years
- Annual rainfall standard deviation
- Maximum daily rainfall

Extreme rainfall:
- Counts of days with rainfall >= 50 mm
- Counts of days with rainfall >= 100 mm
- Counts of days with rainfall >= 150 mm

Additional calibration-defined thresholds:
- 99th percentile: approximately 52.437780 mm
- 99.5th percentile: approximately 70.923264 mm

These percentile thresholds were derived from the calibration data
and must not be recalculated using the holdout observations.

For each threshold, report:
- Observed number of exceedance days
- Mean simulated number of exceedance days
- 95% simulated interval for the exceedance coun
- Simulated probability of at least one exceedance over the
  complete five-year holdout period

## 6. Statistical reporting

For simulation-derived statistics, distinguish:

A. Monte Carlo mean:
   Average result across the 2,000 simulated paths.

B. Monte Carlo confidence interval:
   Uncertainty in estimating the model-implied mean.

C. Simulation predictive interval:
   Variation across individual simulated five-year paths.

D. Observed holdout statistic:
   Statistic calculated directly from the 2021–2025 data.

Compare observed holdout statistics with simulated distributions.

A confidence interval for the Monte Carlo mean must not be
interpreted as an interval for actual rainfall.

The simulated intervals are conditional on fitted model
parameters and do not account for all structural or
climate-related uncertainties.

## 7. Model evaluation rules

- Report every predefined candidate model: M1, M2, and M3.
- Do not select a model solely because it produces the mos
  extreme simulated event.
- Do not refit parameters on the holdout period.
- Do not change thresholds after seeing holdout results.
- Do not modify the three candidate definitions retrospectively.
- Do not claim statistically significant superiority withou
  a suitable uncertainty analysis.
- Report underprediction and overprediction transparently.
- Treat very rare events cautiously because observed even
  counts may be small.

Model assessment should consider occurrence behavior, annual
rainfall, annual variability, and extreme-event performance
together rather than declaring a winner using one metric.

## 8. Model limitations to disclose

Known calibration-stage concerns include:

- Potential underestimation of rainfall >= 150 mm
- Insufficient representation of annual rainfall variability
- Assumption of a pooled wet-day intensity distribution
- Limited ability of a first-order Markov chain to reproduce
  complex rainfall dependence
- Sampling uncertainty associated with rare extreme events
- Parameter estimates treated as fixed during simulation

The independent evaluation will assess these limitations
without modifying the original candidates.

## 9. Software verification prerequisites

Before holdout access:

- Full synthetic/calibration-safe regression suite must pass.
- Existing result: 118 tests discovered, 110 passed,
  8 raw-data-dependent tests skipped, 0 failures.
- Validate simulation dates and record counts.
- Verify reproducibility for fixed seeds.
- Verify that simulated states agree with rainfall values.
- Verify that vectorized model outputs are consistent with
  the intended probabilistic models.
- Record source-code revision and dependency versions.

## 10. Holdout access rule

Do not inspect holdout rainfall values until:

1. This protocol is finalized.
2. Evaluation code and synthetic tests are ready.
3. All predefined thresholds and metrics are recorded.
4. Model source files and parameters are frozen.
5. The frozen revision is recorded.

After the holdout is opened, any necessary correction must be
documented as a correction, not presented as an original
pre-holdout decision.

## 11. Academic reporting

Clearly distinguish:
- Calibration fi
- Internal chronological validation
- Simulation-based diagnostics
- Independent holdout evaluation

Document limitations, numerical methods, reproducibility
settings, and actual AI-tool assistance according to the
module's academic-integrity requirements.