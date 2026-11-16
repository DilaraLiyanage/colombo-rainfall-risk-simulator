#!/usr/bin/env python3
"""
preprocess_rainfall.py -- validated raw-to-daily preprocessing of ERA5-Land
hourly total precipitation for a Colombo-area grid point (about 6.9 N, 79.9 E).

PURPOSE
    Turn the raw hourly CSV (de-accumulated ERA5-Land total precipitation)
    into a validated daily rainfall series (mm per UTC day), 2001-01-01 to
    2025-12-31 inclusive. Nothing else: no wet/dry classification, no model
    fitting, no simulation.

INPUT  (never modified; opened read-only)
    CSV with columns: valid_time, tp, latitude, longitude
        valid_time : timestamp "YYYY-MM-DD HH:MM:SS", UTC, no zone marker
        tp         : hourly precipitation, metres of water equivalent [m]
        latitude   : degrees north
        longitude  : degrees east

OUTPUT (written to --output-dir)
    colombo_daily_rainfall_2001_2025.csv   columns: date, rainfall_mm
    preprocessing_quality_report.md        checks, statistics, checksums

TIME CONVENTION (ERA5-Land time-series product, see the quality report for
the documentation URLs)
    Each timestamp is the END of its one-hour accumulation interval.
        label 01:00 -> precipitation from 00:00 to 01:00
        label 00:00 -> precipitation from 23:00 of the previous day to 24:00
    Therefore:   interval_start = valid_time - 1 hour
                 UTC day        = calendar date of interval_start
    i.e. UTC day D = labels D 01:00 ... (D+1) 00:00.

UNITS
    metres [m] in the file; converted ONCE to millimetres [mm] with
    mm = m * 1000. Daily values are not rounded.

USAGE
    python preprocess_rainfall.py \
        --input  data/raw/<raw file>.csv \
        --output-dir data/processed

    Only pandas and NumPy are needed. No API key, no network access.
    Exit code 0 = every check passed; non-zero = execution failed. Candidate files are staged before publication;
    existing outputs are preserved on validation or report-writing failure.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import math
import platform
import os
import shutil
import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIGURATION (all constants in one place; historical decision labels D#)
# --------------------------------------------------------------------------
RAW_FILENAME = "reanalysis-era5-land-timeseries-sfc-pressure-precipitationycy4vwq7.csv"
EXPECTED_SHA256 = "5cca3bfc3ab1dc36a86a9232752b5ac77afa45cd4838beb7d6d706d45fa14d48"  # D2
EXPECTED_COLUMNS = ["valid_time", "tp", "latitude", "longitude"]                      # D2
EXPECTED_ROWS = 219_168
EXPECTED_FIRST_LABEL = "2001-01-01 00:00:00"   # UTC, hour-ending label
EXPECTED_LAST_LABEL = "2026-01-01 23:00:00"    # UTC, hour-ending label
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"              # D3: explicit parse format

EXPECTED_LAT_DEG = 6.9
EXPECTED_LON_DEG = 79.9
COORD_TOL_DEG = 1e-6                           # [deg]

# Project-selected tolerance informed by observed artefact magnitudes; not prescribed by ECMWF.
NEG_TOL_M = 1e-7                               # D9 [m]  (= 1e-4 mm)
M_TO_MM = 1000.0                               # D10 [mm per m], applied once

FIRST_DAY = "2001-01-01"                       # D7 first retained UTC day
LAST_DAY = "2025-12-31"                        # D7 last retained UTC day
HOURS_PER_DAY = 24

# Review flags and sanity bounds (flag only, except the hard limits)
HOURLY_REVIEW_MM = 50.0                        # [mm/h] flag for review, never removed
DAILY_REVIEW_MM = 150.0                        # [mm/day] flag for review, never removed
DAILY_HARD_MAX_MM = 1000.0                     # [mm/day] project screening bound -> FAIL
ANNUAL_MEAN_MIN_MM = 500.0                     # [mm/yr] unit-sanity gate
ANNUAL_MEAN_MAX_MM = 6000.0                    # [mm/yr] unit-sanity gate

# Numerical tolerances
AGG_TOL_MM = 1e-9                              # pandas vs NumPy daily sums [mm]
CONSERVATION_TOL_MM = 1e-6                     # total hourly vs total daily [mm]

OUT_DAILY = "colombo_daily_rainfall_2001_2025.csv"
OUT_REPORT = "preprocessing_quality_report.md"

DOC_URLS = [
    "https://confluence.ecmwf.int/spaces/CKB/pages/536218894/ERA5-Land+hourly+Analysis+Ready+Cloud+Optimised+ARCO+data+on+single+levels+from+1950+to+present+Product+User+Guide+PUG",
    "https://confluence.ecmwf.int/spaces/CKB/pages/197702790/Conversion+table+for+accumulated+variables+total+precipitation+fluxes",
    "https://confluence.ecmwf.int/spaces/UDOC/pages/208501579/Why+are+there+sometimes+small+negative+precipitation+accumulations+-+ecCodes+GRIB+FAQ",
]


# --------------------------------------------------------------------------
# PASS/FAIL machinery: print every check, fail loudly on the first failure
# --------------------------------------------------------------------------
class ValidationError(RuntimeError):
    """Raised when any check fails. Existing outputs are preserved."""


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f"  --  {detail}" if detail else ""))
        self.rows.append((status, name, detail))
        if not ok:
            raise ValidationError(f"{name}: {detail}")


# --------------------------------------------------------------------------
# Core functions (used by BOTH the real data and the synthetic tests)
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in binary chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def interval_start_day(valid_time_utc: pd.Series) -> pd.Series:
    """UTC calendar day (midnight, tz-aware) of each interval START.

    valid_time marks the END of the hour, so interval_start = valid_time - 1 h.
    """
    interval_start = valid_time_utc - pd.Timedelta(hours=1)
    return interval_start.dt.floor("D")


def daily_sums(valid_time_utc: pd.Series, values) -> pd.DataFrame:
    """Method 1 (pandas): sum values and count hours per UTC day of interval start."""
    day = pd.DatetimeIndex(interval_start_day(valid_time_utc))
    s = pd.Series(np.asarray(values, dtype="float64"), index=day)
    grouped = s.groupby(level=0)
    return pd.DataFrame({"sum_mm": grouped.sum(), "n_hours": grouped.count()})


def daily_sums_reshape(hourly_mm: np.ndarray, n_days: int, skip_leading: int) -> np.ndarray:
    """Method 2 (NumPy, positional): independent of any date arithmetic.

    The first `skip_leading` labels (here 1: label 2001-01-01 00:00, which is
    the last hour of 2000-12-31) are dropped; the next n_days*24 labels are
    reshaped to (n_days, 24) and summed along the hour axis.
    """
    stop = skip_leading + n_days * HOURS_PER_DAY
    block = np.asarray(hourly_mm[skip_leading:stop], dtype="float64")
    if block.size != n_days * HOURS_PER_DAY:
        raise ValidationError("reshape block has the wrong size")
    return block.reshape(n_days, HOURS_PER_DAY).sum(axis=1)


def hourly_index(start: str, periods: int) -> pd.Series:
    """Synthetic tz-aware hourly UTC labels (used by tests only)."""
    idx = pd.date_range(start=start, periods=periods, freq=pd.Timedelta(hours=1), tz="UTC")
    return pd.Series(idx)


def run_synthetic_boundary_tests(chk: Checks) -> None:
    """Synthetic data ONLY to test the day-assignment logic. Never written out."""

    # Test A: 2026-01-01 00:00 label belongs to 2025-12-31 (end-of-year boundary)
    t = hourly_index("2025-12-30 00:00", 24 * 3 + 1)            # ... through 2026-01-01 00:00
    v = np.zeros(len(t)); pos = int(np.where(t == pd.Timestamp("2026-01-01 00:00", tz="UTC"))[0][0]); v[pos] = 1.0
    day = interval_start_day(t)
    chk.check("Synthetic test A1: label 2026-01-01 00:00 -> day 2025-12-31",
              day.iloc[pos] == pd.Timestamp("2025-12-31", tz="UTC"), f"assigned {day.iloc[pos].date()}")
    d = daily_sums(t, v)
    chk.check("Synthetic test A2: 1 mm pulse at 2026-01-01 00:00 appears in 2025-12-31 total",
              math.isclose(d.loc[pd.Timestamp("2025-12-31", tz="UTC"), "sum_mm"], 1.0, abs_tol=1e-12)
              and d.loc[pd.Timestamp("2025-12-30", tz="UTC"), "sum_mm"] == 0.0,
              f"2025-12-31 sum = {d.loc[pd.Timestamp('2025-12-31', tz='UTC'), 'sum_mm']}")

    # Test B: 2001-01-01 01:00 belongs to 2001-01-01; 2001-01-01 00:00 belongs to 2000-12-31
    t = hourly_index("2000-12-31 00:00", 24 * 2 + 1)
    v = np.zeros(len(t))
    p01 = int(np.where(t == pd.Timestamp("2001-01-01 01:00", tz="UTC"))[0][0]); v[p01] = 1.0
    p00 = int(np.where(t == pd.Timestamp("2001-01-01 00:00", tz="UTC"))[0][0]); v[p00] = 2.0
    day = interval_start_day(t)
    chk.check("Synthetic test B1: label 2001-01-01 01:00 -> day 2001-01-01",
              day.iloc[p01] == pd.Timestamp("2001-01-01", tz="UTC"), f"assigned {day.iloc[p01].date()}")
    chk.check("Synthetic test B2: label 2001-01-01 00:00 -> day 2000-12-31 (orphan, excluded)",
              day.iloc[p00] == pd.Timestamp("2000-12-31", tz="UTC"), f"assigned {day.iloc[p00].date()}")
    d = daily_sums(t, v)
    chk.check("Synthetic test B3: pulse sums land in the correct days",
              d.loc[pd.Timestamp("2001-01-01", tz="UTC"), "sum_mm"] == 1.0
              and d.loc[pd.Timestamp("2000-12-31", tz="UTC"), "sum_mm"] == 2.0,
              "2001-01-01 = 1.0, 2000-12-31 = 2.0")

    # Test C: constant 1 mm/h gives 24 mm on every complete day
    t = hourly_index("2024-02-28 01:00", 24 * 3)                 # spans leap day 2024-02-29
    d = daily_sums(t, np.ones(len(t)))
    chk.check("Synthetic test C: constant 1 mm/h -> 24 mm on each complete day (incl. leap day)",
              bool(np.all(d["sum_mm"].to_numpy() == 24.0)) and bool(np.all(d["n_hours"].to_numpy() == 24)),
              f"{len(d)} days checked")


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def publish_outputs(pairs, staging: Path) -> None:
    """Replace validated files, restoring prior bytes on ordinary replace failures.

    Single-writer contract. Each replace is atomic, but the pair is not a
    crash-atomic transaction. Process kill/power loss during publication is
    outside this guarantee. Validation/report errors occur before this function.
    """
    backups = {}
    for i, (_, target) in enumerate(pairs):
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValidationError(f"Refusing non-regular output target: {target}")
        if target.exists():
            backup = staging / f"backup-{i}"
            shutil.copy2(target, backup)
            backups[target] = backup
    replaced = []
    try:
        for candidate, target in pairs:
            os.replace(candidate, target)
            replaced.append(target)
    except BaseException:
        for target in reversed(replaced):
            if target in backups:
                os.replace(backups[target], target)
            else:
                target.unlink(missing_ok=True)
        raise


def run(input_path: Path, output_dir: Path) -> int:
    chk = Checks()
    print("=" * 78)
    print("ERA5-Land hourly -> daily UTC rainfall preprocessing")
    print("=" * 78)

    # ---- 1. Input file exists and hash matches (D1, D2) ------------------
    if not input_path.is_file():
        raise ValidationError(f"input file not found: {input_path}")
    sha_in = sha256_file(input_path)
    chk.check("Input byte fingerprint matches expected (not provenance authentication)", sha_in == EXPECTED_SHA256, sha_in)

    # ---- 2. Load (read-only) with explicit dtypes ------------------------
    raw = pd.read_csv(
        input_path,
        dtype={"valid_time": "string", "tp": "float64", "latitude": "float64", "longitude": "float64"},
        float_precision="round_trip",   # exact decimal->binary conversion, deterministic
    )
    chk.check("CSV format: exactly the four expected columns, in order (not provenance)",
              list(raw.columns) == EXPECTED_COLUMNS, str(list(raw.columns)))
    chk.check("Row count equals expected", len(raw) == EXPECTED_ROWS, f"{len(raw)} rows")

    # ---- 3. Timestamps: explicit UTC parse (D3) --------------------------
    valid_time = pd.to_datetime(raw["valid_time"].astype("object"), format=TIME_FORMAT, utc=True, errors="raise")
    chk.check("Timestamps parsed with assumed UTC timezone (no NaT; source timezone not proved)",
              str(valid_time.dt.tz) == "UTC" and not valid_time.isna().any(), f"dtype {valid_time.dtype}")
    chk.check("Timestamps strictly increasing", bool(valid_time.is_monotonic_increasing and valid_time.is_unique))
    gaps = valid_time.diff().dropna()
    chk.check("Hourly continuity: every step is exactly 1 hour",
              bool((gaps == pd.Timedelta(hours=1)).all()), f"{len(gaps)} steps")
    first_label, last_label = valid_time.iloc[0], valid_time.iloc[-1]
    chk.check("First label as expected", first_label == pd.Timestamp(EXPECTED_FIRST_LABEL, tz="UTC"), str(first_label))
    chk.check("Last label as expected", last_label == pd.Timestamp(EXPECTED_LAST_LABEL, tz="UTC"), str(last_label))
    full = pd.date_range(first_label, last_label, freq=pd.Timedelta(hours=1))
    chk.check("Complete coverage: labels equal the full hourly range (no gaps, no extras)",
              len(full) == len(valid_time) and bool((full == pd.DatetimeIndex(valid_time)).all()),
              f"{len(full)} expected hours")

    # ---- 4. Numeric sanity (no NaN / inf), coordinates -------------------
    for col in ["tp", "latitude", "longitude"]:
        chk.check(f"No NaN or infinite values in '{col}'", bool(np.isfinite(raw[col].to_numpy()).all()))
    lat, lon = raw["latitude"].to_numpy(), raw["longitude"].to_numpy()
    chk.check("Latitude within tolerance of 6.9 deg N",
              bool(np.all(np.abs(lat - EXPECTED_LAT_DEG) < COORD_TOL_DEG)), f"range [{float(lat.min())!r}, {float(lat.max())!r}]")
    chk.check("Longitude within tolerance of 79.9 deg E",
              bool(np.all(np.abs(lon - EXPECTED_LON_DEG) < COORD_TOL_DEG)), f"range [{float(lon.min())!r}, {float(lon.max())!r}]")

    chk.check("Coordinates exactly constant across all rows",
              bool(np.all(lat == lat[0]) and np.all(lon == lon[0])),
              "Exact constancy is separate from tolerance to target coordinates")

    # ---- 5. Negative values and clipping (D9) ----------------------------
    tp_raw_m = raw["tp"].to_numpy()                      # [m], untouched
    neg = tp_raw_m < 0.0
    below_tol = tp_raw_m < -NEG_TOL_M
    n_neg = int(neg.sum())
    raw_min_m = float(tp_raw_m.min())
    chk.check(f"No value below -NEG_TOL_M (= -{NEG_TOL_M:g} m)", not bool(below_tol.any()),
              f"most negative raw value = {raw_min_m:.6e} m")
    clipped = neg & ~below_tol                           # exactly the values in [-NEG_TOL_M, 0)
    n_clipped = int(clipped.sum())
    clipped_mag_total_m = math.fsum((-tp_raw_m[clipped]).tolist())
    tp_clean_m = tp_raw_m.copy()
    tp_clean_m[clipped] = 0.0                            # only clip in [-tol, 0)
    chk.check("Cleaned precipitation is non-negative", bool((tp_clean_m >= 0.0).all()),
              f"{n_clipped} values clipped, total magnitude {clipped_mag_total_m * M_TO_MM:.6e} mm")

    # ---- 6. Unit conversion, exactly once (D10) --------------------------
    hourly_mm = tp_clean_m * M_TO_MM                     # [mm]; the ONLY conversion in the script

    # ---- 7. Synthetic boundary tests (logic only, never saved) -----------
    run_synthetic_boundary_tests(chk)

    # ---- 8. Aggregate to UTC days, method 1 (pandas) (D4, D5, D6) --------
    all_days = daily_sums(valid_time, hourly_mm)         # includes partial days at both ends
    first_day = pd.Timestamp(FIRST_DAY, tz="UTC")
    last_day = pd.Timestamp(LAST_DAY, tz="UTC")
    partial = all_days[all_days["n_hours"] != HOURS_PER_DAY]
    excluded_days = {d.strftime("%Y-%m-%d"): int(n) for d, n in partial["n_hours"].items()}
    retained = all_days[(all_days.index >= first_day) & (all_days.index <= last_day)]  # D7

    expected_days = pd.date_range(first_day, last_day, freq="D")
    n_days = len(expected_days)
    chk.check("Retained days cover every calendar date 2001-01-01..2025-12-31 exactly once",
              len(retained) == n_days and bool((retained.index == expected_days).all()), f"{n_days} days")
    chk.check("Every retained UTC day has exactly 24 hourly values",
              bool((retained["n_hours"] == HOURS_PER_DAY).all()), f"counts min/max = {retained['n_hours'].min()}/{retained['n_hours'].max()}")
    chk.check("Excluded partial days are exactly 2000-12-31 (1 h) and 2026-01-01 (23 h)",
              excluded_days == {"2000-12-31": 1, "2026-01-01": 23}, str(excluded_days))

    # Row accounting: 1 orphan + 9131*24 used + 23 trailing = all rows
    n_used = n_days * HOURS_PER_DAY
    n_orphan = excluded_days["2000-12-31"]
    n_trailing = excluded_days["2026-01-01"]
    chk.check("Row accounting: orphan + used + trailing = total rows",
              n_orphan + n_used + n_trailing == len(raw), f"{n_orphan} + {n_used} + {n_trailing} = {len(raw)}")

    # Leap-day / calendar coverage
    dates = retained.index.strftime("%Y-%m-%d")
    per_year = pd.Series(dates.str[:4]).value_counts().sort_index()
    cal_ok = all(int(per_year[str(y)]) == (366 if calendar.isleap(y) else 365) for y in range(2001, 2026))
    leap_years = [y for y in range(2001, 2026) if calendar.isleap(y)]
    leap_ok = all(f"{y}-02-29" in set(dates) for y in leap_years) and len(leap_years) == 6
    chk.check("Calendar coverage: 365/366 days per year", cal_ok, "all 25 years")
    chk.check("Leap days present (2004, 2008, 2012, 2016, 2020, 2024)", leap_ok, f"leap years: {leap_years}")

    # ---- 9. Numerical cross-check, method 2 (NumPy reshape) -----------
    skip = 1                                              # orphan label 2001-01-01 00:00
    pos_first = valid_time.iloc[skip]
    pos_last = valid_time.iloc[skip + n_used - 1]
    chk.check("Positional alignment: block starts at 2001-01-01 01:00 and ends at 2026-01-01 00:00",
              pos_first == pd.Timestamp("2001-01-01 01:00", tz="UTC") and pos_last == pd.Timestamp("2026-01-01 00:00", tz="UTC"),
              f"{pos_first} .. {pos_last}")
    daily_np = daily_sums_reshape(hourly_mm, n_days, skip)
    max_diff = float(np.max(np.abs(daily_np - retained["sum_mm"].to_numpy())))
    chk.check(f"Numerical aggregation cross-check (shared preprocessing) agrees with pandas within {AGG_TOL_MM:g} mm",
              max_diff <= AGG_TOL_MM, f"max |difference| = {max_diff:.3e} mm")

    # ---- 10. Conservation of precipitation -------------------------------
    used_hourly = hourly_mm[skip:skip + n_used]
    total_hourly_mm = math.fsum(used_hourly.tolist())
    total_daily_mm = math.fsum(retained["sum_mm"].tolist())
    chk.check(f"Conservation: sum(hourly used) = sum(daily) within {CONSERVATION_TOL_MM:g} mm",
              abs(total_hourly_mm - total_daily_mm) <= CONSERVATION_TOL_MM,
              f"hourly {total_hourly_mm:.9f} mm, daily {total_daily_mm:.9f} mm, diff {abs(total_hourly_mm - total_daily_mm):.3e}")

    # ---- 11. Bounds, unit gate, convention evidence, review flags --------
    daily_mm = retained["sum_mm"].to_numpy()
    chk.check("Daily totals >= 0", bool((daily_mm >= 0.0).all()), f"min = {daily_mm.min():.9g} mm")
    chk.check(f"Daily totals < {DAILY_HARD_MAX_MM:g} mm (screening bound, not physical validation)", bool((daily_mm < DAILY_HARD_MAX_MM).all()),
              f"max = {daily_mm.max():.6f} mm")
    n_complete_years = last_day.year - first_day.year + 1
    mean_annual_mm = total_daily_mm / n_complete_years
    chk.check(f"Plausibility screen: mean annual total in [{ANNUAL_MEAN_MIN_MM:g}, {ANNUAL_MEAN_MAX_MM:g}] mm (does not verify units)",
              ANNUAL_MEAN_MIN_MM <= mean_annual_mm <= ANNUAL_MEAN_MAX_MM, f"{mean_annual_mm:.1f} mm/yr")
    hour01 = (valid_time.dt.hour.to_numpy() == 1)
    chk.check("Supporting pattern only: label 01:00 never negative (does not prove convention)",
              bool((tp_raw_m[hour01] >= 0.0).all()), f"{int(hour01.sum())} values at 01:00 UTC")

    hourly_used_series = pd.Series(used_hourly, index=pd.DatetimeIndex(valid_time.iloc[skip:skip + n_used]))
    hourly_flag = hourly_used_series[hourly_used_series > HOURLY_REVIEW_MM]
    hourly_max_ts = hourly_used_series.idxmax()
    daily_series = retained["sum_mm"].copy(); daily_series.index = dates
    daily_flag = daily_series[daily_series > DAILY_REVIEW_MM].sort_values(ascending=False)
    print(f"[INFO] Hourly values above {HOURLY_REVIEW_MM:g} mm flagged for review: {len(hourly_flag)} "
          f"(max {used_hourly.max():.4f} mm at label {hourly_max_ts})")
    print(f"[INFO] Daily totals above {DAILY_REVIEW_MM:g} mm flagged for review (kept in data): {len(daily_flag)}")

    # ---- 12. Stage and validate candidate outputs before replacing anything
    # ---- Write daily CSV (D11: no rounding) --------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    out_daily = output_dir / OUT_DAILY
    out_report = output_dir / OUT_REPORT
    out = pd.DataFrame({"date": dates, "rainfall_mm": daily_mm})
    with tempfile.TemporaryDirectory(prefix=".rainfall-stage-", dir=output_dir) as stage_dir:
        staging = Path(stage_dir)
        candidate_daily = staging / OUT_DAILY
        candidate_report = staging / OUT_REPORT
        out.to_csv(candidate_daily, index=False, lineterminator="\n")   # "\n" -> same bytes on every OS
        back = pd.read_csv(candidate_daily, dtype={"date": "string"}, float_precision="round_trip")
        chk.check("Output round-trip: re-read file equals in-memory daily values exactly",
                  list(back.columns) == ["date", "rainfall_mm"] and len(back) == n_days
                  and bool(np.array_equal(back["rainfall_mm"].to_numpy(), daily_mm))
                  and list(back["date"].astype("object")) == list(dates),
                  f"{len(back)} rows")
        sha_out = sha256_file(candidate_daily)
        chk.check("Input file unchanged after processing (hash re-check)", sha256_file(input_path) == EXPECTED_SHA256)
        print(f"[INFO] Daily output SHA-256: {sha_out}")

        # ---- 13. Quality report ---------------------------------------------
        stats = dict(
            n_days=n_days, first=dates[0], last=dates[-1],
            min_mm=float(daily_mm.min()), min_date=dates[int(daily_mm.argmin())],
            max_mm=float(daily_mm.max()), max_date=dates[int(daily_mm.argmax())],
            total_mm=total_daily_mm, mean_mm=float(daily_mm.mean()),
            median_mm=float(np.median(daily_mm)), std_mm=float(daily_mm.std(ddof=1)),
            mean_annual_mm=mean_annual_mm,
        )
        annual = daily_series.groupby(daily_series.index.str[:4]).sum()
        n_neg_used = int((tp_raw_m[skip:skip + n_used] < 0).sum())
        clip_used_mm = math.fsum((-tp_raw_m[skip:skip + n_used][tp_raw_m[skip:skip + n_used] < 0]).tolist()) * M_TO_MM
        write_report(
            candidate_report, chk, sha_in, sha_out, input_path, out_daily, len(raw), first_label, last_label,
            n_orphan, n_used, n_trailing, n_neg, n_clipped, n_neg_used, raw_min_m, clipped_mag_total_m,
            clip_used_mm, stats, annual, daily_flag, hourly_flag, float(used_hourly.max()), hourly_max_ts,
        )
        report_text = candidate_report.read_text(encoding="utf-8")
        if sha_out not in report_text or not all(name in report_text for _, name, _ in chk.rows):
            raise ValidationError("Candidate report read-back failed")
        publish_outputs([(candidate_daily, out_daily), (candidate_report, out_report)], staging)
    print(f"[INFO] Report written: {out_report}")
    print("=" * 78)
    print(f"ALL {len(chk.rows)} CHECKS PASSED")
    print("=" * 78)
    return 0


def write_report(path, chk, sha_in, sha_out, input_path, out_daily, n_rows, first_label, last_label,
                 n_orphan, n_used, n_trailing, n_neg, n_clipped, n_neg_used, raw_min_m, clipped_mag_total_m,
                 clip_used_mm, stats, annual, daily_flag, hourly_flag, hourly_max_mm, hourly_max_ts) -> None:
    """Write the Markdown quality report (deterministic: no run timestamp)."""
    L: list[str] = []
    a = L.append
    a("# Preprocessing quality report: ERA5-Land hourly -> daily UTC rainfall\n")
    a(f"**Status: ALL {len(chk.rows)} CHECKS PASSED** (script `preprocess_rainfall.py`; "
      f"Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__})\n")

    a("## 1. Input file\n")
    a(f"- File name: `{input_path.name}`")
    a(f"- SHA-256: `{sha_in}` (matches the expected value; re-checked after processing, file unchanged)")
    a(f"- Rows: {n_rows:,}; columns: `valid_time, tp, latitude, longitude`")
    a(f"- Timestamp range (hour-ending labels, UTC): {first_label} to {last_label}")
    a("- Grid point: latitude 6.9, longitude 79.9 (within 1e-6 degrees of the target; exact constancy checked separately)")
    a("- Missing values: none. Duplicate or missing hourly timestamps: none.\n")

    a("## 2. Source information\n")
    a("- Provider/product: Copernicus Climate Change Service (C3S) / ECMWF, ERA5-Land hourly time-series data. "
      "This is **inferred** from the file-name pattern and the documented time-series conventions; "
      "the CSV itself carries no metadata.")
    a("- CDS request record, download date, dataset version/DOI and licence text: **not recorded in the file. "
      "To be completed by the author from the CDS download record before submission.**")
    a("- This is a reanalysis product (model-based, ~0.1 degree grid), not a rain-gauge record.\n")

    a("## 3. Conventions and decisions applied\n")
    a("- Units: `tp` is metres of water equivalent per hour (de-accumulated); converted once to mm (x 1000). "
      "Units are taken from the documented product; the plausibility screen cannot verify them. Source attribution remains conditional on the outstanding provenance record.")
    a("- Timestamps assigned UTC under the documented-product assumption; parsing does not prove the source timezone. Each timestamp is the END of its one-hour interval; "
      "`interval_start = valid_time - 1 h`; day = UTC calendar date of `interval_start` "
      "(UTC day D = labels D 01:00 ... D+1 00:00; 00:00-24:00 UTC = 05:30-05:30 Sri Lanka time).")
    a(f"- Retained days: 2001-01-01 to 2025-12-31 inclusive, complete days only (exactly 24 hourly intervals).")
    a(f"- Excluded: label 2001-01-01 00:00 ({n_orphan} row; it is the last hour of 2000-12-31) and labels "
      f"2026-01-01 01:00-23:00 ({n_trailing} rows; they belong to the incomplete day 2026-01-01). "
      "Label 2026-01-01 00:00 IS used: it is the final hour of 2025-12-31.")
    a(f"- Row accounting: {n_orphan} + {n_used:,} + {n_trailing} = {n_orphan + n_used + n_trailing:,} rows.")
    a(f"- Negative values: tolerance `NEG_TOL_M` = {NEG_TOL_M:g} m (0.0001 mm). Values in [-1e-7, 0) m clipped to 0; "
      "any value below -1e-7 m would stop the script.")
    a("- The 1e-7 m clipping tolerance is project-selected from observed numerical artefacts, not prescribed by ECMWF; original GRIB packing-error metadata is unavailable.\n")
    a("- Daily totals are not rounded. Wet/dry classification, fitting and simulation are NOT part of this step.\n")

    a("## 4. Negative values and clipping\n")
    a(f"- Negative raw values in the whole file: {n_neg:,}; all clipped (all lay within the tolerance).")
    a(f"- Most negative raw value: {raw_min_m:.6e} m ({raw_min_m * M_TO_MM:.6e} mm).")
    a(f"- Total magnitude removed by clipping (whole file): {clipped_mag_total_m * M_TO_MM:.6e} mm.")
    a(f"- Among the {n_used:,} hours actually used: {n_neg_used:,} negatives clipped, "
      f"net effect on the total {clip_used_mm:.6e} mm (clipping only ever adds this amount).")
    a("- Rationale: ECMWF documents GRIB packing as a mechanism for small negative values after de-accumulation and "
      "advises setting them to zero (see references). Spurious small *positive* values of similar size "
      "cannot be identified and remain in the data.\n")

    a("## 5. Daily series statistics\n")
    a(f"- Daily records: **{stats['n_days']:,}** ({stats['first']} to {stats['last']})")
    a(f"- Minimum daily rainfall: {stats['min_mm']:.9g} mm ({stats['min_date']})")
    a(f"- Maximum daily rainfall: {stats['max_mm']:.6f} mm ({stats['max_date']})")
    a(f"- Total rainfall: {stats['total_mm']:.6f} mm; mean daily {stats['mean_mm']:.6f} mm; "
      f"median {stats['median_mm']:.6f} mm; std {stats['std_mm']:.6f} mm")
    a(f"- Mean annual total (25 complete calendar years): {stats['mean_annual_mm']:.10f} mm/yr\n")
    a("| Year | Total (mm) |\n|---|---|")
    for y, v in annual.items():
        a(f"| {y} | {v:.1f} |")
    a("")

    a("## 6. Validation checks\n")
    a("| # | Status | Check | Detail |\n|---|---|---|---|")
    for i, (s, n, d) in enumerate(chk.rows, 1):
        a(f"| {i} | {s} | {n} | {d.replace('|', '/')} |")
    a("")

    a("## 7. Flags for review (nothing removed)\n")
    a(f"- Daily totals above {DAILY_REVIEW_MM:g} mm (a review threshold chosen for this report, "
      f"not a statistical definition of 'extreme'): {len(daily_flag)}")
    for d, v in daily_flag.items():
        a(f"  - {d}: {v:.3f} mm")
    a(f"- Hourly values above {HOURLY_REVIEW_MM:g} mm: {len(hourly_flag)}; maximum hourly value "
      f"{hourly_max_mm:.4f} mm at label {hourly_max_ts} (UTC, hour-ending).")
    a("- These days should be compared with independent sources before being treated as genuine or erroneous.\n")

    a("## 8. Output files and checksums\n")
    a(f"- `{out_daily.name}`: columns `date, rainfall_mm`; {stats['n_days']:,} rows; SHA-256 `{sha_out}`")
    a(f"- Input: `{input_path.name}` SHA-256 `{sha_in}`\n")

    a("- Validation checks establish file integrity and numerical consistency under stated assumptions; they do not authenticate provenance or physically validate rainfall. See provenance_manifest.md for outstanding source fields.\n")
    a("- Output safety: CSV and report are staged in a temporary directory, validated before replacement, and removed on failure. Existing files survive pre-publication failures. Ordinary replacement failures trigger rollback; concurrent writers and process/power loss during the two-file publication are outside this guarantee.\n")
    a("## 9. Limitations and anomalies\n")
    a("- Reanalysis precipitation can differ from gauge observations, particularly in the tails; "
      "this step makes no comparison with gauges.")
    a("- The day is a UTC day, not a local day (local midnight falls mid-interval in the hourly data).")
    a("- The convention check on label 01:00 is indirect evidence; the primary basis is the ECMWF documentation.")
    a("- The grid cell's land/sea character and representativeness for Colombo were not checked here.")
    a("- Dataset citation, licence terms and download record are not in the file and must be added by the author.")
    a(f"- Anomalies in this run: orphan/trailing rows excluded as designed; {n_neg:,} tiny negatives clipped; "
      f"{len(daily_flag)} days above the review threshold. No failed checks.\n")

    a("## 10. References (documentation consulted)\n")
    for u in DOC_URLS:
        a(f"- {u}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validated ERA5-Land hourly -> daily UTC rainfall preprocessing.")
    ap.add_argument("--input", type=Path, default=Path("data/raw") / RAW_FILENAME,
                    help="path to the raw hourly CSV (default: data/raw/<raw file name>)")
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed"),
                    help="directory for the daily CSV and quality report (default: data/processed)")
    args = ap.parse_args()
    try:
        return run(args.input, args.output_dir)
    except (ValidationError, OSError, ValueError) as exc:
        print("=" * 78)
        print(f"PREPROCESSING FAILED: {exc}")
        print("=" * 78)
        return 1


if __name__ == "__main__":
    sys.exit(main())
