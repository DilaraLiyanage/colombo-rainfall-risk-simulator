# Preprocessing quality report: ERA5-Land hourly -> daily UTC rainfall

**Status: ALL 38 CHECKS PASSED** (script `preprocess_rainfall.py`; Python 3.12.14, pandas 2.2.3, NumPy 2.3.5)

## 1. Input file

- File name: `reanalysis-era5-land-timeseries-sfc-pressure-precipitationycy4vwq7.csv`
- SHA-256: `5cca3bfc3ab1dc36a86a9232752b5ac77afa45cd4838beb7d6d706d45fa14d48` (matches the expected value; re-checked after processing, file unchanged)
- Rows: 219,168; columns: `valid_time, tp, latitude, longitude`
- Timestamp range (hour-ending labels, UTC): 2001-01-01 00:00:00+00:00 to 2026-01-01 23:00:00+00:00
- Grid point: latitude 6.9, longitude 79.9 (within 1e-6 degrees of the target; exact constancy checked separately)
- Missing values: none. Duplicate or missing hourly timestamps: none.

## 2. Source information

- Provider/product: Copernicus Climate Change Service (C3S) / ECMWF, ERA5-Land hourly time-series data. This is **inferred** from the file-name pattern and the documented time-series conventions; the CSV itself carries no metadata.
- CDS request record, download date, dataset version/DOI and licence text: **not recorded in the file. To be completed by the author from the CDS download record before submission.**
- This is a reanalysis product (model-based, ~0.1 degree grid), not a rain-gauge record.

## 3. Conventions and decisions applied

- Units: `tp` is metres of water equivalent per hour (de-accumulated); converted once to mm (x 1000). Units are taken from the documented product; the plausibility screen cannot verify them. Source attribution remains conditional on the outstanding provenance record.
- Timestamps assigned UTC under the documented-product assumption; parsing does not prove the source timezone. Each timestamp is the END of its one-hour interval; `interval_start = valid_time - 1 h`; day = UTC calendar date of `interval_start` (UTC day D = labels D 01:00 ... D+1 00:00; 00:00-24:00 UTC = 05:30-05:30 Sri Lanka time).
- Retained days: 2001-01-01 to 2025-12-31 inclusive, complete days only (exactly 24 hourly intervals).
- Excluded: label 2001-01-01 00:00 (1 row; it is the last hour of 2000-12-31) and labels 2026-01-01 01:00-23:00 (23 rows; they belong to the incomplete day 2026-01-01). Label 2026-01-01 00:00 IS used: it is the final hour of 2025-12-31.
- Row accounting: 1 + 219,144 + 23 = 219,168 rows.
- Negative values: tolerance `NEG_TOL_M` = 1e-07 m (0.0001 mm). Values in [-1e-7, 0) m clipped to 0; any value below -1e-7 m would stop the script.
- The 1e-7 m clipping tolerance is project-selected from observed numerical artefacts, not prescribed by ECMWF; original GRIB packing-error metadata is unavailable.

- Daily totals are not rounded. Wet/dry classification, fitting and simulation are NOT part of this step.

## 4. Negative values and clipping

- Negative raw values in the whole file: 1,363; all clipped (all lay within the tolerance).
- Most negative raw value: -3.911555e-08 m (-3.911555e-05 mm).
- Total magnitude removed by clipping (whole file): 1.089528e-02 mm.
- Among the 219,144 hours actually used: 1,363 negatives clipped, net effect on the total 1.089528e-02 mm (clipping only ever adds this amount).
- Rationale: ECMWF documents GRIB packing as a mechanism for small negative values after de-accumulation and advises setting them to zero (see references). Spurious small *positive* values of similar size cannot be identified and remain in the data.

## 5. Daily series statistics

- Daily records: **9,131** (2001-01-01 to 2025-12-31)
- Minimum daily rainfall: 0 mm (2001-01-06)
- Maximum daily rainfall: 221.967459 mm (2016-05-15)
- Total rainfall: 72158.451470 mm; mean daily 7.902579 mm; median 4.269266 mm; std 11.271863 mm
- Mean annual total (25 complete calendar years): 2886.3380587974 mm/yr

| Year | Total (mm) |
|---|---|
| 2001 | 2534.5 |
| 2002 | 2931.9 |
| 2003 | 2906.7 |
| 2004 | 3243.2 |
| 2005 | 2733.0 |
| 2006 | 3555.0 |
| 2007 | 2805.7 |
| 2008 | 3381.1 |
| 2009 | 2910.0 |
| 2010 | 3260.1 |
| 2011 | 2785.6 |
| 2012 | 2757.7 |
| 2013 | 2679.6 |
| 2014 | 3027.6 |
| 2015 | 3215.8 |
| 2016 | 2365.4 |
| 2017 | 2596.5 |
| 2018 | 2763.1 |
| 2019 | 2881.0 |
| 2020 | 2241.3 |
| 2021 | 3006.0 |
| 2022 | 2444.5 |
| 2023 | 3389.9 |
| 2024 | 3066.9 |
| 2025 | 2676.4 |

## 6. Validation checks

| # | Status | Check | Detail |
|---|---|---|---|
| 1 | PASS | Input byte fingerprint matches expected (not provenance authentication) | 5cca3bfc3ab1dc36a86a9232752b5ac77afa45cd4838beb7d6d706d45fa14d48 |
| 2 | PASS | CSV format: exactly the four expected columns, in order (not provenance) | ['valid_time', 'tp', 'latitude', 'longitude'] |
| 3 | PASS | Row count equals expected | 219168 rows |
| 4 | PASS | Timestamps parsed with assumed UTC timezone (no NaT; source timezone not proved) | dtype datetime64[ns, UTC] |
| 5 | PASS | Timestamps strictly increasing |  |
| 6 | PASS | Hourly continuity: every step is exactly 1 hour | 219167 steps |
| 7 | PASS | First label as expected | 2001-01-01 00:00:00+00:00 |
| 8 | PASS | Last label as expected | 2026-01-01 23:00:00+00:00 |
| 9 | PASS | Complete coverage: labels equal the full hourly range (no gaps, no extras) | 219168 expected hours |
| 10 | PASS | No NaN or infinite values in 'tp' |  |
| 11 | PASS | No NaN or infinite values in 'latitude' |  |
| 12 | PASS | No NaN or infinite values in 'longitude' |  |
| 13 | PASS | Latitude within tolerance of 6.9 deg N | range [6.900000000000828, 6.900000000000828] |
| 14 | PASS | Longitude within tolerance of 79.9 deg E | range [79.89999999999975, 79.89999999999975] |
| 15 | PASS | Coordinates exactly constant across all rows | Exact constancy is separate from tolerance to target coordinates |
| 16 | PASS | No value below -NEG_TOL_M (= -1e-07 m) | most negative raw value = -3.911555e-08 m |
| 17 | PASS | Cleaned precipitation is non-negative | 1363 values clipped, total magnitude 1.089528e-02 mm |
| 18 | PASS | Synthetic test A1: label 2026-01-01 00:00 -> day 2025-12-31 | assigned 2025-12-31 |
| 19 | PASS | Synthetic test A2: 1 mm pulse at 2026-01-01 00:00 appears in 2025-12-31 total | 2025-12-31 sum = 1.0 |
| 20 | PASS | Synthetic test B1: label 2001-01-01 01:00 -> day 2001-01-01 | assigned 2001-01-01 |
| 21 | PASS | Synthetic test B2: label 2001-01-01 00:00 -> day 2000-12-31 (orphan, excluded) | assigned 2000-12-31 |
| 22 | PASS | Synthetic test B3: pulse sums land in the correct days | 2001-01-01 = 1.0, 2000-12-31 = 2.0 |
| 23 | PASS | Synthetic test C: constant 1 mm/h -> 24 mm on each complete day (incl. leap day) | 3 days checked |
| 24 | PASS | Retained days cover every calendar date 2001-01-01..2025-12-31 exactly once | 9131 days |
| 25 | PASS | Every retained UTC day has exactly 24 hourly values | counts min/max = 24/24 |
| 26 | PASS | Excluded partial days are exactly 2000-12-31 (1 h) and 2026-01-01 (23 h) | {'2000-12-31': 1, '2026-01-01': 23} |
| 27 | PASS | Row accounting: orphan + used + trailing = total rows | 1 + 219144 + 23 = 219168 |
| 28 | PASS | Calendar coverage: 365/366 days per year | all 25 years |
| 29 | PASS | Leap days present (2004, 2008, 2012, 2016, 2020, 2024) | leap years: [2004, 2008, 2012, 2016, 2020, 2024] |
| 30 | PASS | Positional alignment: block starts at 2001-01-01 01:00 and ends at 2026-01-01 00:00 | 2001-01-01 01:00:00+00:00 .. 2026-01-01 00:00:00+00:00 |
| 31 | PASS | Numerical aggregation cross-check (shared preprocessing) agrees with pandas within 1e-09 mm | max /difference/ = 1.421e-14 mm |
| 32 | PASS | Conservation: sum(hourly used) = sum(daily) within 1e-06 mm | hourly 72158.451469935 mm, daily 72158.451469935 mm, diff 0.000e+00 |
| 33 | PASS | Daily totals >= 0 | min = 0 mm |
| 34 | PASS | Daily totals < 1000 mm (screening bound, not physical validation) | max = 221.967459 mm |
| 35 | PASS | Plausibility screen: mean annual total in [500, 6000] mm (does not verify units) | 2886.3 mm/yr |
| 36 | PASS | Supporting pattern only: label 01:00 never negative (does not prove convention) | 9132 values at 01:00 UTC |
| 37 | PASS | Output round-trip: re-read file equals in-memory daily values exactly | 9131 rows |
| 38 | PASS | Input file unchanged after processing (hash re-check) |  |

## 7. Flags for review (nothing removed)

- Daily totals above 150 mm (a review threshold chosen for this report, not a statistical definition of 'extreme'): 5
  - 2016-05-15: 221.967 mm
  - 2002-11-02: 199.118 mm
  - 2006-10-23: 186.125 mm
  - 2006-11-06: 158.526 mm
  - 2004-09-26: 154.305 mm
- Hourly values above 50 mm: 0; maximum hourly value 28.0264 mm at label 2018-10-24 19:00:00+00:00 (UTC, hour-ending).
- These days should be compared with independent sources before being treated as genuine or erroneous.

## 8. Output files and checksums

- `colombo_daily_rainfall_2001_2025.csv`: columns `date, rainfall_mm`; 9,131 rows; SHA-256 `c01c4646a8e083dc36607db954f4d804bb26c878e7ac00f63af298014f78e53e`
- Input: `reanalysis-era5-land-timeseries-sfc-pressure-precipitationycy4vwq7.csv` SHA-256 `5cca3bfc3ab1dc36a86a9232752b5ac77afa45cd4838beb7d6d706d45fa14d48`

- Validation checks establish file integrity and numerical consistency under stated assumptions; they do not authenticate provenance or physically validate rainfall. See provenance_manifest.md for outstanding source fields.

- Output safety: CSV and report are staged in a temporary directory, validated before replacement, and removed on failure. Existing files survive pre-publication failures. Ordinary replacement failures trigger rollback; concurrent writers and process/power loss during the two-file publication are outside this guarantee.

## 9. Limitations and anomalies

- Reanalysis precipitation can differ from gauge observations, particularly in the tails; this step makes no comparison with gauges.
- The day is a UTC day, not a local day (local midnight falls mid-interval in the hourly data).
- The convention check on label 01:00 is indirect evidence; the primary basis is the ECMWF documentation.
- The grid cell's land/sea character and representativeness for Colombo were not checked here.
- Dataset citation, licence terms and download record are not in the file and must be added by the author.
- Anomalies in this run: orphan/trailing rows excluded as designed; 1,363 tiny negatives clipped; 5 days above the review threshold. No failed checks.

## 10. References (documentation consulted)

- https://confluence.ecmwf.int/spaces/CKB/pages/536218894/ERA5-Land+hourly+Analysis+Ready+Cloud+Optimised+ARCO+data+on+single+levels+from+1950+to+present+Product+User+Guide+PUG
- https://confluence.ecmwf.int/spaces/CKB/pages/197702790/Conversion+table+for+accumulated+variables+total+precipitation+fluxes
- https://confluence.ecmwf.int/spaces/UDOC/pages/208501579/Why+are+there+sometimes+small+negative+precipitation+accumulations+-+ecCodes+GRIB+FAQ
