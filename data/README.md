# Rainfall data and preprocessing

`processed/colombo_daily_rainfall_2001_2025.csv` contains the verified daily rainfall series: 9,131 complete UTC days, 2001-01-01 through 2025-12-31. Schema: `date,rainfall_mm`. Its SHA-256 is:

```text
c01c4646a8e083dc36607db954f4d804bb26c878e7ac00f63af298014f78e53e
```

The raw hourly CSV is intentionally excluded. Keep it outside the repository. The required raw SHA-256 is:

```text
5cca3bfc3ab1dc36a86a9232752b5ac77afa45cd4838beb7d6d706d45fa14d48
```

## Reproduce verification

Run from the repository root with Python 3.12.14 and the versions in requirements.txt. A clean dependency installation has not been tested.

```bash
export RAINFALL_RAW_CSV=/absolute/path/outside/repository/raw.csv
python -m unittest discover -s tests -v
```

This runs the independent decimal verifier and seven output-safety tests. Without RAINFALL_RAW_CSV, the raw-dependent tests explicitly skip. The raw file remains read-only. No fitting or simulation is performed.

To reproduce preprocessing without overwriting the reviewed files, select a new output directory outside the repository:

```bash
python src/preprocess_rainfall.py --input "$RAINFALL_RAW_CSV" --output-dir /absolute/path/outside/repository/new-preprocessing-run
python tests/test_preprocessing.py --raw "$RAINFALL_RAW_CSV" --cleaned /absolute/path/outside/repository/new-preprocessing-run/colombo_daily_rainfall_2001_2025.csv --reference data/processed/colombo_daily_rainfall_2001_2025.csv --output /absolute/path/outside/repository/verification-results.json
```

The script emits both a daily CSV and quality report into the chosen directory. The repository's reviewed report is at docs/preprocessing_quality_report.md; the independent report and provenance manifest are beside it.

## Interpretation and limitations

Under the documented ERA5-Land time-series convention, precipitation is already de-accumulated and labels end each hourly interval. UTC day D uses D 01:00 through D+1 00:00. Values in metres are multiplied once by 1000. UTC daily windows span 05:30–05:30 Sri Lanka time. NEG_TOL_M=1e-7 m is a project-selected tolerance, not an ECMWF-prescribed threshold.

Product identity, source units/timezone and acquisition details are conditional on recovering the original download metadata. Outstanding DOI/version/licence/request/date fields are explicit TODOs in docs/provenance_manifest.md. The existing MIT licence is preserved for project code; it does not establish source-data licensing terms. Reanalysis accuracy and grid representativeness have not been validated against gauges.
