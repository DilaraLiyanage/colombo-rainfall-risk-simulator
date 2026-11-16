# Independent verification of corrected preprocessing

Decision: **PASS for the requested software corrections and numerical verification; provenance completion remains outstanding.**

## Changes reviewed

- Mean annual rainfall is total divided by 25 complete calendar years, not duration divided by 365.2425.
- Descriptions distinguish file structure/fingerprint from provenance, UTC assignment from source timezone proof, numerical agreement from physical validation, and plausibility from units.
- Coordinate tolerance checks are retained; a separate exact-constancy check is added (38 checks total).
- CSV and report are written inside a temporary directory on the destination filesystem. The CSV is re-read and compared exactly, input hash rechecked, and report written/read back before replacing either prior output.
- Prior outputs are backed up before replacement. Ordinary second-replacement failure triggers rollback. TemporaryDirectory removes staged files after successful rollback or pre-publication failure.
- NEG_TOL_M=1e-7 m is explicitly project-selected, not prescribed by ECMWF.
- Source manifest has explicit acquisition TODOs; runtime versions and reproducible verifier/tests are included.

## Actual execution

Executed in Python 3.12.14, pandas 2.2.3, NumPy 2.3.5. Corrected script completed successfully with **38 PASS messages**. Execution logs were reviewed during verification and are intentionally excluded from the repository. This was execution in the available environment, not a fresh dependency installation.

An independent standard-library verifier used CSV reading, datetime interval membership and 50-digit Decimal arithmetic. It imports no preprocessing functions and uses neither pandas nor NumPy. It checks input hashes before/after execution, full date sequence, hourly continuity, exact coordinates, negative values, annual/daily sums and reference identity.

- Raw rows: 219168; missing cells, duplicates, non-hourly steps and off-hour labels: zero.
- Coordinates: one exact pair, 6.900000000000828 N / 79.89999999999975 E.
- Complete days: 9131, 2001-01-01 through 2025-12-31; 219144 hourly intervals.
- Exclusions: one interval belonging to 2000-12-31 and 23 belonging to 2026-01-01.
- All six leap days present; each year has its expected 365/366 dates.
- Negatives: 1363; minimum -3.9115548E-8 m; none below -1e-7 m.
- Clipping adds 0.01089528171377277000 mm across retained data.
- Total: 72158.45146993472159173000 mm.
- **Correct mean annual total: 2886.33805879738886366920 mm/year.**
- Maximum: 221.9674587000 mm on 2016-05-15.
- Maximum absolute independent-versus-output difference: **1E-14 mm**.
- Dates attaining that maximum: 2010-11-11, 2019-10-15, 2024-11-19.
- No dates differ by more than 1e-9 mm. Strict decimal differences on 1449 dates reflect binary arithmetic/CSV serialization at negligible scale.
- Corrected versus original cleaned CSV: **byte-for-byte identical**, maximum numerical difference zero.
- Annual sums and exact signed/clipped conservation are recalculated by tests/test_preprocessing.py; its CLI can write a results JSON outside the repository.

## Failure-injection evidence

All **7 tests passed** (tests/test_output_safety.py): early fingerprint failure, late input-hash validation failure, candidate CSV read failure, partial report-write failure, invalid report read-back, second replacement failure with prior outputs, and second replacement failure without prior outputs. Each asserts exact prior bytes/directory contents, including absence of staging debris.

Safety scope: one writer. Two individual atomic replacements do not form a crash-atomic pair. Process termination, power loss, concurrent writers, or persistent filesystem failure during rollback are not covered by the preservation guarantee. Validation and report-writing failures occur before publication, so prior outputs remain untouched. No tests simulate power loss.

## Checksums

| File | SHA-256 |
|---|---|
| Raw CSV (not bundled) | `5cca3bfc3ab1dc36a86a9232752b5ac77afa45cd4838beb7d6d706d45fa14d48` |
| Original reference CSV | `c01c4646a8e083dc36607db954f4d804bb26c878e7ac00f63af298014f78e53e` |
| Corrected daily CSV | `c01c4646a8e083dc36607db954f4d804bb26c878e7ac00f63af298014f78e53e` |
| Corrected script | `e569db7c6d4c47d3879639ac4dd69cdddeeaa160aed271b0ba16d149947e6d3a` |
| Corrected quality report | `61895f2ed36b1692b361c43b47079942b7db66ca12946550136f3300687c6d2d` |

The generated quality report is retained unchanged. This independent report is adapted for repository paths; execution transcripts and private workspace details are excluded.

## Remaining scientific/provenance limitations

Product-specific units and hour-ending UTC interpretation remain conditional on authenticating the original download. The documentation establishes how the named product works, not this file's acquisition history. Missing CDS request identifier/date/licence/DOI/version remain TODOs. Gauge accuracy, grid representativeness, and the physical reality of extremes were not validated. The study period and rainfall aggregation convention have not changed. No model fitting or simulations occurred. These files have been prepared in the working tree for review; no commit, push, PR or publication has occurred.

## Repository integration

The source script, quality report, provenance manifest and processed CSV were copied from the verified corrections. The source script and daily CSV retain their verified hashes above. Test paths now point to src/, data/processed/ and docs/; the independent algorithm remains separate from preprocessing code. tests/test_preprocessing.py supports both CLI verification and unittest discovery. Both test modules use RAINFALL_RAW_CSV to locate an external raw file. Test discovery without that variable explicitly skips the raw-dependent tests.

See data/README.md for commands. requirements.txt pins the installed runtime dependencies; .python-version records Python 3.12.14. Clean installation from a public package index has not been tested. Source provenance TODOs remain unresolved.

### Working-tree execution results

The repository test command `python -m unittest discover -s tests -v`, with RAINFALL_RAW_CSV pointing to the external verified raw input, executed all eight tests successfully (none skipped): seven safety tests plus one independent decimal verification test. The copied source script was also run successfully to an external scratch output directory, reporting 38 passed checks; its generated CSV matches the repository CSV byte for byte. The verifier CLI was additionally run against the original external reference CSV and confirmed byte identity, maximum independent difference 1e-14 mm, and mean annual rainfall 2886.33805879738886366920 mm/year. Original tracked files, including the MIT licence, remain unchanged. Raw data and execution logs were not added to the repository.
