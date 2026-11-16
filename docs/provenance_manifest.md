# Dataset provenance manifest

## Evidence scope

This manifest separates (1) official CDS request evidence, (2) externally checked current catalogue information, and (3) results from the prior numerical preprocessing review. The request receipt and request-history screenshot were supplied for this documentation update; they are not stored in the repository. No preprocessing or scientific tests were rerun for this update.

## Official CDS request evidence

Evidence: supplied official receipt `receipt-134fc596-ac39-44fd-9a2d-156575b02d22.txt`, together with the supplied CDS request-history screenshot. Receipt fields are evidence about the request and packaged result, not hashes of the extracted CSV.

- Dataset/product: ERA5-Land hourly time-series data from 1950 to present.
- CDS collection ID: `reanalysis-era5-land-timeseries`.
- Official dataset page: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries
- CDS request ID: `134fc596-ac39-44fd-9a2d-156575b02d22`.
- Origin: CDS web user interface (`origin: ui`).
- Status: successful, consistent with the screenshot.

The receipt records these request parameters:

```json
{
  "date": ["2001-01-01/2026-01-01"],
  "location": {"latitude": 6.9, "longitude": 79.9},
  "variable": ["total_precipitation"],
  "data_format": "csv"
}
```

### Request timestamps

The receipt's timestamp strings contain no explicit timezone suffix. They are recorded below as UTC, corroborated by the +05:30 correspondence with the screenshot's local submission/end times and the supplied identification of that display timezone as IST. The screenshot itself does not visibly label a timezone. IST here means India Standard Time; Sri Lanka Standard Time has the same UTC+05:30 offset.

| Event | Receipt time, interpreted as UTC | Corresponding local time, UTC+05:30 |
|---|---|---|
| Created | 2026-11-16T05:35:55.139121Z | 2026-11-16 11:05:55.139121 |
| Started | 2026-11-16T05:36:12.936089Z | 2026-11-16 11:06:12.936089 |
| Finished | 2026-11-16T05:36:25.981165Z | 2026-11-16 11:06:25.981165 |

The screenshot shows submission at 11:05:55 AM and end at 11:06:25 AM on 16 November 2026, with a displayed duration of 00:00:30. From the receipt, created-to-finished elapsed time is 30.842044 seconds; started-to-finished time is 13.045076 seconds. Thus the displayed approximately 30 seconds should not be described as the started-to-finished execution interval. The receipt's updated-at time equals its finished-at time.

**Request completion is not proof of local file download. The exact download timestamp remains unverified.**

### Receipt licence/terms labels and package size

The receipt lists the following labels verbatim:

- CC-BY licence (version 1).
- Terms of use of the Copernicus Climate Data Store (version 13).
- Data protection and privacy statement (version 24).

These are receipt labels and document-version identifiers. In particular, “version 1” is not evidence that the Creative Commons legal-code version is 1.0. No licence URL or legal-code version is inferred.

The receipt reports `download-size: 1785425` bytes and a packaged filename ending in `.zip`. This is the **reported packaged download size (1,785,425 bytes)**, not the size of the extracted rainfall CSV. The package itself has not been cryptographically compared with the reviewed CSV; the receipt supplies no archive or extracted-member hash. Private storage paths are omitted.

## Current official catalogue information checked separately

On 16 November 2026, the accessible [official CDS dataset page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries) displayed:

- The matching time-series product, hourly sampling, CSV availability and a 0.1° regular latitude-longitude grid.
- Total precipitation units: metres (`m`).
- DOI: [10.24381/ee82e357](https://doi.org/10.24381/ee82e357).
- A licence label of “CC-BY licence”.

The DOI and current licence label are independently checked catalogue information, not fields contained in the receipt. The accessible page exposed citation/licence buttons but not their complete text or a licence URL, so a full recommended citation and licence legal text/version remain unresolved. Current catalogue metadata does not identify the exact release or generation time of this request's package.

## Previously verified raw-file information

The following facts come from the earlier numerical preprocessing review, not from the request receipt. Numerical preprocessing checks were not rerun during this documentation-only update; repository-file hashes were compared solely to verify the scope of edits.

- Raw filename: `reanalysis-era5-land-timeseries-sfc-pressure-precipitationycy4vwq7.csv`.
- Raw SHA-256: `5cca3bfc3ab1dc36a86a9232752b5ac77afa45cd4838beb7d6d706d45fa14d48`.
- Extracted CSV size: **14,970,156 bytes**; 219,168 hourly records.
- CSV schema: `valid_time, tp, latitude, longitude`.
- Timestamp labels: 2001-01-01 00:00:00 through 2026-01-01 23:00:00; hourly, unique and continuous.
- Coordinates: constant latitude 6.900000000000828, longitude 79.89999999999975, approximately 6.9° N / 79.9° E.
- The CSV itself has no timezone, units, product-version or acquisition metadata.

The requested date range includes 2026-01-01; the retained analysis ends on 2025-12-31. These are different, intentional boundaries, not conflicting study periods. The receipt parameters and reviewed file characteristics are consistent, but consistency does not establish source-package-to-extracted-CSV cryptographic linkage.

## Scientific interpretation and preprocessing convention

The receipt now establishes the requested product identity rather than leaving it inferred solely from a filename. Product documentation establishes that its precipitation is de-accumulated, uses hour-ending validity times and represents depth in metres of water equivalent. Applying that interpretation to the reviewed CSV remains subject to the unresolved package linkage. No additional differencing is applied. Conversion: 1 m = 1000 mm.

The ECMWF time-series guide's de-accumulation example uses previous-day step24 minus step23 at 00:00 and current-day step1 at 01:00. Thus UTC day D uses D 01:00 through D+1 00:00. UTC days span 05:30–05:30 Sri Lanka time.

- Product guide: https://confluence.ecmwf.int/pages/viewpage.action?pageId=699689224
- ERA5-Land conventions and tp units: https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation
- Packing artefacts: https://confluence.ecmwf.int/spaces/UDOC/pages/208501579/Why+are+there+sometimes+small+negative+precipitation+accumulations+-+ecCodes+GRIB+FAQ

These references support product interpretation. The receipt documents the request, but neither the references nor the receipt establish a cryptographic link from its source package to the reviewed CSV. The 1e-7 m negative tolerance is a project-selected policy informed by observed tiny negatives, not an ECMWF-prescribed threshold. Original packing-error metadata is unavailable.

## Previously verified derived data

These results are from the numerical preprocessing review and are not established by the CDS receipt:

- Cleaned filename: `colombo_daily_rainfall_2001_2025.csv`.
- Cleaned CSV SHA-256: `c01c4646a8e083dc36607db954f4d804bb26c878e7ac00f63af298014f78e53e`.
- Retained study period: 2001-01-01 through 2025-12-31; 25 complete UTC calendar years, 9,131 daily observations.
- 1,363 negative hourly values clipped within the project-selected `NEG_TOL_M = 1e-7` metres; total adjustment +0.01089528171377277 mm.
- Retained rainfall total after clipping: approximately 72,158.45146993472 mm.
- Mean annual rainfall across the 25 calendar years: approximately 2,886.3380587974 mm/year.
- Maximum daily rainfall: approximately 221.9674587 mm on 2016-05-15.

## Resolved provenance fields

The receipt resolves the request ID, exact collection ID and dataset URL, request origin, requested variable/date range/location/format, successful status, creation/start/completion timestamps under the corroborated UTC interpretation, licence/terms document labels and versions, and reported packaged download size. The current official catalogue separately resolves the dataset DOI. These do not resolve download time, source-package hashes or exact data release.

## Outstanding evidence — explicit TODOs

- Actual local file-download timestamp, including timezone: **TODO — unverified; request completion is not a substitute**.
- Complete licence legal text, legal-code version and verified licence URL: **TODO — receipt labels/versions are recorded above; do not infer additional terms**.
- Full official recommended citation/attribution: **TODO — DOI verified above, but complete citation text not retrieved**.
- Exact dataset version/release, including consolidated versus preliminary status: **TODO — not established by the receipt or current catalogue**.
- Source-package generation timestamp: **TODO — not established; do not substitute request completion or catalogue update time**.
- Cryptographic linkage from the received ZIP to the reviewed extracted CSV: **TODO — original archive, member listing and archive/member hashes not verified**.
- Original file-level metadata confirming units/timezone/de-accumulation: **TODO — attach or verify if available; product-level documentation and receipt are not file-level metadata**.

The request parameters are transcribed above; the private receipt remains outside the repository. Do not infer unresolved acquisition details from the raw-file hash.

## Scientific limitations retained

- ERA5-Land is gridded reanalysis, not direct rain-gauge observations.
- UTC daily windows correspond to approximately 05:30–05:30 Sri Lanka local time, rather than local midnight-to-midnight days.
- The coastal Colombo grid point may not perfectly represent local station rainfall; grid representativeness and land/sea character have not been independently established here.
- The clipping tolerance is a documented project decision, not an ECMWF-prescribed threshold.
- The request receipt alone does not establish exact source-package-to-extracted-CSV cryptographic verification.
- Dataset provenance does not establish the accuracy of extreme precipitation estimates. No gauge validation or new scientific analysis was performed for this update.
