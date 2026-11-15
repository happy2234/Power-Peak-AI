# Data quality report

Period **2023-04-01 .. 2026-11-08** — 1314/1318 daily reports present (99.7%).

## Missing daily reports
2024-08-05, 2024-12-30..2024-12-31 (2 d), 2025-11-10

### Retry (once)
- 2024-08-05: still not listed
- 2024-12-30: failed again: 404 Client Error: Not Found for url: https://webcdn.grid-india.in/files/grdw/uploads/daily-reports/psp-reports/2024-2025
- 2024-12-31: failed again: 404 Client Error: Not Found for url: https://webcdn.grid-india.in/files/grdw/uploads/daily-reports/psp-reports/2024-2025
- 2025-11-10: still not listed

## Failures recorded in backfill.log
Download problems: 4
- `2024-08-05` — not_listed: not listed on Grid-India
- `2024-12-30` — download_error: 404 Client Error: Not Found for url: https://webcdn.grid-india.in/files/grdw/uploads/daily-reports/psp-reports/2024-2025/30.12.24_NLDC_PSP.xls
- `2024-12-31` — download_error: 404 Client Error: Not Found for url: https://webcdn.grid-india.in/files/grdw/uploads/daily-reports/psp-reports/2024-2025/31.12.24_NLDC_PSP.xls
- `2025-11-10` — not_listed: not listed on Grid-India

Parse failures: 10
- `Week ahead all India demand forecast 16-Sep-2023 to 22-Sep-2023.pdf` — ForecastParseError: Week ahead all India demand forecast 16-Sep-2023 to 22-Sep-2023.pdf: fewer than 3 y-axis tick labels
- `Week ahead all India demand forecast 20-Jan-2024 to 26-Jan-2024.pdf` — ForecastParseError: Week ahead all India demand forecast 20-Jan-2024 to 26-Jan-2024.pdf: inconsistent forecast dates [(datetime.date(2024, 1, 20), datetime.date(2024, 1, 25)), (datetime.date(2024, 1, 20), datetime.date(2024, 1, 26))]; not guessing
- `Week ahead all India demand forecast 24-Aug-2024 to 30-Aug-2024.pdf` — ForecastParseError: Week ahead all India demand forecast 24-Aug-2024 to 30-Aug-2024.pdf: day gridlines not found (x=[])
- `Week ahead all India demand forecast 31-Aug-2024 to 06-Sep-2024.pdf` — ForecastParseError: Week ahead all India demand forecast 31-Aug-2024 to 06-Sep-2024.pdf: day gridlines not found (x=[])
- `Week ahead all India demand forecast 07-Sep-2024 to 13-Sep-2024.pdf` — ForecastParseError: Week ahead all India demand forecast 07-Sep-2024 to 13-Sep-2024.pdf: day gridlines not found (x=[])
- `Week ahead all India demand forecast 03-May-2025 to 09-May-2025_815.pdf` — ForecastParseError: Week ahead all India demand forecast 03-May-2025 to 09-May-2025_815.pdf: no vector chart found (scanned/image PDF?)
- `Week ahead all India demand forecast 10-May-2025 to 16-May-2025_443.pdf` — ForecastParseError: Week ahead all India demand forecast 10-May-2025 to 16-May-2025_443.pdf: no vector chart found (scanned/image PDF?)
- `Week ahead all India demand forecast 17-May-2025 to 23-May-2025_748.pdf` — ForecastParseError: Week ahead all India demand forecast 17-May-2025 to 23-May-2025_748.pdf: no vector chart found (scanned/image PDF?)
- `Week ahead all India demand forecast 26-Jul-2025 to 01-Aug-2025_316.pdf` — ForecastParseError: Week ahead all India demand forecast 26-Jul-2025 to 01-Aug-2025_316.pdf: no vector chart found (scanned/image PDF?)
- `Week ahead all India demand forecast 13-Jul-2026 to 19-Jul-2026_731.pdf` — ForecastParseError: Week ahead all India demand forecast 13-Jul-2026 to 19-Jul-2026_731.pdf: bad range label 238-24

## Targets
| target | available | still missing (not imputed) | imputed (gap ≤ 2 d) | min | median | max |
|---|---|---|---|---|---|---|
| solar_peak | 1289 | 2023-04-01..2023-04-29 (29 d) | 2024-08-05, 2024-12-30..2024-12-31 (2 d), 2025-11-10 | 158,988 | 215,800 | 270,820 |
| evening_peak | 1289 | 2023-04-01..2023-04-29 (29 d) | 2024-08-05, 2024-12-30..2024-12-31 (2 d), 2025-11-10 | 155,179 | 208,845 | 253,294 |
| energy | 1318 | none | 2024-08-05, 2024-12-30..2024-12-31 (2 d), 2025-11-10 | 3,467 | 4,716 | 6,002 |

Missing solar/evening peaks in April 2023 are a source limitation (those reports have no section I). They are left as NaN; models mask them.

## Sanity ranges
- solar_peak: 150,000–300,000
- evening_peak: 130,000–300,000
- energy: 3,000–6,000

Out-of-range values: 2
- 2026-07-16 energy = 6,002
- 2026-07-17 energy = 6,002

## Outliers (|robust z| > 6 vs 15-day rolling median)
21 flagged (kept in the data, flagged `is_outlier`; most are genuine holidays/storms):
- 2023-11-13 solar_peak = 164,063 (z=-7.0)
- 2024-03-25 solar_peak = 182,600 (z=-8.1)
- 2026-11-08 solar_peak = 195,673 (z=-6.7)
- 2023-11-12 evening_peak = 155,179 (z=-7.7)
- 2023-11-13 evening_peak = 163,299 (z=-6.0)
- 2024-01-14 evening_peak = 181,034 (z=-6.6)
- 2024-02-04 evening_peak = 181,371 (z=-6.7)
- 2024-03-03 evening_peak = 176,410 (z=-8.5)
- 2024-11-01 evening_peak = 170,301 (z=-6.5)
- 2024-11-03 evening_peak = 171,952 (z=-6.5)
- 2025-02-02 evening_peak = 193,168 (z=-7.6)
- 2025-02-09 evening_peak = 195,386 (z=-6.7)
- 2025-02-16 evening_peak = 196,286 (z=-6.2)
- 2025-11-16 evening_peak = 180,291 (z=-6.3)
- 2025-11-23 evening_peak = 185,544 (z=-6.5)
- 2025-11-30 evening_peak = 186,457 (z=-7.3)
- 2025-12-14 evening_peak = 194,641 (z=-8.3)
- 2026-02-01 evening_peak = 196,560 (z=-6.2)
- 2026-03-04 evening_peak = 183,675 (z=-12.5)
- 2026-11-08 evening_peak = 176,362 (z=-7.6)
- 2026-11-08 energy = 4,067 (z=-7.9)
