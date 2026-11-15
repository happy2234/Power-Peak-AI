# Baseline before the accuracy round

Snapshot taken 2026-11-11 22:37:48 (models/baseline_snapshot/, not in git). Production models: lgbm 20261109-204711, lstm 20261109-204722, tft 20261109-204738.

Fixed-split test period: every origin from 2025-03-31, labels 2025-04-01 .. 2026-11-08, models trained once on FY 2023-24 + FY 2024-25 (validation Jan–Mar 2025). These numbers are the first of the two allowed looks at the test period.

| model | target | days | MAPE % | MAE | bias | P10–P90 coverage % |
|---|---|---|---|---|---|---|
| ensemble | energy | 587 | 2.66 | 133 | -66 | 71.4 |
| ensemble | evening_peak | 587 | 2.23 | 4,898 | -1,983 | 68.5 |
| ensemble | solar_peak | 587 | 3.12 | 7,077 | -3,020 | 67.3 |
| lgbm@compact | energy | 587 | 2.77 | 138 | -41 | 71.6 |
| lgbm@compact | evening_peak | 587 | 2.27 | 4,978 | -1,545 | 67.6 |
| lgbm@compact | solar_peak | 587 | 3.31 | 7,500 | -3,335 | 66.9 |
| lstm_w@compact | energy | 587 | 2.75 | 138 | -77 | 70.3 |
| lstm_w@compact | evening_peak | 587 | 2.37 | 5,242 | -2,522 | 69.1 |
| lstm_w@compact | solar_peak | 587 | 3.07 | 6,967 | -2,299 | 68.9 |
| seasonal_naive | energy | 587 | 4.11 | 204 | +2 | 70.2 |
| seasonal_naive | evening_peak | 587 | 3.41 | 7,423 | -41 | 60.5 |
| seasonal_naive | solar_peak | 587 | 4.80 | 10,754 | -71 | 60.5 |
| tft@v1 | energy | 587 | 2.61 | 128 | -17 | 66.0 |
| tft@v1 | evening_peak | 587 | 2.15 | 4,677 | +205 | 69.3 |
| tft@v1 | solar_peak | 587 | 3.18 | 7,114 | -1,549 | 61.6 |

Ensemble, mean of the three targets: MAPE 2.667 %, coverage 69.1 %.

Roll back: `bash models/baseline_snapshot/restore.sh`.
