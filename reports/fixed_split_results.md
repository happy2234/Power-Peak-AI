# Fixed train/test evaluation

Models trained **once** on 2023-05-03 .. 2024-12-24 (forecast origins; labels end before 2025-01-01). 2025-01-01 .. 2025-03-31 is the validation slice, used only for early stopping, Optuna tuning, the ensemble weights and the P10–P90 calibration. Test: every daily forecast origin from 2025-03-31 to 2026-11-07, 1–7 days ahead, with the models frozen (no refit, no online recalibration). Weather inputs are the archived forecasts issued at each origin; the anomaly normals, scalers and rare-weather thresholds use the training period only. Peaks in MW (errors in MW), energy in MU.

The feature set of each model (LightGBM and LSTM: compact weather; TFT: previous inputs) was chosen in the rolling backtest, whose months overlap this test period; that choice is the one piece of design informed by test-period data.

## Overall (test period)

| model | target | days | MAPE % | MAE | RMSE | bias | P10–P90 coverage % |
|---|---|---|---|---|---|---|---|
| Ensemble | solar_peak | 587 | 3.12 | 7,077 | 9,264 | -3,020 | 67.3 |
| Ensemble | evening_peak | 587 | 2.23 | 4,898 | 6,278 | -1,983 | 68.5 |
| Ensemble | energy | 587 | 2.66 | 133 | 170 | -66 | 71.4 |
| LSTM (weighted, compact weather) | solar_peak | 587 | 3.07 | 6,967 | 9,084 | -2,299 | 68.9 |
| LSTM (weighted, compact weather) | evening_peak | 587 | 2.37 | 5,242 | 6,689 | -2,522 | 69.1 |
| LSTM (weighted, compact weather) | energy | 587 | 2.75 | 138 | 176 | -77 | 70.3 |
| LightGBM (compact weather) | solar_peak | 587 | 3.31 | 7,500 | 9,714 | -3,335 | 66.9 |
| LightGBM (compact weather) | evening_peak | 587 | 2.27 | 4,978 | 6,466 | -1,545 | 67.6 |
| LightGBM (compact weather) | energy | 587 | 2.77 | 138 | 179 | -41 | 71.6 |
| TFT (v1 weather) | solar_peak | 587 | 3.18 | 7,114 | 9,117 | -1,549 | 61.6 |
| TFT (v1 weather) | evening_peak | 587 | 2.15 | 4,677 | 6,258 | +205 | 69.3 |
| TFT (v1 weather) | energy | 587 | 2.61 | 128 | 168 | -17 | 66.0 |
| Seasonal naive | solar_peak | 587 | 4.80 | 10,754 | 14,099 | -71 | 60.5 |
| Seasonal naive | evening_peak | 587 | 3.41 | 7,423 | 10,138 | -41 | 60.5 |
| Seasonal naive | energy | 587 | 4.11 | 204 | 273 | +2 | 70.2 |

## By financial year — MAPE %

|                                |   days |   Ensemble |   LSTM (weighted, compact weather) |   LightGBM (compact weather) |   TFT (v1 weather) |   Seasonal naive |
|:-------------------------------|-------:|-----------:|-----------------------------------:|-----------------------------:|-------------------:|-----------------:|
| ('energy', 'FY 2025-26')       |    365 |       2.49 |                               2.6  |                         2.64 |               2.61 |             3.76 |
| ('energy', 'FY 2026-27')       |    222 |       2.93 |                               3.01 |                         2.96 |               2.61 |             4.68 |
| ('evening_peak', 'FY 2025-26') |    365 |       2.12 |                               2.25 |                         2.21 |               2.16 |             3.26 |
| ('evening_peak', 'FY 2026-27') |    222 |       2.4  |                               2.58 |                         2.39 |               2.14 |             3.65 |
| ('solar_peak', 'FY 2025-26')   |    365 |       2.87 |                               2.82 |                         3.06 |               3.02 |             4.27 |
| ('solar_peak', 'FY 2026-27')   |    222 |       3.52 |                               3.48 |                         3.72 |               3.43 |             5.68 |

## By season — MAPE %

|                                                  |   days |   Ensemble |   LSTM (weighted, compact weather) |   LightGBM (compact weather) |   TFT (v1 weather) |   Seasonal naive |
|:-------------------------------------------------|-------:|-----------:|-----------------------------------:|-----------------------------:|-------------------:|-----------------:|
| ('solar_peak', 'winter (Dec–Feb)')               |     90 |       1.48 |                               1.77 |                         1.54 |               1.81 |             2.15 |
| ('solar_peak', 'pre-monsoon summer (Mar–May)')   |    153 |       3.57 |                               3.88 |                         3.57 |               3.47 |             6.17 |
| ('solar_peak', 'monsoon (Jun–Sep)')              |    244 |       3.52 |                               3.13 |                         3.81 |               3.42 |             4.72 |
| ('solar_peak', 'post-monsoon (Oct–Nov)')         |    100 |       2.94 |                               2.88 |                         3.3  |               3.38 |             5.34 |
| ('evening_peak', 'winter (Dec–Feb)')             |     90 |       1.22 |                               1.31 |                         1.26 |               1.43 |             1.71 |
| ('evening_peak', 'pre-monsoon summer (Mar–May)') |    153 |       2.76 |                               3.01 |                         2.68 |               2.48 |             4.76 |
| ('evening_peak', 'monsoon (Jun–Sep)')            |    244 |       2.3  |                               2.42 |                         2.33 |               2.17 |             3    |
| ('evening_peak', 'post-monsoon (Oct–Nov)')       |    100 |       2.16 |                               2.26 |                         2.45 |               2.26 |             3.9  |
| ('energy', 'winter (Dec–Feb)')                   |     90 |       1.2  |                               1.29 |                         1.26 |               1.57 |             1.75 |
| ('energy', 'pre-monsoon summer (Mar–May)')       |    153 |       3.33 |                               3.44 |                         3.19 |               2.9  |             5.67 |
| ('energy', 'monsoon (Jun–Sep)')                  |    244 |       2.81 |                               2.9  |                         2.94 |               2.68 |             3.86 |
| ('energy', 'post-monsoon (Oct–Nov)')             |    100 |       2.58 |                               2.69 |                         3.06 |               2.94 |             4.5  |

## By weather regime (observed weather, evaluation only) — MAPE %

|                                      |   days |   Ensemble |   LSTM (weighted, compact weather) |   LightGBM (compact weather) |   TFT (v1 weather) |   Seasonal naive |
|:-------------------------------------|-------:|-----------:|-----------------------------------:|-----------------------------:|-------------------:|-----------------:|
| ('solar_peak', 'heavy rain/storm')   |     52 |       3    |                               2.99 |                         3.18 |               3.29 |             6.33 |
| ('solar_peak', 'cold/fog')           |     40 |       1.57 |                               1.75 |                         1.57 |               2.13 |             2.35 |
| ('solar_peak', 'hot')                |     74 |       4.71 |                               5.1  |                         4.67 |               3.57 |             6.25 |
| ('solar_peak', 'rainy')              |    218 |       3.35 |                               2.88 |                         3.72 |               3.43 |             4.73 |
| ('solar_peak', 'cloudy')             |     11 |       3.6  |                               3.4  |                         3.81 |               3.3  |             5.16 |
| ('solar_peak', 'humid')              |      3 |       3.91 |                               2.87 |                         4.23 |               2.8  |             5.31 |
| ('solar_peak', 'normal')             |    189 |       2.55 |                               2.78 |                         2.66 |               2.92 |             4.39 |
| ('evening_peak', 'heavy rain/storm') |     52 |       2.25 |                               2.39 |                         2.31 |               2.69 |             4.21 |
| ('evening_peak', 'cold/fog')         |     40 |       1.4  |                               1.54 |                         1.33 |               1.73 |             2    |
| ('evening_peak', 'hot')              |     74 |       3.18 |                               3.67 |                         2.85 |               1.85 |             4.68 |
| ('evening_peak', 'rainy')            |    218 |       2.28 |                               2.29 |                         2.48 |               2.33 |             3.31 |
| ('evening_peak', 'cloudy')           |     11 |       2.26 |                               2.67 |                         2.2  |               1.26 |             3.18 |
| ('evening_peak', 'humid')            |      3 |       1.82 |                               2.23 |                         2.05 |               0.98 |             2.57 |
| ('evening_peak', 'normal')           |    189 |       1.96 |                               2.11 |                         2    |               2.07 |             3.12 |
| ('energy', 'heavy rain/storm')       |     52 |       2.43 |                               2.47 |                         2.7  |               2.61 |             5.18 |
| ('energy', 'cold/fog')               |     40 |       1.4  |                               1.48 |                         1.35 |               1.96 |             2.03 |
| ('energy', 'hot')                    |     74 |       4.16 |                               4.45 |                         3.47 |               2.45 |             5.52 |
| ('energy', 'rainy')                  |    218 |       2.65 |                               2.72 |                         3.05 |               2.83 |             3.98 |
| ('energy', 'cloudy')                 |     11 |       3.21 |                               3.36 |                         3.07 |               2.43 |             4.47 |
| ('energy', 'humid')                  |      3 |       1.97 |                               2.22 |                         2.44 |               1.62 |             3.26 |
| ('energy', 'normal')                 |    189 |       2.37 |                               2.45 |                         2.46 |               2.59 |             3.84 |

## By horizon (days ahead) — MAPE %

|                     |   days |   Ensemble |   LSTM (weighted, compact weather) |   LightGBM (compact weather) |   TFT (v1 weather) |   Seasonal naive |
|:--------------------|-------:|-----------:|-----------------------------------:|-----------------------------:|-------------------:|-----------------:|
| ('energy', 1)       |    587 |       1.99 |                               2    |                         2.04 |               2.09 |             4.11 |
| ('energy', 2)       |    586 |       2.3  |                               2.36 |                         2.39 |               2.33 |             4.11 |
| ('energy', 3)       |    585 |       2.59 |                               2.61 |                         2.61 |               2.52 |             4.11 |
| ('energy', 4)       |    584 |       2.74 |                               2.84 |                         2.82 |               2.68 |             4.11 |
| ('energy', 5)       |    583 |       2.88 |                               3.04 |                         2.96 |               2.79 |             4.11 |
| ('energy', 6)       |    582 |       3.03 |                               3.21 |                         3.22 |               2.94 |             4.11 |
| ('energy', 7)       |    581 |       3.08 |                               3.22 |                         3.33 |               2.93 |             4.11 |
| ('evening_peak', 1) |    587 |       1.75 |                               1.82 |                         1.86 |               1.76 |             3.4  |
| ('evening_peak', 2) |    586 |       1.99 |                               2.1  |                         2.05 |               1.93 |             3.41 |
| ('evening_peak', 3) |    585 |       2.16 |                               2.29 |                         2.18 |               2.11 |             3.41 |
| ('evening_peak', 4) |    584 |       2.29 |                               2.44 |                         2.3  |               2.22 |             3.4  |
| ('evening_peak', 5) |    583 |       2.38 |                               2.57 |                         2.39 |               2.28 |             3.41 |
| ('evening_peak', 6) |    582 |       2.47 |                               2.71 |                         2.54 |               2.36 |             3.41 |
| ('evening_peak', 7) |    581 |       2.54 |                               2.7  |                         2.6  |               2.42 |             3.42 |
| ('solar_peak', 1)   |    587 |       2.35 |                               2.31 |                         2.62 |               2.58 |             4.8  |
| ('solar_peak', 2)   |    586 |       2.71 |                               2.6  |                         2.94 |               2.81 |             4.81 |
| ('solar_peak', 3)   |    585 |       2.92 |                               2.87 |                         3.17 |               3.04 |             4.81 |
| ('solar_peak', 4)   |    584 |       3.1  |                               3.11 |                         3.34 |               3.23 |             4.81 |
| ('solar_peak', 5)   |    583 |       3.41 |                               3.35 |                         3.52 |               3.38 |             4.8  |
| ('solar_peak', 6)   |    582 |       3.7  |                               3.58 |                         3.79 |               3.59 |             4.8  |
| ('solar_peak', 7)   |    581 |       3.66 |                               3.69 |                         3.82 |               3.62 |             4.8  |

## Drift: monthly MAPE over the test period (mean of the three targets)

| month    |   Ensemble |   LSTM (weighted, compact weather) |   LightGBM (compact weather) |   TFT (v1 weather) |   Seasonal naive |
|:---------|-----------:|-----------------------------------:|-----------------------------:|-------------------:|-----------------:|
| Apr 2025 |       2.43 |                               2.81 |                         2.21 |               2.28 |             3.7  |
| May 2025 |       2.99 |                               3.23 |                         3.1  |               2.88 |             5.86 |
| Jun 2025 |       3.74 |                               3.78 |                         3.91 |               3.36 |             5.92 |
| Jul 2025 |       2.23 |                               1.96 |                         2.5  |               2.1  |             1.61 |
| Aug 2025 |       2.07 |                               1.8  |                         2.53 |               2.89 |             3.25 |
| Sep 2025 |       4.05 |                               4.26 |                         3.8  |               4.36 |             3.49 |
| Oct 2025 |       2.72 |                               2.41 |                         3.85 |               3.32 |             6.97 |
| Nov 2025 |       2.85 |                               3.07 |                         2.59 |               2.57 |             2.72 |
| Dec 2025 |       1.26 |                               1.3  |                         1.38 |               1.44 |             1.98 |
| Jan 2026 |       1.68 |                               1.82 |                         1.68 |               2.3  |             2.11 |
| Feb 2026 |       0.94 |                               1.22 |                         0.97 |               1.02 |             1.49 |
| Mar 2026 |       2.91 |                               3.03 |                         3    |               2.54 |             5.84 |
| Apr 2026 |       3.08 |                               3.22 |                         2.91 |               2.67 |             4.52 |
| May 2026 |       4.56 |                               4.84 |                         4.38 |               4.28 |             7.48 |
| Jun 2026 |       3.24 |                               3.47 |                         3.07 |               2.2  |             3.85 |
| Jul 2026 |       2.9  |                               2.91 |                         3.14 |               3.24 |             6.88 |
| Aug 2026 |       2.54 |                               2.29 |                         2.87 |               1.86 |             2.44 |
| Sep 2026 |       2.31 |                               2.16 |                         2.44 |               2.08 |             3.48 |
| Oct 2026 |       2.18 |                               2.44 |                         2.41 |               2.46 |             4.38 |
| Nov 2026 |       2.3  |                               2.32 |                         2.71 |               3.76 |             3.09 |

Seasons dominate the monthly pattern (pre-monsoon summer and festival months are hard, winter is easy), so drift is measured on the same calendar months one year apart. Ensemble, Apr–Nov 2025 vs Apr–Nov 2026: 2.89 % vs 2.89 % — the error barely changes after a year without retraining.

The frozen models under-forecast on average (ensemble bias -3.0 GW solar-hour, -2.0 GW evening, -66 MU energy). FY 2025-26 had unusually weak demand growth (about 1 %); because the models predict demand relative to the last 7 days, a change in the growth rate shows up as a small level error rather than as an error that accumulates.

P10–P90 coverage falls to 67 %, 68 %, 71 % (solar-hour, evening, energy) instead of 80 %. The intervals were calibrated on Jan–Mar 2025, the season with the smallest errors, and never updated; the rolling backtest recalibrates every month and holds about 80 %. Interval calibration drifts faster than the point forecast.

## Grid-India on the same test days (ensemble)

| target | our basis | Grid-India | days | our MAPE % | Grid-India MAPE % | our error lower by | win rate |
|---|---|---|---|---|---|---|---|
| energy | mean of 1–7-day forecasts | as published | 552 | 2.51 | 4.15 | +37.4% | 64% |
| evening_peak | mean of 1–7-day forecasts | as published | 552 | 2.12 | 3.29 | +34.7% | 63% |
| solar_peak | mean of 1–7-day forecasts | as published | 552 | 2.93 | 4.81 | +37.2% | 66% |
| energy | mean of 1–7-day forecasts | bias-corrected | 552 | 2.51 | 3.95 | +34.4% | 64% |
| evening_peak | mean of 1–7-day forecasts | bias-corrected | 552 | 2.12 | 3.13 | +31.2% | 64% |
| solar_peak | mean of 1–7-day forecasts | bias-corrected | 552 | 2.93 | 4.62 | +35.0% | 66% |
| energy | 7-day-ahead forecast only | as published | 546 | 3.03 | 4.16 | +24.5% | 61% |
| evening_peak | 7-day-ahead forecast only | as published | 546 | 2.49 | 3.30 | +23.1% | 60% |
| solar_peak | 7-day-ahead forecast only | as published | 546 | 3.56 | 4.79 | +23.3% | 59% |
| energy | 7-day-ahead forecast only | bias-corrected | 546 | 3.03 | 3.96 | +20.9% | 59% |
| evening_peak | 7-day-ahead forecast only | bias-corrected | 546 | 2.49 | 3.13 | +18.8% | 58% |
| solar_peak | 7-day-ahead forecast only | bias-corrected | 546 | 3.56 | 4.61 | +20.8% | 60% |

Grid-India's forecast is prepared about 5 days before its week starts (6–12 days ahead), so even the 7-day-ahead basis favours us slightly; the matched comparison is in fairness_audit.md.

## Three stages compared — MAPE %

| model | target | fixed split, whole test (Apr 2025 →) | fixed split, same 12 months as rolling | rolling backtest (monthly refit) | live (days) |
|---|---|---|---|---|---|
| Ensemble | solar_peak | 3.12 | 2.97 | 2.40 | – |
| Ensemble | evening_peak | 2.23 | 2.10 | 1.79 | – |
| Ensemble | energy | 2.66 | 2.55 | 2.00 | – |
| LSTM (weighted, compact weather) | solar_peak | 3.07 | 3.03 | 2.39 | 1.43 (7) |
| LSTM (weighted, compact weather) | evening_peak | 2.37 | 2.25 | 1.91 | 3.39 (7) |
| LSTM (weighted, compact weather) | energy | 2.75 | 2.63 | 2.06 | 2.56 (7) |
| LightGBM (compact weather) | solar_peak | 3.31 | 3.11 | 2.78 | 1.39 (7) |
| LightGBM (compact weather) | evening_peak | 2.27 | 2.10 | 1.81 | 2.67 (7) |
| LightGBM (compact weather) | energy | 2.77 | 2.57 | 2.26 | 3.62 (7) |
| TFT (v1 weather) | solar_peak | 3.18 | 2.94 | 2.80 | 1.55 (7) |
| TFT (v1 weather) | evening_peak | 2.15 | 1.93 | 1.96 | 3.39 (7) |
| TFT (v1 weather) | energy | 2.61 | 2.38 | 2.32 | 2.86 (7) |
| Seasonal naive | solar_peak | 4.80 | 4.74 | 4.74 | – |
| Seasonal naive | evening_peak | 3.41 | 3.15 | 3.15 | – |
| Seasonal naive | energy | 4.11 | 3.93 | 3.93 | – |

Same 12 months, same models: the frozen ensemble's MAPE is +0.48 points versus the monthly refits. That is the value of retraining: each monthly refit sees more recent data and recalibrates its intervals. It is not accuracy fading over time: the frozen ensemble's error on the same calendar months is unchanged a year later (see Drift). The seasonal-naive rows, which learn no weights, agree between the two columns, as they should.

Live so far: 7 days scored (2026-11-02 .. 2026-11-08) from 1 forecast origin(s); 0 of those days were forecast before the day began. The scored forecast was produced when the pipeline was first run (9 Nov 2026) for origin 1 Nov, using only data up to 1 Nov, by the models of the previous round; the ensemble has no live score yet. Too few days to compare; the table updates with every pipeline run.

## Ensemble and calibration (fitted on the validation months)

Ensemble weight on the LSTM by target × horizon:

|              |    1 |   2 |    3 |    4 |    5 |    6 |    7 |
|:-------------|-----:|----:|-----:|-----:|-----:|-----:|-----:|
| solar_peak   | 0.35 | 0.3 | 0.35 | 0.35 | 0.15 | 0.1  | 0.25 |
| evening_peak | 0.55 | 0.6 | 0.5  | 0.5  | 0.55 | 0.3  | 0.45 |
| energy       | 0.95 | 0.8 | 0.95 | 0.8  | 0.7  | 0.55 | 0.65 |

<!-- accuracy-round:start -->
## Before/after the accuracy round (Nov 2026)

Final recipe: **baseline + a1 + a3 + c1 + c2** — LightGBM gains temperature × weekend and CDD × season interactions and festival-aligned same-day-last-year features; LightGBM and LSTM weight training samples with a 365-day half-life. Everything else as the baseline. Chosen on the validation months only (reports/experiments.md); trained once on the same fit origins with the same seed as the baseline.

| | baseline | final | change |
|---|---|---|---|
| Validation MAPE % (Jan–Mar 2025, OOF ensemble, 3 seeds) | 1.646 | 1.567 | -0.080 |
| Validation P10–P90 coverage % | 80.2 | 80.3 | +0.1 |
| **Test MAPE %** (Apr 2025 → Nov 2026, ensemble, mean of targets) | 2.667 | 2.573 | -0.094 (95 % interval -0.159 to -0.028) |
| Test P10–P90 coverage % | 69.1 | 69.9 | +0.9 |
| Test MAPE %, LSTM alone | 2.733 | 2.521 | -0.213 |
| Test MAPE %, LightGBM alone | 2.784 | 2.748 | -0.036 |

The test interval is a paired block bootstrap over weeks of target dates (2,000 resamples).

### Rolling backtest with the final recipe (monthly refits, as production is operated)

Same 12 reported months as reports/results.md (forecasts issued from 1 Nov 2025), every model refit monthly on all data known before the month, ensemble weights and calibration fitted on past months only. This window lies inside the fixed split's test period; it was run after the final measurement and used only for the deployment decision, never to change the recipe.

| MAPE % (mean of targets) | baseline | final recipe | change |
|---|---|---|---|
| Ensemble | 2.064 | 2.141 | +0.077 |
| LSTM | 2.120 | 2.149 | +0.029 |
| LightGBM | 2.283 | 2.424 | +0.141 |
| Ensemble P10–P90 coverage % | 80.1 | 80.5 | +0.4 |

Ensemble vs Grid-India (mean of our 1–7-day forecasts per day):

| target | baseline: our error lower by | final: our error lower by | baseline win rate | final win rate |
|---|---|---|---|---|
| solar_peak | 52.9% | 49.9% | 76% | 75% |
| evening_peak | 44.3% | 41.7% | 72% | 68% |
| energy | 50.8% | 50.1% | 78% | 76% |

**Decision: the baseline stays in production.** The final recipe meets the round's rule (better on validation and on the test period, coverage not worse), but with monthly refits, which is how the production models are operated, it is worse: ensemble 2.064 → 2.141 % in the rolling backtest, with both members worse. Its gains hold for a model frozen for 19 months, not for one retrained every month. Deploying it would most likely make the live forecasts worse, so nothing was switched; to deploy it anyway: `python scripts/backtest.py --reuse --force lgbm=lgbm@compact+nov26 lstm=lstm_w@compact+nov26 && python scripts/train.py --model all && python scripts/predict.py`.

### By target — ensemble

| target       |   baseline MAPE % |   final MAPE % |   change |   baseline coverage % |   final coverage % |
|:-------------|------------------:|---------------:|---------:|----------------------:|-------------------:|
| energy       |              2.66 |           2.49 |    -0.16 |                 71.38 |              75.32 |
| evening_peak |              2.23 |           2.13 |    -0.1  |                 68.49 |              69.47 |
| solar_peak   |              3.12 |           3.1  |    -0.02 |                 67.29 |              65.04 |

### By horizon (days ahead) — ensemble

|                     |   baseline MAPE % |   final MAPE % |   change |   baseline coverage % |   final coverage % |
|:--------------------|------------------:|---------------:|---------:|----------------------:|-------------------:|
| ('energy', 1)       |              1.99 |           1.86 |    -0.13 |                 67.29 |              74.28 |
| ('energy', 2)       |              2.3  |           2.14 |    -0.16 |                 73.72 |              77.82 |
| ('energy', 3)       |              2.59 |           2.31 |    -0.28 |                 69.06 |              76.07 |
| ('energy', 4)       |              2.74 |           2.5  |    -0.24 |                 76.54 |              79.45 |
| ('energy', 5)       |              2.88 |           2.73 |    -0.15 |                 73.58 |              78.73 |
| ('energy', 6)       |              3.03 |           2.92 |    -0.11 |                 72.85 |              73.88 |
| ('energy', 7)       |              3.08 |           3    |    -0.08 |                 66.61 |              66.95 |
| ('evening_peak', 1) |              1.75 |           1.64 |    -0.11 |                 67.97 |              73.59 |
| ('evening_peak', 2) |              1.99 |           1.85 |    -0.15 |                 67.75 |              70.99 |
| ('evening_peak', 3) |              2.16 |           2.02 |    -0.14 |                 68.21 |              70.6  |
| ('evening_peak', 4) |              2.29 |           2.16 |    -0.13 |                 69.86 |              70.72 |
| ('evening_peak', 5) |              2.38 |           2.31 |    -0.07 |                 72.73 |              69.3  |
| ('evening_peak', 6) |              2.47 |           2.45 |    -0.02 |                 68.9  |              69.76 |
| ('evening_peak', 7) |              2.54 |           2.49 |    -0.04 |                 64.03 |              61.27 |
| ('solar_peak', 1)   |              2.35 |           2.27 |    -0.09 |                 71.55 |              68.48 |
| ('solar_peak', 2)   |              2.71 |           2.68 |    -0.03 |                 69.45 |              65.7  |
| ('solar_peak', 3)   |              2.92 |           2.87 |    -0.05 |                 66.15 |              65.13 |
| ('solar_peak', 4)   |              3.1  |           3.04 |    -0.06 |                 69.86 |              67.29 |
| ('solar_peak', 5)   |              3.41 |           3.38 |    -0.03 |                 66.72 |              64.84 |
| ('solar_peak', 6)   |              3.7  |           3.64 |    -0.06 |                 63.75 |              62.03 |
| ('solar_peak', 7)   |              3.66 |           3.8  |     0.14 |                 63.51 |              61.79 |

### By financial year — ensemble

|                                |   baseline MAPE % |   final MAPE % |   change |   baseline coverage % |   final coverage % |
|:-------------------------------|------------------:|---------------:|---------:|----------------------:|-------------------:|
| ('energy', 'FY 2025-26')       |              2.49 |           2.29 |    -0.19 |                 73.68 |              77.94 |
| ('energy', 'FY 2026-27')       |              2.93 |           2.82 |    -0.11 |                 67.63 |              71.04 |
| ('evening_peak', 'FY 2025-26') |              2.12 |           2.01 |    -0.11 |                 70.84 |              71.67 |
| ('evening_peak', 'FY 2026-27') |              2.4  |           2.32 |    -0.08 |                 64.67 |              65.89 |
| ('solar_peak', 'FY 2025-26')   |              2.87 |           2.82 |    -0.05 |                 69.69 |              68.15 |
| ('solar_peak', 'FY 2026-27')   |              3.52 |           3.55 |     0.02 |                 63.38 |              59.97 |

### By weather regime (observed weather) — ensemble

|                                      |   baseline MAPE % |   final MAPE % |   change |   baseline coverage % |   final coverage % |
|:-------------------------------------|------------------:|---------------:|---------:|----------------------:|-------------------:|
| ('energy', 'cloudy')                 |              3.21 |           2.87 |    -0.35 |                 59.74 |              68.83 |
| ('energy', 'cold/fog')               |              1.4  |           1.26 |    -0.14 |                 87.5  |              90.71 |
| ('energy', 'heavy rain/storm')       |              2.43 |           2.45 |     0.02 |                 79.95 |              74.18 |
| ('energy', 'hot')                    |              4.16 |           3.59 |    -0.57 |                 35.59 |              46.62 |
| ('energy', 'humid')                  |              1.97 |           2.17 |     0.2  |                 95.24 |             100    |
| ('energy', 'normal')                 |              2.37 |           2.21 |    -0.16 |                 72.99 |              77.97 |
| ('energy', 'rainy')                  |              2.65 |           2.59 |    -0.06 |                 77.39 |              80.21 |
| ('evening_peak', 'cloudy')           |              2.26 |           2.09 |    -0.17 |                 61.04 |              67.53 |
| ('evening_peak', 'cold/fog')         |              1.4  |           1.38 |    -0.02 |                 87.14 |              86.43 |
| ('evening_peak', 'heavy rain/storm') |              2.25 |           2.28 |     0.03 |                 68.68 |              64.29 |
| ('evening_peak', 'hot')              |              3.18 |           2.89 |    -0.29 |                 38.68 |              45.07 |
| ('evening_peak', 'humid')            |              1.82 |           1.71 |    -0.11 |                 76.19 |              76.19 |
| ('evening_peak', 'normal')           |              1.96 |           1.87 |    -0.09 |                 72.22 |              72.06 |
| ('evening_peak', 'rainy')            |              2.28 |           2.2  |    -0.08 |                 72.21 |              73.66 |
| ('solar_peak', 'cloudy')             |              3.6  |           3.41 |    -0.19 |                 57.14 |              62.34 |
| ('solar_peak', 'cold/fog')           |              1.57 |           1.66 |     0.1  |                 84.29 |              85    |
| ('solar_peak', 'heavy rain/storm')   |              3    |           3.11 |     0.11 |                 65.38 |              59.34 |
| ('solar_peak', 'hot')                |              4.71 |           4.43 |    -0.29 |                 38.88 |              40.43 |
| ('solar_peak', 'humid')              |              3.91 |           3.78 |    -0.13 |                 61.9  |              57.14 |
| ('solar_peak', 'normal')             |              2.55 |           2.59 |     0.04 |                 67.38 |              65.16 |
| ('solar_peak', 'rainy')              |              3.35 |           3.31 |    -0.04 |                 74.77 |              71.23 |

### Test-period access

The test period was evaluated exactly twice during this round:

- 2026-11-11 22:37:48: **baseline** — fixed-split test metrics of the production recipe (saved predictions, summarised, not recomputed) (61,320 rows)
- 2026-11-11 23:29:34: **final** — final recipe 'baseline + a1 + a3 + c1 + c2' (seed 42), trained once, scored on the test period (36,792 rows)

Every experiment ran on validation origins only; the harness refuses origins after 2025-03-24 (src/test_access.py). The baseline's test numbers were read once when the baseline was snapshotted, before any experiment; the experiment list was fixed in advance, so they did not steer the choices.
<!-- accuracy-round:end -->
