# Backtest results

Rolling-origin backtest, reported window 2025-11-01 .. 2026-11-08 (forecasts issued from 2025-11-01): models refit monthly on all data known before the month, daily forecast origins, 7 days ahead. Weather inputs are archived forecasts issued at the origin (no hindsight). Six earlier months (2025-05 .. 2025-10) were backtested the same way; their out-of-sample predictions are used only to fit the ensemble weights, the conformal calibration and the record-probability recalibration of the following months (each month uses only rows whose actuals were already published).

**Best model (lowest overall MAPE): `ensemble`.** Peaks in MW, energy in MU.

## Summary (12 reported months)

| model                |   solar-hour peak MAPE % |   evening peak MAPE % |   energy MAPE % |   overall MAPE % |   P10–P90 coverage % |   coverage before calibration % |
|:---------------------|-------------------------:|----------------------:|----------------:|-----------------:|---------------------:|--------------------------------:|
| seasonal_naive       |                     4.74 |                  3.15 |            3.93 |             3.94 |                81.87 |                           78.77 |
| lgbm                 |                     2.77 |                  1.89 |            2.3  |             2.32 |                80.59 |                           78.95 |
| lgbm@compact         |                     2.78 |                  1.81 |            2.26 |             2.28 |                80.99 |                           79.75 |
| lgbm@compact+nov26   |                     2.91 |                  2.02 |            2.35 |             2.42 |                81.22 |                           76.56 |
| lgbm@v1              |                     2.84 |                  1.9  |            2.27 |             2.34 |                81.07 |                           78.51 |
| lgbm_w               |                     2.79 |                  1.93 |            2.33 |             2.35 |                80.72 |                           77.74 |
| lgbm_w@compact       |                     2.77 |                  1.85 |            2.25 |             2.29 |                81.21 |                           78.82 |
| lgbm_w@v1            |                     2.88 |                  1.94 |            2.29 |             2.37 |                81.12 |                           78.82 |
| lstm                 |                     2.84 |                  1.96 |            2.22 |             2.34 |                78.98 |                           80.75 |
| lstm@compact         |                     2.46 |                  1.86 |            2.07 |             2.13 |                81.75 |                           81.24 |
| lstm@v1              |                     2.74 |                  1.95 |            2.23 |             2.31 |                79.44 |                           80.13 |
| lstm_w               |                     2.76 |                  1.93 |            2.23 |             2.31 |                79.69 |                           79.82 |
| lstm_w@compact       |                     2.39 |                  1.91 |            2.06 |             2.12 |                80.42 |                           80.37 |
| lstm_w@compact+nov26 |                     2.5  |                  1.87 |            2.07 |             2.15 |                80.36 |                           79.37 |
| lstm_w@v1            |                     2.7  |                  2.01 |            2.21 |             2.3  |                78.73 |                           79.79 |
| tft                  |                     3.07 |                  2.12 |            2.57 |             2.58 |                81.98 |                           79.16 |
| tft@v1               |                     2.8  |                  1.96 |            2.32 |             2.36 |                79.25 |                           82.09 |
| ensemble             |                     2.4  |                  1.79 |            2    |             2.06 |                80.09 |                           82.27 |
| grid_india           |                     4.73 |                  3.08 |            3.87 |             3.89 |               nan    |                          nan    |

Coverage = share of days inside P10–P90 (target 80 %) after per-horizon conformal calibration; the last column is the model's own interval before that step. Grid-India publishes a point forecast only.

## Model changes and acceptance rule

A change is kept only if the overall MAPE (mean over the three targets, 12 reported months) does not get worse. Each model family was backtested with three weather feature sets — `v1` (the previous round's inputs: temperature, humidity, rain, radiation, CDD), `compact` (v1 + night temperature, HDD, North-India fog, daytime cloud, dew point, heat index, gusts, 3-day rain and four anomalies) and `full` (all 81 weather columns) — each re-tuned with Optuna, with and without rare-weather sample weights (`_w`). The variant with the lowest overall MAPE is kept (ties: the simpler one).

Previous round (same 12 months): lgbm 2.302, lstm 2.203, tft 2.467.

| family | variant | weather features | rare-weather weights | overall MAPE % | kept |
|---|---|---|---|---|---|
| lgbm | lgbm@compact | compact | no | 2.283 | **yes** |
| lgbm | lgbm_w@compact | compact | yes | 2.289 |  |
| lgbm | lgbm | full | no | 2.321 |  |
| lgbm | lgbm@v1 | v1 | no | 2.339 |  |
| lgbm | lgbm_w | full | yes | 2.347 |  |
| lgbm | lgbm_w@v1 | v1 | yes | 2.370 |  |
| lgbm | lgbm@compact+nov26 | compact | no | 2.424 |  |
| lstm | lstm_w@compact | compact | yes | 2.120 | **yes** |
| lstm | lstm@compact | compact | no | 2.129 |  |
| lstm | lstm_w@compact+nov26 | compact | yes | 2.149 |  |
| lstm | lstm_w@v1 | v1 | yes | 2.303 |  |
| lstm | lstm_w | full | yes | 2.307 |  |
| lstm | lstm@v1 | v1 | no | 2.310 |  |
| lstm | lstm | full | no | 2.341 |  |
| tft | tft@v1 | v1 | no | 2.361 | **yes** |
| tft | tft | full | no | 2.585 |  |

Ensemble of lstm_w@compact + lgbm@compact: overall MAPE 2.064 vs 2.120 for the better member → kept.

## solar_peak

| model | n | MAPE % | MAE | RMSE | bias | pinball | P10–P90 coverage % |
|---|---|---|---|---|---|---|---|
| lstm_w@compact | 2583 | 2.39 | 5,582 | 7,351 | -171 | 1,886 | 81.5 |
| ensemble | 2583 | 2.40 | 5,607 | 7,468 | -508 | 1,888 | 81.3 |
| lstm@compact | 2583 | 2.46 | 5,764 | 7,508 | -976 | 1,889 | 82.6 |
| lstm_w@compact+nov26 | 2583 | 2.50 | 5,873 | 7,742 | -520 | 1,941 | 82.2 |
| lstm_w@v1 | 2583 | 2.70 | 6,343 | 8,316 | -1,988 | 2,189 | 78.7 |
| lstm@v1 | 2583 | 2.74 | 6,435 | 8,348 | -1,730 | 2,192 | 79.6 |
| lstm_w | 2583 | 2.76 | 6,553 | 8,644 | -2,298 | 2,230 | 79.7 |
| lgbm | 2583 | 2.77 | 6,523 | 8,659 | -1,793 | 2,223 | 81.8 |
| lgbm_w@compact | 2583 | 2.77 | 6,500 | 8,688 | -1,477 | 2,296 | 82.2 |
| lgbm@compact | 2583 | 2.78 | 6,533 | 8,753 | -1,532 | 2,256 | 81.9 |
| lgbm_w | 2583 | 2.79 | 6,566 | 8,733 | -1,558 | 2,277 | 81.3 |
| tft@v1 | 2583 | 2.80 | 6,508 | 8,511 | -1,225 | 2,277 | 79.4 |
| lgbm@v1 | 2583 | 2.84 | 6,661 | 8,886 | -1,611 | 2,287 | 81.8 |
| lstm | 2583 | 2.84 | 6,778 | 8,875 | -2,647 | 2,281 | 79.2 |
| lgbm_w@v1 | 2583 | 2.88 | 6,752 | 9,032 | -1,854 | 2,321 | 82.1 |
| lgbm@compact+nov26 | 2583 | 2.91 | 6,828 | 9,052 | -1,592 | 2,400 | 82.0 |
| tft | 2583 | 3.07 | 7,197 | 9,306 | -2,209 | 2,373 | 82.2 |
| grid_india | 366 | 4.73 | 10,802 | 13,676 | 2,957 | – | – |
| seasonal_naive | 2583 | 4.74 | 11,058 | 14,397 | -777 | 3,869 | 83.4 |

## evening_peak

| model | n | MAPE % | MAE | RMSE | bias | pinball | P10–P90 coverage % |
|---|---|---|---|---|---|---|---|
| ensemble | 2583 | 1.79 | 3,986 | 5,267 | -389 | 1,412 | 80.1 |
| lgbm@compact | 2583 | 1.81 | 4,033 | 5,371 | -890 | 1,458 | 80.9 |
| lgbm_w@compact | 2583 | 1.85 | 4,123 | 5,459 | -794 | 1,485 | 80.5 |
| lstm@compact | 2583 | 1.86 | 4,141 | 5,397 | -790 | 1,389 | 81.6 |
| lstm_w@compact+nov26 | 2583 | 1.87 | 4,174 | 5,490 | -168 | 1,446 | 79.6 |
| lgbm | 2583 | 1.89 | 4,222 | 5,630 | -907 | 1,475 | 79.6 |
| lgbm@v1 | 2583 | 1.90 | 4,235 | 5,635 | -744 | 1,485 | 80.5 |
| lstm_w@compact | 2583 | 1.91 | 4,244 | 5,501 | -92 | 1,462 | 80.6 |
| lgbm_w | 2583 | 1.93 | 4,293 | 5,667 | -1,072 | 1,503 | 80.0 |
| lstm_w | 2583 | 1.93 | 4,296 | 5,605 | -1,094 | 1,466 | 79.8 |
| lgbm_w@v1 | 2583 | 1.94 | 4,321 | 5,877 | -830 | 1,491 | 79.9 |
| lstm@v1 | 2583 | 1.95 | 4,354 | 5,693 | -652 | 1,489 | 80.2 |
| lstm | 2583 | 1.96 | 4,371 | 5,688 | -1,233 | 1,512 | 79.0 |
| tft@v1 | 2583 | 1.96 | 4,359 | 5,759 | -441 | 1,514 | 78.4 |
| lstm_w@v1 | 2583 | 2.01 | 4,472 | 5,863 | -812 | 1,514 | 80.2 |
| lgbm@compact+nov26 | 2583 | 2.02 | 4,476 | 6,067 | -394 | 1,580 | 80.3 |
| tft | 2583 | 2.12 | 4,731 | 6,199 | -1,109 | 1,639 | 81.6 |
| grid_india | 366 | 3.08 | 6,810 | 8,888 | 4,134 | – | – |
| seasonal_naive | 2583 | 3.15 | 7,045 | 9,737 | -378 | 2,770 | 80.9 |

## energy

| model | n | MAPE % | MAE | RMSE | bias | pinball | P10–P90 coverage % |
|---|---|---|---|---|---|---|---|
| ensemble | 2583 | 2.00 | 101 | 136 | -17 | 36 | 78.8 |
| lstm_w@compact | 2583 | 2.06 | 104 | 136 | -11 | 36 | 79.2 |
| lstm@compact | 2583 | 2.07 | 105 | 138 | -32 | 36 | 81.0 |
| lstm_w@compact+nov26 | 2583 | 2.07 | 105 | 139 | -7 | 37 | 79.3 |
| lstm_w@v1 | 2583 | 2.21 | 113 | 149 | -33 | 38 | 77.3 |
| lstm | 2583 | 2.22 | 114 | 149 | -41 | 40 | 78.7 |
| lstm_w | 2583 | 2.23 | 114 | 148 | -39 | 39 | 79.6 |
| lstm@v1 | 2583 | 2.23 | 114 | 147 | -29 | 39 | 78.6 |
| lgbm_w@compact | 2583 | 2.25 | 114 | 155 | -31 | 40 | 81.0 |
| lgbm@compact | 2583 | 2.26 | 115 | 157 | -35 | 40 | 80.1 |
| lgbm@v1 | 2583 | 2.27 | 116 | 157 | -32 | 40 | 80.9 |
| lgbm_w@v1 | 2583 | 2.29 | 117 | 157 | -37 | 41 | 81.4 |
| lgbm | 2583 | 2.30 | 117 | 156 | -40 | 40 | 80.4 |
| tft@v1 | 2583 | 2.32 | 117 | 153 | -24 | 41 | 79.9 |
| lgbm_w | 2583 | 2.33 | 119 | 157 | -35 | 40 | 80.8 |
| lgbm@compact+nov26 | 2583 | 2.35 | 119 | 160 | -29 | 43 | 81.3 |
| tft | 2583 | 2.57 | 130 | 173 | -41 | 45 | 82.1 |
| grid_india | 366 | 3.87 | 192 | 247 | 107 | – | – |
| seasonal_naive | 2583 | 3.93 | 201 | 273 | -13 | 73 | 81.3 |

## Ensemble weights (weight on lstm_w@compact; the rest on lgbm@compact), last reported month

|              |   1 |    2 |    3 |    4 |   5 |    6 |    7 |
|:-------------|----:|-----:|-----:|-----:|----:|-----:|-----:|
| solar_peak   | 0.7 | 0.85 | 0.95 | 1    | 1   | 1    | 1    |
| evening_peak | 0.5 | 0.6  | 0.6  | 0.45 | 0.2 | 0.4  | 0.35 |
| energy       | 0.6 | 0.85 | 0.95 | 0.9  | 0.8 | 0.85 | 0.9  |

Weights are re-learned every month on the previous 180 days of out-of-sample predictions (grid 0..1 in steps of 0.05, minimising MAPE of the P50).

## MAPE by horizon (days ahead)

|                                          |    1 |    2 |    3 |    4 |    5 |    6 |    7 |
|:-----------------------------------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| ('energy', 'ensemble')                   | 1.58 | 1.77 | 1.9  | 2.03 | 2.13 | 2.24 | 2.37 |
| ('energy', 'lgbm')                       | 1.69 | 1.96 | 2.16 | 2.32 | 2.49 | 2.71 | 2.81 |
| ('energy', 'lgbm@compact')               | 1.61 | 1.9  | 2.12 | 2.28 | 2.43 | 2.65 | 2.83 |
| ('energy', 'lgbm@compact+nov26')         | 1.71 | 1.99 | 2.21 | 2.36 | 2.52 | 2.74 | 2.93 |
| ('energy', 'lgbm@v1')                    | 1.62 | 1.9  | 2.15 | 2.28 | 2.46 | 2.68 | 2.85 |
| ('energy', 'lgbm_w')                     | 1.73 | 2.02 | 2.2  | 2.38 | 2.54 | 2.67 | 2.77 |
| ('energy', 'lgbm_w@compact')             | 1.65 | 1.91 | 2.12 | 2.28 | 2.42 | 2.62 | 2.76 |
| ('energy', 'lgbm_w@v1')                  | 1.71 | 1.96 | 2.2  | 2.28 | 2.48 | 2.64 | 2.78 |
| ('energy', 'lstm')                       | 1.8  | 1.94 | 2.14 | 2.25 | 2.37 | 2.46 | 2.6  |
| ('energy', 'lstm@compact')               | 1.81 | 1.89 | 1.97 | 2.05 | 2.12 | 2.24 | 2.41 |
| ('energy', 'lstm@v1')                    | 1.9  | 2.02 | 2.17 | 2.29 | 2.36 | 2.41 | 2.48 |
| ('energy', 'lstm_w')                     | 1.81 | 1.92 | 2.11 | 2.25 | 2.38 | 2.5  | 2.68 |
| ('energy', 'lstm_w@compact')             | 1.73 | 1.83 | 1.94 | 2.06 | 2.16 | 2.28 | 2.4  |
| ('energy', 'lstm_w@compact+nov26')       | 1.76 | 1.84 | 1.95 | 2.06 | 2.18 | 2.29 | 2.44 |
| ('energy', 'lstm_w@v1')                  | 1.85 | 1.95 | 2.1  | 2.25 | 2.34 | 2.42 | 2.53 |
| ('energy', 'seasonal_naive')             | 3.93 | 3.92 | 3.93 | 3.93 | 3.94 | 3.94 | 3.95 |
| ('energy', 'tft')                        | 1.96 | 2.18 | 2.37 | 2.53 | 2.77 | 2.95 | 3.21 |
| ('energy', 'tft@v1')                     | 1.84 | 2.06 | 2.24 | 2.33 | 2.47 | 2.62 | 2.7  |
| ('evening_peak', 'ensemble')             | 1.45 | 1.6  | 1.72 | 1.78 | 1.88 | 2.01 | 2.11 |
| ('evening_peak', 'lgbm')                 | 1.51 | 1.69 | 1.83 | 1.9  | 1.97 | 2.11 | 2.24 |
| ('evening_peak', 'lgbm@compact')         | 1.47 | 1.61 | 1.72 | 1.78 | 1.84 | 2.06 | 2.19 |
| ('evening_peak', 'lgbm@compact+nov26')   | 1.62 | 1.8  | 1.92 | 2.01 | 2.08 | 2.28 | 2.42 |
| ('evening_peak', 'lgbm@v1')              | 1.52 | 1.7  | 1.81 | 1.87 | 1.94 | 2.17 | 2.3  |
| ('evening_peak', 'lgbm_w')               | 1.54 | 1.72 | 1.85 | 1.93 | 1.99 | 2.16 | 2.3  |
| ('evening_peak', 'lgbm_w@compact')       | 1.48 | 1.67 | 1.78 | 1.82 | 1.89 | 2.1  | 2.23 |
| ('evening_peak', 'lgbm_w@v1')            | 1.55 | 1.73 | 1.87 | 1.92 | 1.99 | 2.2  | 2.34 |
| ('evening_peak', 'lstm')                 | 1.67 | 1.77 | 1.9  | 1.98 | 2.04 | 2.09 | 2.26 |
| ('evening_peak', 'lstm@compact')         | 1.6  | 1.69 | 1.78 | 1.86 | 1.92 | 2.02 | 2.16 |
| ('evening_peak', 'lstm@v1')              | 1.71 | 1.79 | 1.9  | 1.99 | 2.06 | 2.1  | 2.13 |
| ('evening_peak', 'lstm_w')               | 1.67 | 1.75 | 1.87 | 1.95 | 2    | 2.06 | 2.22 |
| ('evening_peak', 'lstm_w@compact')       | 1.61 | 1.71 | 1.83 | 1.93 | 2.01 | 2.1  | 2.19 |
| ('evening_peak', 'lstm_w@compact+nov26') | 1.58 | 1.69 | 1.78 | 1.88 | 1.95 | 2.04 | 2.2  |
| ('evening_peak', 'lstm_w@v1')            | 1.72 | 1.81 | 1.93 | 2.04 | 2.11 | 2.19 | 2.25 |
| ('evening_peak', 'seasonal_naive')       | 3.15 | 3.15 | 3.15 | 3.15 | 3.15 | 3.15 | 3.16 |
| ('evening_peak', 'tft')                  | 1.69 | 1.86 | 1.98 | 2.11 | 2.24 | 2.38 | 2.59 |
| ('evening_peak', 'tft@v1')               | 1.62 | 1.79 | 1.88 | 1.98 | 2.05 | 2.16 | 2.26 |
| ('solar_peak', 'ensemble')               | 1.92 | 2.11 | 2.22 | 2.35 | 2.54 | 2.73 | 2.93 |
| ('solar_peak', 'lgbm')                   | 2.07 | 2.34 | 2.55 | 2.74 | 2.99 | 3.25 | 3.44 |
| ('solar_peak', 'lgbm@compact')           | 2    | 2.31 | 2.55 | 2.74 | 3.01 | 3.33 | 3.57 |
| ('solar_peak', 'lgbm@compact+nov26')     | 2.17 | 2.47 | 2.69 | 2.89 | 3.12 | 3.43 | 3.59 |
| ('solar_peak', 'lgbm@v1')                | 2.1  | 2.37 | 2.59 | 2.76 | 3.07 | 3.39 | 3.62 |
| ('solar_peak', 'lgbm_w')                 | 2.1  | 2.39 | 2.61 | 2.77 | 3.02 | 3.22 | 3.43 |
| ('solar_peak', 'lgbm_w@compact')         | 2.02 | 2.33 | 2.55 | 2.7  | 2.98 | 3.3  | 3.52 |
| ('solar_peak', 'lgbm_w@v1')              | 2.11 | 2.38 | 2.65 | 2.86 | 3.15 | 3.43 | 3.6  |
| ('solar_peak', 'lstm')                   | 2.32 | 2.56 | 2.71 | 2.86 | 3.02 | 3.16 | 3.29 |
| ('solar_peak', 'lstm@compact')           | 2.08 | 2.22 | 2.32 | 2.41 | 2.54 | 2.73 | 2.91 |
| ('solar_peak', 'lstm@v1')                | 2.24 | 2.43 | 2.61 | 2.79 | 2.92 | 3.04 | 3.17 |
| ('solar_peak', 'lstm_w')                 | 2.31 | 2.48 | 2.63 | 2.76 | 2.91 | 3.04 | 3.19 |
| ('solar_peak', 'lstm_w@compact')         | 1.99 | 2.1  | 2.21 | 2.35 | 2.51 | 2.69 | 2.9  |
| ('solar_peak', 'lstm_w@compact+nov26')   | 2.06 | 2.15 | 2.31 | 2.48 | 2.64 | 2.84 | 3.04 |
| ('solar_peak', 'lstm_w@v1')              | 2.18 | 2.32 | 2.51 | 2.72 | 2.88 | 3.06 | 3.23 |
| ('solar_peak', 'seasonal_naive')         | 4.72 | 4.72 | 4.72 | 4.73 | 4.74 | 4.75 | 4.76 |
| ('solar_peak', 'tft')                    | 2.35 | 2.58 | 2.82 | 2.99 | 3.3  | 3.6  | 3.85 |
| ('solar_peak', 'tft@v1')                 | 2.27 | 2.48 | 2.67 | 2.78 | 2.94 | 3.15 | 3.32 |

## P10–P90 coverage by horizon: before → after conformal calibration (target 80 %)

Split-conformal per model × target × horizon on the previous 90 days of residuals (past only), with a monthly adaptive level (adaptive conformal inference, γ = 0.7): each month the quantile level moves by γ × (0.2 − last month's miss rate), using only forecasts whose actuals were already published.

|                               | h1          | h2          | h3          | h4          | h5          | h6          | h7          |
|:------------------------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|
| solar_peak / ensemble         | 87.9 → 81.5 | 86.8 → 80.3 | 83.5 → 80.8 | 81.0 → 81.0 | 78.5 → 81.8 | 76.6 → 82.0 | 73.5 → 82.0 |
| solar_peak / lstm_w@compact   | 87.1 → 80.1 | 84.9 → 80.1 | 83.5 → 81.4 | 82.7 → 81.6 | 81.0 → 82.6 | 78.2 → 81.7 | 73.8 → 82.8 |
| solar_peak / lgbm@compact     | 83.1 → 80.9 | 80.1 → 81.4 | 79.5 → 82.7 | 75.9 → 82.9 | 72.6 → 82.3 | 72.5 → 81.5 | 69.4 → 81.7 |
| solar_peak / tft@v1           | 86.6 → 79.6 | 84.9 → 80.1 | 83.8 → 78.4 | 83.2 → 79.1 | 79.9 → 78.3 | 77.9 → 78.5 | 77.3 → 82.0 |
| evening_peak / ensemble       | 88.7 → 79.0 | 87.1 → 79.8 | 84.1 → 80.3 | 82.4 → 79.1 | 81.5 → 80.7 | 80.4 → 80.9 | 76.8 → 80.9 |
| evening_peak / lstm_w@compact | 85.2 → 79.6 | 82.5 → 80.1 | 81.4 → 79.7 | 78.9 → 80.2 | 77.7 → 81.0 | 74.9 → 81.7 | 72.7 → 82.2 |
| evening_peak / lgbm@compact   | 89.0 → 79.3 | 86.8 → 80.1 | 81.9 → 81.1 | 81.6 → 80.8 | 80.2 → 81.2 | 80.4 → 82.0 | 80.9 → 82.0 |
| evening_peak / tft@v1         | 86.3 → 77.7 | 83.3 → 77.9 | 81.1 → 77.8 | 81.3 → 79.4 | 79.1 → 79.1 | 78.2 → 78.5 | 79.5 → 78.7 |
| energy / ensemble             | 88.4 → 76.1 | 87.9 → 77.4 | 83.2 → 78.6 | 82.1 → 78.6 | 80.4 → 79.6 | 79.6 → 81.2 | 76.8 → 80.3 |
| energy / lstm_w@compact       | 87.9 → 78.0 | 85.7 → 77.6 | 82.2 → 79.2 | 79.7 → 78.9 | 78.5 → 79.1 | 76.3 → 80.7 | 72.7 → 80.9 |
| energy / lgbm@compact         | 86.6 → 79.0 | 81.1 → 79.8 | 79.5 → 78.9 | 80.2 → 79.9 | 79.3 → 80.2 | 76.6 → 81.2 | 77.6 → 82.0 |
| energy / tft@v1               | 90.3 → 77.4 | 86.3 → 79.2 | 83.2 → 80.3 | 81.3 → 80.2 | 81.0 → 80.2 | 79.8 → 80.9 | 79.2 → 81.1 |

Range across horizons/targets for these models: before 69.4–90.3 % (mean |coverage − 80| = 3.5 pts); after 76.1–82.9 % (mean |coverage − 80| = 1.2 pts). Plain split-conformal without the adaptive step: 77.9–85.9 % (mean |coverage − 80| = 2.2 pts) — it over-covers because errors in the preceding months (summer, smaller training set) are larger than in the months being forecast. The window (90 d) and γ were chosen by comparing a few settings on this backtest, so the 'after' coverage is slightly optimistic.

## MAPE by weather regime (observed weather; evaluation only)

Each day gets one label, first match in this order: heavy rain/storm (all-India rain ≥ 14.0 mm or gusts ≥ 43 km/h, top 5 %), cold/fog (Nov–Feb, North-India night ≤ 8 °C or fog proxy ≥ 0.3), hot (all-India max ≥ 37.4 °C or heat index ≥ 40.1 °C, top 10 %), rainy (≥ 2.5 mm), cloudy (daytime cloud ≥ 60 %), humid (dew point ≥ 22 °C), normal. `days` = distinct target days.

### solar_peak

| label            |   days |   grid_india |   seasonal_naive |   lgbm@compact |   lstm_w@compact |   tft@v1 |   ensemble |
|:-----------------|-------:|-------------:|-----------------:|---------------:|-----------------:|---------:|-----------:|
| heavy rain/storm |     32 |         6.92 |             7.48 |           2.89 |             2.64 |     3.12 |       2.68 |
| cold/fog         |     40 |         3.7  |             2.35 |           1.78 |             2.07 |     1.95 |       1.94 |
| hot              |     39 |         5.16 |             6.04 |           3.18 |             2.94 |     3.27 |       2.83 |
| rainy            |     90 |         4.59 |             4.99 |           3.3  |             2.5  |     2.89 |       2.62 |
| cloudy           |      6 |         3.97 |             4.19 |           3.36 |             2.39 |     3.18 |       2.58 |
| humid            |      3 |         5.14 |             5.31 |           3.56 |             2.9  |     2.17 |       2.9  |
| normal           |    156 |         4.54 |             4.31 |           2.56 |             2.2  |     2.78 |       2.19 |

### evening_peak

| label            |   days |   grid_india |   seasonal_naive |   lgbm@compact |   lstm_w@compact |   tft@v1 |   ensemble |
|:-----------------|-------:|-------------:|-----------------:|---------------:|-----------------:|---------:|-----------:|
| heavy rain/storm |     32 |         4.8  |             4.69 |           2.09 |             2.32 |     2.22 |       2.09 |
| cold/fog         |     40 |         1.76 |             2    |           1.42 |             1.92 |     1.5  |       1.66 |
| hot              |     39 |         2.38 |             4.51 |           2.24 |             2.22 |     2.33 |       2.12 |
| rainy            |     90 |         3.77 |             2.76 |           1.95 |             1.91 |     1.97 |       1.88 |
| cloudy           |      6 |         1.06 |             2.83 |           2.17 |             1.87 |     2    |       2.02 |
| humid            |      3 |         4.64 |             2.57 |           1.73 |             1.28 |     1    |       1.52 |
| normal           |    156 |         2.88 |             3.05 |           1.64 |             1.76 |     1.94 |       1.62 |

### energy

| label            |   days |   grid_india |   seasonal_naive |   lgbm@compact |   lstm_w@compact |   tft@v1 |   ensemble |
|:-----------------|-------:|-------------:|-----------------:|---------------:|-----------------:|---------:|-----------:|
| heavy rain/storm |     32 |         5.38 |             6.13 |           2.24 |             2.21 |     2.4  |       2.12 |
| cold/fog         |     40 |         1.78 |             2.03 |           1.48 |             1.75 |     1.45 |       1.6  |
| hot              |     39 |         3.7  |             5.32 |           3.02 |             2.45 |     2.86 |       2.51 |
| rainy            |     90 |         4.32 |             3.86 |           2.52 |             2.25 |     2.45 |       2.25 |
| cloudy           |      6 |         2.15 |             3.39 |           2.51 |             2.03 |     2.81 |       2.08 |
| humid            |      3 |         4.17 |             3.26 |           1.7  |             1.53 |     1.33 |       1.6  |
| normal           |    156 |         3.94 |             3.71 |           2.11 |             1.9  |     2.32 |       1.8  |

## MAPE by season

### solar_peak

| label                        |   days |   grid_india |   seasonal_naive |   lgbm@compact |   lstm_w@compact |   tft@v1 |   ensemble |
|:-----------------------------|-------:|-------------:|-----------------:|---------------:|-----------------:|---------:|-----------:|
| winter (Dec–Feb)             |     90 |         3.38 |             2.15 |           1.71 |             1.9  |     1.78 |       1.81 |
| pre-monsoon summer (Mar–May) |     92 |         6.76 |             6.62 |           3.28 |             3.07 |     3.31 |       2.99 |
| monsoon (Jun–Sep)            |    115 |         4.68 |             5.35 |           3.18 |             2.46 |     2.82 |       2.55 |
| post-monsoon (Oct–Nov)       |     69 |         3.87 |             4.49 |           2.83 |             2    |     3.45 |       2.08 |

### evening_peak

| label                        |   days |   grid_india |   seasonal_naive |   lgbm@compact |   lstm_w@compact |   tft@v1 |   ensemble |
|:-----------------------------|-------:|-------------:|-----------------:|---------------:|-----------------:|---------:|-----------:|
| winter (Dec–Feb)             |     90 |         1.96 |             1.71 |           1.28 |             1.6  |     1.3  |       1.44 |
| pre-monsoon summer (Mar–May) |     92 |         3.8  |             5.08 |           2.31 |             2.37 |     2.56 |       2.2  |
| monsoon (Jun–Sep)            |    115 |         3.56 |             3.05 |           1.88 |             1.86 |     1.8  |       1.83 |
| post-monsoon (Oct–Nov)       |     69 |         2.76 |             2.61 |           1.7  |             1.78 |     2.32 |       1.63 |

### energy

| label                        |   days |   grid_india |   seasonal_naive |   lgbm@compact |   lstm_w@compact |   tft@v1 |   ensemble |
|:-----------------------------|-------:|-------------:|-----------------:|---------------:|-----------------:|---------:|-----------:|
| winter (Dec–Feb)             |     90 |         1.77 |             1.75 |           1.28 |             1.45 |     1.39 |       1.33 |
| pre-monsoon summer (Mar–May) |     92 |         5.21 |             6.19 |           2.98 |             2.72 |     3.04 |       2.65 |
| monsoon (Jun–Sep)            |    115 |         4.07 |             4.11 |           2.5  |             2.15 |     2.3  |       2.19 |
| post-monsoon (Oct–Nov)       |     69 |         4.48 |             3.45 |           2.15 |             1.79 |     2.63 |       1.65 |

## Before / after: previous models vs the all-weather models

Same reported window and same day labels. *Before* = the models of the previous backtest (temperature, humidity, rain and radiation features only, no ensemble or per-horizon calibration).

### solar_peak

| label               |   days |   lgbm (before) |   lgbm@compact (after) |   lstm (before) |   lstm_w@compact (after) |   tft (before) |   tft@v1 (after) |   ensemble (after) |
|:--------------------|-------:|----------------:|-----------------------:|----------------:|-------------------------:|---------------:|-----------------:|-------------------:|
| all days            |    365 |            2.82 |                   2.78 |            2.64 |                     2.39 |           2.95 |             2.8  |               2.4  |
| heavy rain/storm    |     32 |            2.97 |                   2.89 |            3.79 |                     2.64 |           3.56 |             3.12 |               2.68 |
| cold/fog            |     40 |            1.56 |                   1.78 |            1.8  |                     2.07 |           1.97 |             1.95 |               1.94 |
| hot                 |     39 |            3.35 |                   3.18 |            4.06 |                     2.94 |           3.78 |             3.27 |               2.83 |
| rainy               |     94 |            3.31 |                   3.3  |            2.7  |                     2.5  |           3.18 |             2.89 |               2.62 |
| cloudy              |      7 |            3.57 |                   3.36 |            2.74 |                     2.39 |           2.93 |             3.18 |               2.58 |
| humid               |      3 |            3.3  |                   3.56 |            3.04 |                     2.9  |           2.04 |             2.17 |               2.9  |
| normal              |    150 |            2.62 |                   2.56 |            2.19 |                     2.2  |           2.73 |             2.78 |               2.19 |
| hottest 5 % of days |     22 |            3.48 |                   3.29 |            4.89 |                     2.98 |           3.99 |             3.53 |               2.94 |

### evening_peak

| label               |   days |   lgbm (before) |   lgbm@compact (after) |   lstm (before) |   lstm_w@compact (after) |   tft (before) |   tft@v1 (after) |   ensemble (after) |
|:--------------------|-------:|----------------:|-----------------------:|----------------:|-------------------------:|---------------:|-----------------:|-------------------:|
| all days            |    365 |            1.84 |                   1.81 |            1.88 |                     1.91 |           2.03 |             1.96 |               1.79 |
| heavy rain/storm    |     32 |            2.14 |                   2.09 |            2.52 |                     2.32 |           2.43 |             2.22 |               2.09 |
| cold/fog            |     40 |            1.28 |                   1.42 |            1.7  |                     1.92 |           1.56 |             1.5  |               1.66 |
| hot                 |     39 |            2.24 |                   2.24 |            2.53 |                     2.22 |           2.53 |             2.33 |               2.12 |
| rainy               |     94 |            1.98 |                   1.95 |            1.85 |                     1.91 |           2.04 |             1.97 |               1.88 |
| cloudy              |      7 |            2.26 |                   2.17 |            1.86 |                     1.87 |           1.64 |             2    |               2.02 |
| humid               |      3 |            1.82 |                   1.73 |            1.37 |                     1.28 |           1.42 |             1    |               1.52 |
| normal              |    150 |            1.72 |                   1.64 |            1.64 |                     1.76 |           1.97 |             1.94 |               1.62 |
| hottest 5 % of days |     22 |            1.92 |                   2.11 |            2.92 |                     2.31 |           2.5  |             2.48 |               2.07 |

### energy

| label               |   days |   lgbm (before) |   lgbm@compact (after) |   lstm (before) |   lstm_w@compact (after) |   tft (before) |   tft@v1 (after) |   ensemble (after) |
|:--------------------|-------:|----------------:|-----------------------:|----------------:|-------------------------:|---------------:|-----------------:|-------------------:|
| all days            |    365 |            2.25 |                   2.26 |            2.09 |                     2.06 |           2.42 |             2.32 |               2    |
| heavy rain/storm    |     32 |            2.32 |                   2.24 |            2.88 |                     2.21 |           2.76 |             2.4  |               2.12 |
| cold/fog            |     40 |            1.5  |                   1.48 |            1.42 |                     1.75 |           1.61 |             1.45 |               1.6  |
| hot                 |     39 |            2.98 |                   3.02 |            3.21 |                     2.45 |           2.81 |             2.86 |               2.51 |
| rainy               |     94 |            2.51 |                   2.52 |            2.12 |                     2.25 |           2.56 |             2.45 |               2.25 |
| cloudy              |      7 |            2.38 |                   2.51 |            2.03 |                     2.03 |           2.32 |             2.81 |               2.08 |
| humid               |      3 |            1.32 |                   1.7  |            1    |                     1.53 |           1.36 |             1.33 |               1.6  |
| normal              |    150 |            2.09 |                   2.11 |            1.82 |                     1.9  |           2.4  |             2.32 |               1.8  |
| hottest 5 % of days |     22 |            2.4  |                   2.54 |            3.61 |                     2.19 |           2.43 |             2.79 |               2.12 |

### The heatwave weakness

- solar_peak: before, LSTM 4.89 % vs LightGBM 3.48 % on the hottest 5 % of days; after, lstm_w@compact 2.98 % vs lgbm@compact 3.29 %, ensemble 2.94 %.
- energy: before, LSTM 3.61 % vs LightGBM 2.40 % on the hottest 5 % of days; after, lstm_w@compact 2.19 % vs lgbm@compact 2.54 %, ensemble 2.12 %.
- evening_peak: before, LSTM 2.92 % vs LightGBM 1.92 % on the hottest 5 % of days; after, lstm_w@compact 2.31 % vs lgbm@compact 2.11 %, ensemble 2.07 %.

## Worst days for ensemble (mean APE over the 7 forecasts of that day), with weather and holidays

| date       | target       |   ape | day   | weather   | holiday          | festival   |   tmax_ai |   tmax_anom |   rain_ai_mm |   rain_max_region_mm |   gust_ai_kmh |   cloud_day_ai |
|:-----------|:-------------|------:|:------|:----------|:-----------------|:-----------|----------:|------------:|-------------:|---------------------:|--------------:|---------------:|
| 2026-03-04 | evening_peak | 11.54 | Wed   | normal    | national holiday | Holi       |      34.1 |         2.7 |          0   |                  0   |            36 |              1 |
| 2026-05-01 | solar_peak   | 10.46 | Fri   | normal    | national holiday |            |      35.9 |        -1.2 |          0.1 |                  8.3 |            37 |             25 |
| 2026-03-20 | solar_peak   |  9.79 | Fri   | rainy     |                  |            |      28.9 |        -4.7 |          3.3 |                 16.8 |            35 |             47 |
| 2026-11-08 | evening_peak |  9.75 | Sun   | normal    | national holiday | Diwali     |      31.5 |         1.4 |          0.8 |                  3.2 |            27 |             19 |
| 2026-03-04 | solar_peak   |  9.69 | Wed   | normal    | national holiday | Holi       |      34.1 |         2.7 |          0   |                  0   |            36 |              1 |
| 2026-03-20 | energy       |  9.21 | Fri   | rainy     |                  |            |      28.9 |        -4.7 |          3.3 |                 16.8 |            35 |             47 |
| 2026-01-26 | solar_peak   |  8.82 | Mon   | cold/fog  | national holiday |            |      26.1 |        -0.7 |          0.6 |                  2.3 |            27 |             46 |
| 2026-07-16 | solar_peak   |  8.75 | Thu   | rainy     |                  |            |      33.8 |         2   |          3.1 |                 11.4 |            38 |             73 |
| 2026-07-14 | solar_peak   |  8.72 | Tue   | cloudy    |                  |            |      34.2 |         2.3 |          1.6 |                 18.7 |            39 |             71 |
| 2026-06-27 | solar_peak   |  8.4  | Sat   | hot       |                  |            |      35.4 |         1.7 |          2.5 |                  5.4 |            32 |             41 |
| 2026-05-01 | energy       |  8.27 | Fri   | normal    | national holiday |            |      35.9 |        -1.2 |          0.1 |                  8.3 |            37 |             25 |
| 2026-05-04 | evening_peak |  8.26 | Mon   | rainy     |                  |            |      34.9 |        -2.3 |          3.5 |                  8.9 |            40 |             35 |
| 2026-07-15 | solar_peak   |  8.26 | Wed   | rainy     |                  |            |      33.8 |         2   |          3.4 |                 35.5 |            39 |             58 |
| 2026-05-31 | solar_peak   |  8.25 | Sun   | normal    |                  |            |      35.5 |        -0.9 |          1.4 |                  2   |            39 |             35 |
| 2026-05-03 | solar_peak   |  8.11 | Sun   | normal    |                  |            |      36.4 |        -0.8 |          2.3 |                 19.2 |            41 |             35 |

`tmax_anom` = all-India max temperature minus its normal for the date; rain is demand-weighted all-India and the wettest region; all observed (reanalysis), for explanation only.

## Record-peak probability: raw vs recalibrated (model ensemble, 366 origins)

Recalibration maps are fitted month by month on earlier origins whose 7-day outcome was already known (Platt = logistic on logit(p); isotonic = monotone step function).

| record | method | events | Brier | climatology | skill |
|---|---|---|---|---|---|
| all-time | raw | 18 | 0.0543 | 0.0468 | -0.160 |
| all-time | platt | 18 | 0.0613 | 0.0468 | -0.310 |
| all-time | isotonic | 18 | 0.0637 | 0.0468 | -0.361 |
| year | raw | 26 | 0.1023 | 0.0660 | -0.550 |
| year | platt | 26 | 0.0501 | 0.0660 | +0.241 |
| year | isotonic | 26 | 0.0631 | 0.0660 | +0.043 |

Live gauges use Platt scaling. Status: **Experimental — not yet reliable** (skill must exceed 0.05 for both records).

## Head-to-head vs Grid-India (original basis)

Our mean of the 1–7-day-ahead forecasts for each day vs Grid-India's value for that day. This basis favours our model (Grid-India's forecast is prepared ~5 days before its week starts); see [fairness_audit.md](fairness_audit.md) for the matched comparison.

| target | days | our MAE | Grid-India MAE | improvement | win rate |
|---|---|---|---|---|---|
| energy | 365 | 93 | 191 | +51.1% | 78% |
| evening_peak | 365 | 3,755 | 6,777 | +44.6% | 72% |
| solar_peak | 365 | 5,064 | 10,779 | +53.0% | 76% |

## Compute

Device: GPU: NVIDIA RTX 3050 Ti Laptop (4 GB), torch 2.14.1+cu130. Backtest wall time (18 monthly refits incl. warm-up): seasonal_naive 0.0 min, lgbm 1.0 min, lstm 2.3 min, tft 9.5 min, lgbm_w 1.6 min, lstm_w 2.3 min, lgbm@v1 0.9 min, lgbm_w@v1 1.5 min, lgbm@compact 1.3 min, lgbm_w@compact 2.4 min, lstm@v1 4.7 min, lstm_w@v1 4.7 min, lstm@compact 3.7 min, lstm_w@compact 3.8 min, tft@v1 4.5 min, lgbm@compact+nov26 3.4 min, lstm_w@compact+nov26 5.2 min

