# Backtest results

Rolling-origin backtest 2025-11-01 .. 2026-11-01: models refit monthly on all data known before the month, daily forecast origins, 7-day horizon. Weather inputs are the archived forecasts issued at the origin (no hindsight). Peaks in MW (errors shown in MW), energy in MU.

**Best model (lowest scaled pinball across targets): `lstm`**

## solar_peak

| model | n | MAPE % | MAE | RMSE | bias | pinball | P10–P90 coverage % |
|---|---|---|---|---|---|---|---|
| lstm | 2534 | 2.64 | 6,212 | 8,167 | -1,473 | 2,016 | 78.5 |
| lgbm | 2534 | 2.82 | 6,615 | 8,865 | -1,929 | 2,230 | 77.7 |
| tft | 2534 | 2.95 | 6,866 | 9,118 | -952 | 2,282 | 78.2 |
| seasonal_naive | 2534 | 4.78 | 11,181 | 14,493 | -883 | 3,629 | 77.1 |
| grid_india | 359 | 4.79 | 10,941 | 13,787 | 2,898 | – | – |

## evening_peak

| model | n | MAPE % | MAE | RMSE | bias | pinball | P10–P90 coverage % |
|---|---|---|---|---|---|---|---|
| lgbm | 2534 | 1.84 | 4,110 | 5,442 | -742 | 1,358 | 80.7 |
| lstm | 2534 | 1.88 | 4,195 | 5,506 | -465 | 1,374 | 78.9 |
| tft | 2534 | 2.03 | 4,546 | 5,997 | -47 | 1,451 | 77.6 |
| grid_india | 359 | 3.05 | 6,775 | 8,812 | 4,019 | – | – |
| seasonal_naive | 2534 | 3.13 | 7,026 | 9,689 | -495 | 2,378 | 79.8 |

## energy

| model | n | MAPE % | MAE | RMSE | bias | pinball | P10–P90 coverage % |
|---|---|---|---|---|---|---|---|
| lstm | 2534 | 2.09 | 107 | 141 | -23 | 35 | 78.9 |
| lgbm | 2534 | 2.25 | 115 | 155 | -32 | 38 | 80.5 |
| tft | 2534 | 2.42 | 123 | 163 | -13 | 40 | 79.1 |
| grid_india | 359 | 3.84 | 191 | 246 | 102 | – | – |
| seasonal_naive | 2534 | 3.95 | 202 | 274 | -15 | 68 | 79.3 |

Grid-India rows: only days with a published week-ahead forecast; it is a point forecast (no quantiles). Energy for Grid-India = integral of its 15-min forecast curve.

## MAPE by horizon (days ahead)

|                                    |    1 |    2 |    3 |    4 |    5 |    6 |    7 |
|:-----------------------------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| ('energy', 'lgbm')                 | 1.61 | 1.86 | 2.12 | 2.26 | 2.44 | 2.63 | 2.83 |
| ('energy', 'lstm')                 | 1.71 | 1.84 | 2    | 2.14 | 2.24 | 2.32 | 2.39 |
| ('energy', 'seasonal_naive')       | 3.94 | 3.94 | 3.94 | 3.95 | 3.95 | 3.96 | 3.96 |
| ('energy', 'tft')                  | 1.82 | 2.09 | 2.32 | 2.45 | 2.62 | 2.77 | 2.89 |
| ('evening_peak', 'lgbm')           | 1.48 | 1.64 | 1.75 | 1.82 | 1.89 | 2.08 | 2.24 |
| ('evening_peak', 'lstm')           | 1.57 | 1.69 | 1.82 | 1.92 | 1.99 | 2.05 | 2.1  |
| ('evening_peak', 'seasonal_naive') | 3.13 | 3.13 | 3.13 | 3.13 | 3.13 | 3.13 | 3.14 |
| ('evening_peak', 'tft')            | 1.61 | 1.83 | 1.91 | 2.06 | 2.15 | 2.27 | 2.41 |
| ('solar_peak', 'lgbm')             | 2.06 | 2.38 | 2.6  | 2.78 | 3.05 | 3.33 | 3.55 |
| ('solar_peak', 'lstm')             | 2.05 | 2.26 | 2.47 | 2.69 | 2.85 | 3.03 | 3.15 |
| ('solar_peak', 'seasonal_naive')   | 4.77 | 4.77 | 4.77 | 4.78 | 4.79 | 4.8  | 4.81 |
| ('solar_peak', 'tft')              | 2.31 | 2.58 | 2.81 | 2.97 | 3.14 | 3.33 | 3.49 |

## Coverage of the P10–P90 interval by horizon (target 80 %)

|                                    |    1 |    2 |    3 |    4 |    5 |    6 |    7 |
|:-----------------------------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| ('energy', 'lgbm')                 | 87.9 | 84.6 | 79.1 | 78.2 | 77.6 | 77.8 | 78.3 |
| ('energy', 'lstm')                 | 84.7 | 82.7 | 79.3 | 78.7 | 76.5 | 76.4 | 73.8 |
| ('energy', 'seasonal_naive')       | 79.5 | 79.4 | 79.3 | 79.3 | 79.2 | 79.2 | 79.1 |
| ('energy', 'tft')                  | 87.4 | 82.4 | 80.4 | 80.1 | 76.2 | 73.3 | 73.5 |
| ('evening_peak', 'lgbm')           | 87.1 | 83.2 | 79.3 | 80.1 | 78.7 | 78.1 | 78.6 |
| ('evening_peak', 'lstm')           | 84.1 | 83   | 81.3 | 79.3 | 77.3 | 75   | 72.1 |
| ('evening_peak', 'seasonal_naive') | 80   | 79.9 | 79.9 | 79.8 | 79.8 | 79.7 | 79.7 |
| ('evening_peak', 'tft')            | 85.8 | 81.6 | 78.2 | 75.4 | 74.2 | 74.4 | 73.3 |
| ('solar_peak', 'lgbm')             | 86.3 | 81.6 | 80.4 | 76.5 | 72.9 | 72.8 | 73.5 |
| ('solar_peak', 'lstm')             | 87.7 | 85.7 | 80.2 | 76.5 | 74.2 | 73.9 | 71.3 |
| ('solar_peak', 'seasonal_naive')   | 77.3 | 77.2 | 77.1 | 77.1 | 77   | 76.9 | 76.9 |
| ('solar_peak', 'tft')              | 85.5 | 84.1 | 80.2 | 76.8 | 75.6 | 73.1 | 72.1 |

## Head-to-head vs Grid-India (same days)

| target | our model | days | our MAE | Grid-India MAE | improvement | win rate |
|---|---|---|---|---|---|---|
| energy | lstm (mean of h=1..7) | 358 | 99 | 190 | +47.9% | 74% |
| evening_peak | lstm (mean of h=1..7) | 358 | 3,987 | 6,741 | +40.9% | 68% |
| solar_peak | lstm (mean of h=1..7) | 358 | 5,723 | 10,918 | +47.6% | 78% |
| energy | lstm @ h=7 | 352 | 118 | 189 | +37.5% | 66% |
| evening_peak | lstm @ h=7 | 352 | 4,642 | 6,783 | +31.6% | 66% |
| solar_peak | lstm @ h=7 | 352 | 7,118 | 10,968 | +35.1% | 68% |

## Record-peak probability calibration (all-time record, model lstm)

Origins: 359, events: 18. Brier 0.0478 vs climatology 0.0476 (skill -0.00).

| bin       |   n |   mean_predicted |   observed_freq |
|:----------|----:|-----------------:|----------------:|
| 0.00–0.02 | 193 |            0.002 |           0.026 |
| 0.02–0.10 |  76 |            0.048 |           0.039 |
| 0.10–0.25 |  58 |            0.16  |           0.052 |
| 0.25–0.50 |  28 |            0.343 |           0.214 |
| 0.50–0.75 |   4 |            0.562 |           0.25  |

## Record-peak probability calibration (year record, model lstm)

Origins: 359, events: 26. Brier 0.0719 vs climatology 0.0672 (skill -0.07).

| bin       |   n |   mean_predicted |   observed_freq |
|:----------|----:|-----------------:|----------------:|
| 0.00–0.02 | 160 |            0.002 |           0.019 |
| 0.02–0.10 |  88 |            0.047 |           0.034 |
| 0.10–0.25 |  38 |            0.151 |           0.105 |
| 0.25–0.50 |  39 |            0.398 |           0.179 |
| 0.50–0.75 |  26 |            0.567 |           0.038 |
| 0.90–1.00 |   8 |            0.98  |           1     |

## Compute

Device: GPU: NVIDIA RTX 3050 Ti Laptop (4 GB), torch 2.14.1+cu130. Backtest wall time per model (12 monthly refits): seasonal_naive 0.0 min, lgbm 0.8 min, lstm 3.8 min, tft 3.3 min

## Where models fail (MAPE % by day type)

|                                             |   grid_india |   lgbm |   lstm |   tft |
|:--------------------------------------------|-------------:|-------:|-------:|------:|
| ('energy', 'Sundays')                       |         4.55 |   2.15 |   2.02 |  2.2  |
| ('energy', 'all days')                      |         3.84 |   2.25 |   2.09 |  2.42 |
| ('energy', 'festival ±1 day')               |         4.94 |   3.06 |   3.2  |  3.11 |
| ('energy', 'heatwave (top 5 % temp)')       |         4.32 |   2.4  |   3.61 |  2.43 |
| ('energy', 'monsoon (Jul–Sep)')             |         4.56 |   2.32 |   1.99 |  2.24 |
| ('energy', 'national holidays')             |         4.34 |   3.8  |   3.11 |  4.01 |
| ('energy', 'normal weekdays')               |         3.52 |   2.13 |   1.97 |  2.33 |
| ('evening_peak', 'Sundays')                 |         3.65 |   1.96 |   2.23 |  1.98 |
| ('evening_peak', 'all days')                |         3.05 |   1.84 |   1.88 |  2.03 |
| ('evening_peak', 'festival ±1 day')         |         4.08 |   2.83 |   2.96 |  2.81 |
| ('evening_peak', 'heatwave (top 5 % temp)') |         2.73 |   1.92 |   2.92 |  2.5  |
| ('evening_peak', 'monsoon (Jul–Sep)')       |         4.13 |   1.77 |   1.68 |  1.78 |
| ('evening_peak', 'national holidays')       |         3.42 |   3.85 |   3.47 |  4.08 |
| ('evening_peak', 'normal weekdays')         |         2.81 |   1.7  |   1.72 |  1.92 |
| ('solar_peak', 'Sundays')                   |         5.49 |   2.75 |   2.75 |  2.97 |
| ('solar_peak', 'all days')                  |         4.79 |   2.82 |   2.64 |  2.95 |
| ('solar_peak', 'festival ±1 day')           |         6.09 |   3.46 |   3.5  |  3.32 |
| ('solar_peak', 'heatwave (top 5 % temp)')   |         5.9  |   3.48 |   4.89 |  3.99 |
| ('solar_peak', 'monsoon (Jul–Sep)')         |         5.13 |   3.4  |   2.98 |  3.15 |
| ('solar_peak', 'national holidays')         |         5.41 |   4.64 |   3.71 |  4.53 |
| ('solar_peak', 'normal weekdays')           |         4.44 |   2.71 |   2.52 |  2.82 |

## Worst days for lstm (mean APE over the 7 forecasts of that day)

| date                | target       |   ape | day   |   holiday/festival |   tmax_ai |
|:--------------------|:-------------|------:|:------|-------------------:|----------:|
| 2026-03-04 00:00:00 | evening_peak | 10.85 | Wed   |                  1 |      34.1 |
| 2026-03-20 00:00:00 | energy       | 10.46 | Fri   |                  0 |      28.9 |
| 2026-03-20 00:00:00 | solar_peak   | 10.29 | Fri   |                  0 |      28.9 |
| 2026-07-14 00:00:00 | solar_peak   |  8.97 | Tue   |                  0 |      34.2 |
| 2026-04-25 00:00:00 | solar_peak   |  8.94 | Sat   |                  0 |      39.8 |
| 2026-07-16 00:00:00 | solar_peak   |  8.92 | Thu   |                  0 |      33.8 |
| 2026-03-19 00:00:00 | evening_peak |  8.58 | Thu   |                  0 |      31.8 |
| 2026-05-01 00:00:00 | solar_peak   |  8.55 | Fri   |                  1 |      35.9 |
| 2026-07-15 00:00:00 | solar_peak   |  8.46 | Wed   |                  0 |      33.8 |
| 2026-04-27 00:00:00 | solar_peak   |  8.36 | Mon   |                  0 |      39.1 |
| 2026-06-27 00:00:00 | solar_peak   |  8.17 | Sat   |                  0 |      35.4 |
| 2026-09-25 00:00:00 | solar_peak   |  8.12 | Fri   |                  0 |      30.6 |

