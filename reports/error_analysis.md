# Error analysis (validation, Jan–Mar 2025)

Baseline recipe (production): LSTM + LightGBM ensemble, weights per target × horizon. Forecasts from 2025-01-01 to 2025-03-24 (83 origins, 1–7 days ahead), seed 42. The ensemble and its P10–P90 calibration are fitted out of fold (January ↔ February–March), the way every experiment of this round is scored. Errors are signed % of the actual (+ = forecast too high). No test-period data is used anywhere in this report.

## Findings: the three biggest error sources

1. **Festivals and the days around them.** Holiday/festival days and the 3 days around them are 19 % of the validation rows but 31 % of the error, with a systematic over-forecast (+1.9 % on those rows; Holi on 14 Mar 2025 and Makar Sankranti on 14 Jan lead the worst-day list). The festival season confirms it: Oct–Dec 2024 festival rows are 24 % of the rows, 35 % of the error, bias +1.9 % (Dussehra, Diwali week).
   *Hypothesis:* each festival occurs only once or twice in the training data and the calendar flags are binary, so the model learns one generic holiday dip and cannot size it per festival. The same festival last year, aligned on the festival date rather than on the weekday, is a direct analogue of its size and of the days around it (experiment a, `lag365`).

2. **Level drift that grows with the horizon.** Ensemble MAPE rises from 1.29 % one day ahead to 1.84 % at seven days, and some months carry a systematic bias (energy -1.4 % in January, solar-hour +1.1 % in March; Oct 2024 +2.2 % energy).
   *Hypothesis:* the label is demand relative to the last 7 days, so the model must infer where the level is heading at seasonal turning points (winter peak, the March warm-up, the post-monsoon decline) from weather alone. Year-on-year growth and momentum features (experiment a, `growth`), a different label base (b) and weighting recent months (c) each attack this directly.

3. **Weather-sensitive transition days.** Cold/fog days score 1.97 % MAPE against 1.54 % on normal days, and the March heat spell (tmax about 3 °C above normal) coincides with a solar-hour over-forecast.
   *Hypothesis:* the temperature response differs by season and by weekday (cooling load on a working day vs a Sunday), which a model with additive weather effects and few examples per season captures poorly. Season × degree-day and temperature × weekend interactions (experiment a, `interactions`) test this.

**Also found: the 7-day-ahead weather inputs are biased.** Both members jump to an over-forecast at h = 7 in both windows. The archived Open-Meteo forecasts issued 7 days ahead are much further from the observed weather than leads 1–6 (training period, forecast minus observed):

|    |   tmax_ai |   tmin_nr |   precip_ai |   cdd_ai |
|---:|----------:|----------:|------------:|---------:|
|  1 |      0.67 |      2.01 |       -0.25 |     0.86 |
|  2 |      0.73 |      1.97 |       -0.59 |     0.87 |
|  3 |      0.81 |      1.93 |       -0.6  |     0.9  |
|  4 |      0.87 |      1.99 |       -0.74 |     0.96 |
|  5 |      0.86 |      1.98 |       -0.5  |     0.98 |
|  6 |      0.96 |      1.99 |       -0.77 |     1.03 |
|  7 |      1.85 |      4.45 |       -2.63 |     1.83 |

All 24 hours are present (checked on a sample), so the day-7 series is a different, coarser forecast, not a truncated day. *Hypothesis:* removing the lead-7 offset (measured on the training period only) removes the h = 7 bias. Whether today's live day-7 forecast carries the same offset cannot be checked yet (7 live days). Tested as an extra experiment (a0) before the planned list.

## By model and target — MAPE %

| model    |   energy |   evening_peak |   solar_peak |   mean |
|:---------|---------:|---------------:|-------------:|-------:|
| ensemble |     1.68 |           1.4  |         1.73 |   1.6  |
| lgbm     |     1.88 |           1.51 |         1.74 |   1.71 |
| lstm     |     1.69 |           1.51 |         2.01 |   1.74 |

## By horizon

|   h |   MAPE solar_peak |   MAPE evening_peak |   MAPE energy |   MAPE mean |   bias solar_peak |   bias evening_peak |   bias energy |   days |
|----:|------------------:|--------------------:|--------------:|------------:|------------------:|--------------------:|--------------:|-------:|
|   1 |              1.41 |                1.14 |          1.31 |        1.29 |              0.12 |               -0.15 |         -0.42 |     83 |
|   2 |              1.56 |                1.27 |          1.48 |        1.44 |              0.16 |               -0.21 |         -0.55 |     83 |
|   3 |              1.64 |                1.37 |          1.61 |        1.54 |              0.22 |               -0.29 |         -0.66 |     83 |
|   4 |              1.7  |                1.46 |          1.74 |        1.63 |              0.25 |               -0.42 |         -0.78 |     83 |
|   5 |              1.77 |                1.52 |          1.83 |        1.71 |              0.25 |               -0.52 |         -0.91 |     83 |
|   6 |              1.84 |                1.58 |          1.94 |        1.79 |              0.28 |               -0.54 |         -0.98 |     83 |
|   7 |              2.21 |                1.46 |          1.85 |        1.84 |              0.83 |                0.41 |         -0.06 |     83 |

## By weekday of the target day

| weekday   |   MAPE solar_peak |   MAPE evening_peak |   MAPE energy |   MAPE mean |   bias solar_peak |   bias evening_peak |   bias energy |   days |
|:----------|------------------:|--------------------:|--------------:|------------:|------------------:|--------------------:|--------------:|-------:|
| Mon       |              1.44 |                1.25 |          1.16 |        1.29 |              0.37 |               -0.55 |         -0.71 |     13 |
| Tue       |              1.59 |                1.68 |          1.9  |        1.73 |             -0.08 |               -0.28 |         -1.04 |     12 |
| Wed       |              1.69 |                1.54 |          1.85 |        1.69 |             -0.28 |               -0.27 |         -0.72 |     12 |
| Thu       |              1.61 |                1.39 |          1.66 |        1.56 |              0.31 |               -0.34 |         -0.54 |     13 |
| Fri       |              2.35 |                1.46 |          2.08 |        1.97 |              0.04 |               -0.24 |         -0.74 |     13 |
| Sat       |              1.77 |                1.37 |          1.89 |        1.68 |              0.88 |                0.13 |         -0.1  |     13 |
| Sun       |              1.68 |                1.1  |          1.21 |        1.33 |              0.86 |               -0.15 |         -0.49 |     13 |

## By month (bias = systematic over/under-forecast)

| month    |   MAPE solar_peak |   MAPE evening_peak |   MAPE energy |   MAPE mean |   bias solar_peak |   bias evening_peak |   bias energy |   days |
|:---------|------------------:|--------------------:|--------------:|------------:|------------------:|--------------------:|--------------:|-------:|
| Feb 2025 |              0.99 |                0.66 |          0.83 |        0.83 |              0.58 |               -0.18 |         -0.51 |     28 |
| Jan 2025 |              2.04 |                1.87 |          2.47 |        2.12 |             -0.78 |               -0.63 |         -1.41 |     30 |
| Mar 2025 |              2.18 |                1.69 |          1.78 |        1.88 |              1.06 |                0.07 |          0.03 |     31 |

## By weather regime (observed weather; evaluation only)

| regime   |   MAPE solar_peak |   MAPE evening_peak |   MAPE energy |   MAPE mean |   bias solar_peak |   bias evening_peak |   bias energy |   days |
|:---------|------------------:|--------------------:|--------------:|------------:|------------------:|--------------------:|--------------:|-------:|
| cold/fog |              1.91 |                1.9  |          2.09 |        1.97 |              0.18 |                0.15 |         -0.31 |     19 |
| rainy    |              0.94 |                0.97 |          0.87 |        0.92 |              0.01 |                0.61 |          0.12 |      1 |
| cloudy   |              0.68 |                0.53 |          0.34 |        0.52 |              0.56 |               -0.28 |          0.15 |      1 |
| normal   |              1.72 |                1.3  |          1.61 |        1.54 |              0.33 |               -0.35 |         -0.72 |     68 |

## Holidays and festivals

| calendar                    |   MAPE solar_peak |   MAPE evening_peak |   MAPE energy |   MAPE mean |   bias solar_peak |   bias evening_peak |   bias energy |   days |
|:----------------------------|------------------:|--------------------:|--------------:|------------:|------------------:|--------------------:|--------------:|-------:|
| ordinary day                |              1.5  |                1.12 |          1.48 |        1.37 |             -0.13 |               -0.74 |         -1.15 |     71 |
| within 3 days of a festival |              2.37 |                1.93 |          2.21 |        2.17 |              1.69 |                1.51 |          1.45 |     13 |
| holiday/festival day        |              3.87 |                4.55 |          3.45 |        3.96 |              3.52 |                3.12 |          2.44 |      5 |

Share of the total error:

| calendar                    |   share of error % |   share of rows % |
|:----------------------------|-------------------:|------------------:|
| holiday/festival day        |               12.3 |               5   |
| ordinary day                |               69.3 |              81.4 |
| within 3 days of a festival |               18.4 |              13.6 |

## Bias vs variance

For each target and month: mean signed error (bias), its spread, and how much of the squared error is systematic (bias²) rather than day-to-day noise.

|                              |   bias % |   sd % |   rows |   bias² share of MSE % |
|:-----------------------------|---------:|-------:|-------:|-----------------------:|
| ('energy', 'Feb 2025')       |    -0.51 |   0.93 |    196 |                  22.88 |
| ('energy', 'Jan 2025')       |    -1.41 |   2.49 |    189 |                  24.25 |
| ('energy', 'Mar 2025')       |     0.03 |   2.62 |    196 |                   0.01 |
| ('evening_peak', 'Feb 2025') |    -0.18 |   0.88 |    196 |                   4.2  |
| ('evening_peak', 'Jan 2025') |    -0.63 |   2.28 |    189 |                   7.18 |
| ('evening_peak', 'Mar 2025') |     0.07 |   2.3  |    196 |                   0.1  |
| ('solar_peak', 'Feb 2025')   |     0.58 |   1.19 |    196 |                  19.12 |
| ('solar_peak', 'Jan 2025')   |    -0.78 |   2.37 |    189 |                   9.73 |
| ('solar_peak', 'Mar 2025')   |     1.06 |   3.03 |    196 |                  10.96 |

## The 20 worst validation days (ensemble, target-day error averaged over the 1–7-day forecasts)

| date            | target       |   actual |   forecast |   error % (mean of 1–7 d) | calendar                    | regime   |   tmax °C |   tmax vs normal |   N-India tmin °C |   rain mm |
|:----------------|:-------------|---------:|-----------:|--------------------------:|:----------------------------|:---------|----------:|-----------------:|------------------:|----------:|
| Fri 14 Mar 2025 | solar_peak   |    199.1 |      220.2 |                     10.58 | holiday/festival day        | normal   |     35.72 |             3.05 |             20.07 |      0.09 |
| Sat 15 Mar 2025 | solar_peak   |    210   |      227.4 |                      8.29 | within 3 days of a festival | normal   |     35.62 |             2.82 |             20.27 |      0.08 |
| Sat 15 Mar 2025 | energy       |   4489.3 |     4838.1 |                      7.77 | within 3 days of a festival | normal   |     35.62 |             2.82 |             20.27 |      0.08 |
| Fri 14 Mar 2025 | energy       |   4398.3 |     4710.6 |                      7.1  | holiday/festival day        | normal   |     35.72 |             3.05 |             20.07 |      0.09 |
| Tue 14 Jan 2025 | evening_peak |    185.6 |      198.6 |                      7    | holiday/festival day        | cold/fog |     25.63 |            -0.37 |              9.43 |      0.04 |
| Fri 14 Mar 2025 | evening_peak |    186.7 |      198.7 |                      6.44 | holiday/festival day        | normal   |     35.72 |             3.05 |             20.07 |      0.09 |
| Wed 15 Jan 2025 | energy       |   4219   |     4424.4 |                      4.87 | within 3 days of a festival | cold/fog |     24.86 |            -1.19 |              9.17 |      0.65 |
| Tue 07 Jan 2025 | solar_peak   |    228.1 |      217.4 |                     -4.66 | ordinary day                | cold/fog |     25.13 |            -0.58 |              9.8  |      0.33 |
| Tue 14 Jan 2025 | energy       |   4179   |     4374   |                      4.66 | holiday/festival day        | cold/fog |     25.63 |            -0.37 |              9.43 |      0.04 |
| Sat 15 Mar 2025 | evening_peak |    199   |      207.4 |                      4.24 | within 3 days of a festival | normal   |     35.62 |             2.82 |             20.27 |      0.08 |
| Wed 15 Jan 2025 | evening_peak |    193.7 |      201.8 |                      4.16 | within 3 days of a festival | cold/fog |     24.86 |            -1.19 |              9.17 |      0.65 |
| Tue 07 Jan 2025 | energy       |   4496.6 |     4309.9 |                     -4.15 | ordinary day                | cold/fog |     25.13 |            -0.58 |              9.8  |      0.33 |
| Tue 28 Jan 2025 | energy       |   4665.1 |     4472.8 |                     -4.12 | ordinary day                | normal   |     27.75 |             0.69 |              9.2  |      0    |
| Wed 15 Jan 2025 | solar_peak   |    216.8 |      225.7 |                      4.1  | within 3 days of a festival | cold/fog |     24.86 |            -1.19 |              9.17 |      0.65 |
| Tue 14 Jan 2025 | solar_peak   |    215.6 |      224.3 |                      4.04 | holiday/festival day        | cold/fog |     25.63 |            -0.37 |              9.43 |      0.04 |
| Thu 13 Mar 2025 | evening_peak |    202.9 |      210.8 |                      3.9  | within 3 days of a festival | normal   |     35.33 |             2.79 |             21.5  |      0.75 |
| Wed 08 Jan 2025 | solar_peak   |    227.2 |      218.4 |                     -3.87 | ordinary day                | normal   |     24.33 |            -1.41 |              8.57 |      0.04 |
| Wed 29 Jan 2025 | energy       |   4678.5 |     4501.8 |                     -3.77 | ordinary day                | normal   |     28.21 |             1.05 |             10.8  |      0.02 |
| Wed 29 Jan 2025 | solar_peak   |    237.3 |      228.8 |                     -3.59 | ordinary day                | normal   |     28.21 |             1.05 |             10.8  |      0.02 |
| Wed 08 Jan 2025 | energy       |   4491.5 |     4333.1 |                     -3.53 | ordinary day                | normal   |     24.33 |            -1.41 |              8.57 |      0.04 |

## Second view: inner fold Oct–Dec 2024 (training period; festival season)

Same recipe, trained on origins up to Sep 2024, normals up to Sep 2024.

By month:

| month    |   MAPE solar_peak |   MAPE evening_peak |   MAPE energy |   MAPE mean |   bias solar_peak |   bias evening_peak |   bias energy |   days |
|:---------|------------------:|--------------------:|--------------:|------------:|------------------:|--------------------:|--------------:|-------:|
| Dec 2024 |              1.68 |                1.36 |          1.5  |        1.51 |             -0.43 |               -0    |          0.01 |     31 |
| Nov 2024 |              3.42 |                2.4  |          1.58 |        2.47 |             -2.1  |                0.42 |         -0.36 |     30 |
| Oct 2024 |              3.42 |                2.09 |          2.86 |        2.79 |              2.15 |                1.7  |          2.23 |     30 |

Holidays and festivals:

| calendar                    |   MAPE solar_peak |   MAPE evening_peak |   MAPE energy |   MAPE mean |   bias solar_peak |   bias evening_peak |   bias energy |   days |
|:----------------------------|------------------:|--------------------:|--------------:|------------:|------------------:|--------------------:|--------------:|-------:|
| ordinary day                |              2.66 |                1.42 |          1.74 |        1.94 |             -0.72 |               -0.02 |          0.32 |     69 |
| within 3 days of a festival |              3.19 |                2.68 |          2.63 |        2.83 |              1.43 |                2.24 |          1.38 |     14 |
| holiday/festival day        |              3.87 |                5.48 |          2.72 |        4.03 |              1.4  |                4.2  |          1.42 |      8 |

Share of the total error:

| calendar                    |   share of error % |   share of rows % |
|:----------------------------|-------------------:|------------------:|
| holiday/festival day        |               15   |               8.4 |
| ordinary day                |               65.2 |              75.8 |
| within 3 days of a festival |               19.8 |              15.8 |

By horizon:

|   h |   MAPE solar_peak |   MAPE evening_peak |   MAPE energy |   MAPE mean |   bias solar_peak |   bias evening_peak |   bias energy |   days |
|----:|------------------:|--------------------:|--------------:|------------:|------------------:|--------------------:|--------------:|-------:|
|   1 |              2.05 |                1.54 |          1.47 |        1.69 |             -0.21 |                0.42 |          0.26 |     85 |
|   2 |              2.28 |                1.54 |          1.57 |        1.8  |             -0.23 |                0.4  |          0.31 |     85 |
|   3 |              2.72 |                1.77 |          1.85 |        2.11 |             -0.24 |                0.51 |          0.44 |     85 |
|   4 |              2.97 |                1.93 |          2.12 |        2.34 |             -0.26 |                0.59 |          0.57 |     85 |
|   5 |              3.15 |                2.1  |          2.15 |        2.47 |             -0.35 |                0.67 |          0.58 |     85 |
|   6 |              3.31 |                2.27 |          2.19 |        2.59 |             -0.49 |                0.7  |          0.48 |     85 |
|   7 |              3.45 |                2.57 |          2.38 |        2.8  |              0.38 |                1.54 |          1.44 |     85 |

Worst days:

| date            | target       |   actual |   forecast |   error % (mean of 1–7 d) | calendar                    | regime   |   tmax °C |   tmax vs normal |   N-India tmin °C |   rain mm |
|:----------------|:-------------|---------:|-----------:|--------------------------:|:----------------------------|:---------|----------:|-----------------:|------------------:|----------:|
| Sat 12 Oct 2024 | evening_peak |    182.9 |      203.6 |                     11.3  | holiday/festival day        | rainy    |     30.81 |            -1.27 |             21.53 |      3.65 |
| Sat 28 Dec 2024 | solar_peak   |    197.7 |      218.9 |                     10.71 | ordinary day                | cold/fog |     25.57 |            -0.06 |             13.5  |      0.85 |
| Thu 31 Oct 2024 | evening_peak |    175.5 |      194   |                     10.52 | holiday/festival day        | normal   |     32    |             1.09 |             21.57 |      1.76 |
| Fri 01 Nov 2024 | evening_peak |    170.3 |      188.2 |                     10.51 | holiday/festival day        | normal   |     31.77 |             0.96 |             19.67 |      1.81 |
| Sat 02 Nov 2024 | evening_peak |    173.4 |      191.4 |                     10.42 | within 3 days of a festival | normal   |     31.51 |             0.79 |             18.5  |      0.63 |
| Sat 12 Oct 2024 | solar_peak   |    187   |      202.7 |                      8.39 | holiday/festival day        | rainy    |     30.81 |            -1.27 |             21.53 |      3.65 |
| Sat 28 Dec 2024 | energy       |   4029.6 |     4350.4 |                      7.96 | ordinary day                | cold/fog |     25.57 |            -0.06 |             13.5  |      0.85 |
| Sat 02 Nov 2024 | solar_peak   |    170.7 |      184.1 |                      7.84 | within 3 days of a festival | normal   |     31.51 |             0.79 |             18.5  |      0.63 |
| Sun 03 Nov 2024 | evening_peak |    172   |      185.2 |                      7.68 | within 3 days of a festival | normal   |     31.44 |             0.82 |             18.73 |      0.49 |
| Sun 13 Oct 2024 | energy       |   4178.6 |     4495.5 |                      7.58 | within 3 days of a festival | rainy    |     30.59 |            -1.45 |             21.27 |      4.31 |

