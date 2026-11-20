# Accuracy round: final measurement


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


## Summary of all experiments

Validation = OOF ensemble MAPE % on Jan–Mar 2025, mean of 3 seeds (reports/experiments.md). Test numbers exist only for the baseline and the final recipe: the test period was evaluated twice, never per experiment.

| # | experiment | validation before | validation after (best variant) | kept | test |
|---|---|---|---|---|---|
| – | baseline (production) | 1.646 | – | – | 2.667 |
| a0 | Lead-7 weather debiasing (extra, from the error analysis) | 1.646 | 1.646 (lead-7 weather debiased) | no | – |
| a1 | Features: interactions (temperature × weekend, CDD × season) | 1.646 | 1.627 (+interactions) | yes | – |
| a2 | Features: demand growth (YoY 7- and 30-day, 30/90-day momentum) | 1.627 | 1.638 (+growth) | no | – |
| a3 | Features: same day last year, festival-aligned (level and shape) | 1.627 | 1.612 (+lag365 festival-aligned) | yes | – |
| b | Target transform (LightGBM): day-over-day and week-over-week ratio vs the 7-day-mean ratio | 1.612 | 1.621 (target / origin day) | no | – |
| c1 | Recency weighting, LightGBM (half-life tuned) | 1.612 | 1.586 (lgbm half-life 365 d) | yes | – |
| c2 | Recency weighting, LSTM (half-life tuned) | 1.586 | 1.567 (lstm half-life 365 d) | yes | – |
| d1 | LSTM window length 45 / 90 days (current 60) | 1.567 | 1.593 (LSTM window 90 d) | no | – |
| d2 | LSTM hidden size 64 / 128 (current 96) | 1.567 | 1.561 (LSTM hidden 128) | no | – |
| d3 | LSTM dropout 0.2 / 0.5 (current 0.34) | 1.567 | 1.576 (LSTM dropout 0.2) | no | – |
| e | Ensemble: add TFT, weights per target × horizon on a 3-member simplex | 1.567 | 1.632 (LSTM + LightGBM + TFT) | no | – |
| f | Residual correction: small LightGBM on the ensemble's validation residuals | 1.567 | 1.749 (+residual correction) | no | – |
| final | baseline + a1 + a3 + c1 + c2 | 1.646 | 1.567 | – | 2.573 (rolling, refit monthly: 2.141 vs 2.064) |

### What moved the needle

On validation, four changes each helped a little and add up to 0.080 points: recency weighting for LightGBM (+0.026) and the LSTM (+0.019, carried by one of three seeds), temperature × weekend and CDD × season interactions (+0.019), and the festival-aligned same-day-last-year features (+0.016, the only feature change that helped on every seed, in line with festivals being the largest error source). Each gain is about the size of the seed-to-seed spread (0.04), and choosing the half-lives on the same three winter months flatters them. Everything else did nothing or hurt: a different label base, growth features, a longer or shorter LSTM window, more or less LSTM capacity, TFT as a third member (−0.07) and the residual corrector (−0.18, overfits ~290 rows per fold). On the test period, with models trained once, the recipe gained 0.094 points, almost all of it through the LSTM (+0.213 alone vs +0.036 for LightGBM), mostly on energy and the evening peak. But retrained monthly, as production runs, the same recipe loses (2.064 → 2.141 %), LightGBM most. The honest reading: none of the planned changes is a robust improvement for the system as it is operated; the baseline stays. The next round should validate on rolling refits inside the training period rather than on one frozen three-month window.
