# Fairness audit: our forecast vs Grid-India's week-ahead forecast

Final model: `ensemble`. Reported window: Grid-India forecast days from 2025-11-01 (1098 day × target rows with actuals). Peaks in GW, energy in MU. Improvement = 1 − our MAE / Grid-India MAE; win rate = share of days where we are closer; bias = mean(forecast − actual).

## 1. When was each Grid-India forecast made?

The Grid-India listing API records, for every week-ahead PDF, the date the forecast was prepared (`Field2`) and the upload time (`CreatedOn`); only 35 of 125 PDFs have a readable footer date, and none in the reported window, so the listing is the source used. Days between 'prepared' and the first day of the forecast week, per weekly issue:

| days before week start | weeks |
|---|---|
| 1 | 1 |
| 3 | 2 |
| 4 | 4 |
| 5 | 45 |
| 6 | 1 |

So for most weeks the forecast for the first day is 6 days ahead and for the last day 12 days ahead (counting from the last day whose demand report was published when it was prepared). The original comparison averaged our 1–7-day-ahead forecasts, which use up to a week more recent information.

**Matched basis.** For each Grid-India week we take our forecast issued at origin = prepared date − 1 (the newest published demand report on that day), and compare each day at the same lead. Our neural models only forecast 7 days, so the matched comparison uses a long-lead LightGBM (leads 1–14), backtested month by month exactly like the main models. Leads 1–7 use the archived weather forecast issued at the origin; leads 8–14 use climatology (normals from data before the backtest), because no weather forecast that far ahead is archived. Lead distribution of the matched rows: 2 d: 3, 3 d: 3, 4 d: 9, 5 d: 21, 6 d: 150, 7 d: 153, 8 d: 153, 9 d: 150, 10 d: 150, 11 d: 144, 12 d: 132, 13 d: 3.

Grid-India rows without a matching origin in our backtest: 27.

Long-lead LightGBM MAPE % by lead (sanity check: leads 1–7 should be close to the main LightGBM):

| target       |    1 |    2 |    3 |    4 |    5 |    6 |    7 |    8 |    9 |   10 |   11 |   12 |   13 |   14 |
|:-------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| energy       | 2.01 | 2.17 | 2.34 | 2.47 | 2.54 | 2.68 | 2.79 | 3.32 | 3.37 | 3.44 | 3.44 | 3.46 | 3.45 | 3.47 |
| evening_peak | 1.69 | 1.83 | 1.93 | 2.05 | 2.1  | 2.18 | 2.27 | 2.57 | 2.62 | 2.65 | 2.65 | 2.62 | 2.58 | 2.6  |
| solar_peak   | 2.36 | 2.58 | 2.81 | 2.95 | 3.15 | 3.43 | 3.48 | 3.93 | 4.09 | 4.19 | 4.33 | 4.4  | 4.44 | 4.47 |

## 2. The gap after each correction

### solar_peak (GW)

| step | days | our MAE | GI MAE | our MAPE % | GI MAPE % | improvement | win rate | our bias | GI bias |
|---|---|---|---|---|---|---|---|---|---|
| 0. Original basis: mean of our 1–7-day-ahead forecasts for each day | 365 | 5.06 | 10.78 | 2.19 | 4.71 | +53.0% | 76% | -0.15 | +2.91 |
| 1. Like with like: one P50 per day (our 7-day-ahead forecast) vs Grid-India's value | 359 | 6.60 | 10.82 | 2.82 | 4.72 | +39.1% | 69% | -1.08 | +2.83 |
| 2. Matched issue time: our forecast made with the data Grid-India had (same issue day, same lead per day; long-lead LightGBM) | 357 | 8.60 | 10.85 | 3.68 | 4.72 | +20.7% | 60% | -1.92 | +2.80 |
| 2b. Matched issue time, final model (ensemble) — only the days within its 7-day horizon (the first days of each Grid-India week) | 113 | 5.92 | 9.37 | 2.50 | 4.00 | +36.8% | 64% | -0.88 | +2.24 |
| 3. Matched, Grid-India value read from text labels | 254 | 8.79 | 11.59 | 3.71 | 5.01 | +24.1% | 61% | -0.67 | +3.46 |
| 3. Matched, Grid-India value read from digitised curve | 103 | 8.13 | 9.02 | 3.62 | 4.01 | +9.8% | 58% | -5.01 | +1.18 |
| 4. Matched + Grid-India's constant bias removed (bias estimated on past weeks only) | 343 | 8.45 | 11.47 | 3.58 | 4.96 | +26.3% | 66% | -1.50 | +3.41 |

### evening_peak (GW)

| step | days | our MAE | GI MAE | our MAPE % | GI MAPE % | improvement | win rate | our bias | GI bias |
|---|---|---|---|---|---|---|---|---|---|
| 0. Original basis: mean of our 1–7-day-ahead forecasts for each day | 365 | 3.76 | 6.78 | 1.70 | 3.06 | +44.6% | 72% | -0.30 | +4.09 |
| 1. Like with like: one P50 per day (our 7-day-ahead forecast) vs Grid-India's value | 359 | 4.63 | 6.82 | 2.07 | 3.07 | +32.2% | 64% | -0.86 | +4.11 |
| 2. Matched issue time: our forecast made with the data Grid-India had (same issue day, same lead per day; long-lead LightGBM) | 357 | 5.10 | 6.83 | 2.29 | 3.07 | +25.3% | 65% | -0.60 | +4.10 |
| 2b. Matched issue time, final model (ensemble) — only the days within its 7-day horizon (the first days of each Grid-India week) | 113 | 4.12 | 6.50 | 1.83 | 2.89 | +36.5% | 66% | -0.71 | +3.88 |
| 3. Matched, Grid-India value read from text labels | 282 | 5.22 | 6.39 | 2.34 | 2.89 | +18.3% | 62% | -0.60 | +3.81 |
| 3. Matched, Grid-India value read from digitised curve | 75 | 4.65 | 8.48 | 2.07 | 3.76 | +45.2% | 76% | -0.57 | +5.20 |
| 4. Matched + Grid-India's constant bias removed (bias estimated on past weeks only) | 343 | 5.22 | 6.33 | 2.33 | 2.86 | +17.5% | 61% | -0.58 | +2.20 |

### energy (MU)

| step | days | our MAE | GI MAE | our MAPE % | GI MAPE % | improvement | win rate | our bias | GI bias |
|---|---|---|---|---|---|---|---|---|---|
| 0. Original basis: mean of our 1–7-day-ahead forecasts for each day | 365 | 93.47 | 191.11 | 1.86 | 3.84 | +51.1% | 78% | -13.01 | +105.32 |
| 1. Like with like: one P50 per day (our 7-day-ahead forecast) vs Grid-India's value | 359 | 117.39 | 189.90 | 2.30 | 3.80 | +38.2% | 70% | -29.53 | +102.68 |
| 2. Matched issue time: our forecast made with the data Grid-India had (same issue day, same lead per day; long-lead LightGBM) | 357 | 153.08 | 189.27 | 3.01 | 3.77 | +19.1% | 57% | -34.62 | +101.56 |
| 2b. Matched issue time, final model (ensemble) — only the days within its 7-day horizon (the first days of each Grid-India week) | 113 | 105.89 | 171.05 | 2.07 | 3.37 | +38.1% | 69% | -20.71 | +75.84 |
| 3. Matched, Grid-India value read from digitised curve | 357 | 153.08 | 189.27 | 3.01 | 3.77 | +19.1% | 57% | -34.62 | +101.56 |
| 4. Matched + Grid-India's constant bias removed (bias estimated on past weeks only) | 343 | 156.29 | 188.04 | 3.06 | 3.76 | +16.9% | 58% | -33.00 | +38.17 |

## 3. Text labels vs digitised curve

Grid-India values in the reported window come from the text labels printed on the chart (ranges such as '234-238'; we use the midpoint) or, for weeks whose labels are not text, from our digitisation of the plotted curve (accurate to ~0.1 GW). Energy always comes from the integrated curve.

| target       |   curve |   label |
|:-------------|--------:|--------:|
| energy       |     357 |       0 |
| evening_peak |      75 |     282 |
| solar_peak   |     103 |     254 |

- solar_peak: improvement +24.1% on label weeks (254 days) vs +9.8% on curve weeks (103 days); Grid-India MAPE 5.01 % vs 4.01 %.
- evening_peak: improvement +18.3% on label weeks (282 days) vs +45.2% on curve weeks (75 days); Grid-India MAPE 2.89 % vs 3.76 %.

## 4. Systematic bias

Grid-India over-forecasts on average (positive bias). Removing a constant bias per target, estimated only from earlier days whose actuals were published before each week's forecast was prepared (expanding mean, ≥ 14 days of history; 1029 rows), shows how much of the gap is a deliberate safety margin rather than tracking error. Step 4 in the tables above.

## 5. Leakage check of our weather inputs

Backtest forecast rows in the reported window: 2,604 (origin × horizon).

| weather source | rows |
|---|---|
| archived forecast issued on the origin day (lead = horizon) | 2,583 |
| older archived forecast (issued before the origin) | 0 |
| fallback: persistence of observed weather up to the origin | 21 |

Checked 2,583 rows against the archive: issued on the origin day 2,583, issued after the origin 0, values identical to the observed weather of the target day 0 (mean |forecast − observed| of all-India max temperature 0.56 °C, i.e. real forecast error is present).

21 rows fell back to persistence (observed weather up to the origin only — not hindsight).
The climatology used at leads 8–14 of the long-lead model is built only from observed weather before the backtest (NORMALS_END), never from the forecast days.

### Rerun without the fallback rows

The 21 fallback rows are the last forecasts of the window, whose target days lie beyond the end of the weather-forecast archive. None of their target days has published actuals yet, so they never entered any score — the rerun is identical. Removing them (and long-lead rows with a fallback):

| target | original basis: with → without | matched, bias-corrected: with → without |
|---|---|---|
| solar_peak | +53.0% → +53.0% | +26.3% → +26.3% |
| evening_peak | +44.6% → +44.6% | +17.5% → +17.5% |
| energy | +51.1% → +51.1% | +16.9% → +16.9% |

## 6. Conclusion

| target | original gap | matched issue time | + Grid-India bias removed |
|---|---|---|---|
| solar_peak | +53.0% | +20.7% | +26.3% |
| evening_peak | +44.6% | +25.3% | +17.5% |
| energy | +51.1% | +19.1% | +16.9% |

**In plain language.** The original headline (our error 45%–53% lower than Grid-India's) compared forecasts made with very different information: ours averaged 1–7-day-ahead forecasts, while Grid-India prepares its week-ahead forecast about five days before the week starts, so it forecasts 6–12 days ahead. When our model is given exactly the information Grid-India had on the day it prepared each forecast, the advantage shrinks to 19%–25% — still clearly better on all three targets and on most days (win rates 60%, 65%, 57%). Grid-India over-forecasts systematically (on average +2.8 GW solar-hour, +4.1 GW evening, +102 MU energy), which looks like a deliberate safety margin for grid planning. Removing a constant bias learned from earlier weeks narrows the gap for the evening peak and energy (to +18% and +17%) but not for the solar-hour peak (+26%): Grid-India's solar-hour bias changed over time, so last year's average does not transfer. The gap also depends on how the Grid-India value was read: for the solar-hour peak it is smaller on weeks taken from the digitised curve, for the evening peak larger. The weather inputs contain no hindsight. Fair summary: on a matched basis our forecast is roughly 17%–25% more accurate than Grid-India's week-ahead forecast, not 40–50 %; part of the original gap was the information advantage of forecasting closer to the day, and part is Grid-India's cautious upward bias.

