# Accuracy round: experiments

All numbers are on the **validation months (Jan–Mar 2025)** of the fixed split; the test period is not used here. Score: MAPE of the ensemble's P50, mean of the three targets, averaged over 3 seeds; ensemble weights and P10–P90 calibration fitted out of fold (January ↔ February–March). Coverage: share of actuals inside the calibrated P10–P90 (target 80 %).

Keep rule: lower validation MAPE and coverage at most 2 points lower. A gain under 0.01 points counts as no change. Seed-to-seed spread of the score is about 0.04 points, so gains smaller than that are within noise; each row shows the paired gain per seed (same seeds, before minus after).

| # | experiment | variants tried (validation MAPE %) | before | best after | gain | paired gain per seed | coverage before → after | kept |
|---|---|---|---|---|---|---|---|---|
| a0 | Lead-7 weather debiasing (extra, from the error analysis) | lead-7 weather debiased: 1.646 | 1.646 | 1.646 | +0.001 | +0.013 / +0.015 / -0.027 | 80.2 → 80.2 | no |
| a1 | Features: interactions (temperature × weekend, CDD × season) | +interactions: 1.627 | 1.646 | 1.627 | +0.019 | -0.006 / +0.032 / +0.031 | 80.2 → 79.8 | yes |
| a2 | Features: demand growth (YoY 7- and 30-day, 30/90-day momentum) | +growth: 1.638 | 1.627 | 1.638 | -0.011 | -0.004 / -0.019 / -0.011 | 79.8 → 79.1 | no |
| a3 | Features: same day last year, festival-aligned (level and shape) | +lag365 festival-aligned: 1.612 | 1.627 | 1.612 | +0.016 | +0.037 / +0.005 / +0.005 | 79.8 → 79.5 | yes |
| b | Target transform (LightGBM): day-over-day and week-over-week ratio vs the 7-day-mean ratio | target / origin day: 1.621; target / same weekday last week: 1.651 | 1.612 | 1.621 | -0.009 | -0.016 / -0.014 / +0.003 | 79.5 → 80.2 | no |
| c1 | Recency weighting, LightGBM (half-life tuned) | lgbm half-life 90 d: 1.634; lgbm half-life 180 d: 1.590; lgbm half-life 365 d: 1.586; lgbm half-life 730 d: 1.598 | 1.612 | 1.586 | +0.026 | +0.003 / +0.041 / +0.032 | 79.5 → 79.6 | yes |
| c2 | Recency weighting, LSTM (half-life tuned) | lstm half-life 90 d: 1.571; lstm half-life 180 d: 1.609; lstm half-life 365 d: 1.567; lstm half-life 730 d: 1.580 | 1.586 | 1.567 | +0.019 | -0.019 / +0.084 / -0.007 | 79.6 → 80.3 | yes |
| d1 | LSTM window length 45 / 90 days (current 60) | LSTM window 45 d: 1.602; LSTM window 90 d: 1.593 | 1.567 | 1.593 | -0.026 | -0.018 / -0.054 / -0.006 | 80.3 → 80.3 | no |
| d2 | LSTM hidden size 64 / 128 (current 96) | LSTM hidden 64: 1.578; LSTM hidden 128: 1.561 | 1.567 | 1.561 | +0.005 | +0.021 / -0.012 / +0.007 | 80.3 → 80.3 | no |
| d3 | LSTM dropout 0.2 / 0.5 (current 0.34) | LSTM dropout 0.2: 1.576; LSTM dropout 0.5: 1.605 | 1.567 | 1.576 | -0.009 | -0.017 / -0.013 / +0.004 | 80.3 → 79.9 | no |
| e | Ensemble: add TFT, weights per target × horizon on a 3-member simplex | LSTM + LightGBM + TFT: 1.632 | 1.567 | 1.632 | -0.066 | +0.036 / -0.140 / -0.093 | 80.3 → 78.7 | no |
| f | Residual correction: small LightGBM on the ensemble's validation residuals | +residual correction: 1.749 | 1.567 | 1.749 | -0.182 | -0.181 / -0.333 / -0.032 | 80.3 → 65.6 | no |

## Details

### a0. Lead-7 weather debiasing (extra, from the error analysis)

Why: The archived 7-day-ahead weather is far warmer/drier than leads 1–6; both models over-forecast at h = 7.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline) | 1.646 | 0.037 | 1.766 | 1.457 | 1.716 | 80.2 | 1.749 | 1.734 |
| lead-7 weather debiased | 1.646 | 0.049 | 1.790 | 1.441 | 1.707 | 80.2 | 1.719 | 1.733 |

Decision: not kept (lead-7 weather debiased, gain +0.001, coverage change -0.1 points).

### a1. Features: interactions (temperature × weekend, CDD × season)

Why: Error source 3: the temperature response differs by season and weekday.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline) | 1.646 | 0.037 | 1.766 | 1.457 | 1.716 | 80.2 | 1.749 | 1.734 |
| +interactions | 1.627 | 0.017 | 1.742 | 1.442 | 1.698 | 79.8 | 1.749 | 1.711 |

Decision: kept (+interactions, gain +0.019, coverage change -0.4 points).

### a2. Features: demand growth (YoY 7- and 30-day, 30/90-day momentum)

Why: Error source 2: level drift at seasonal turning points.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline + a1) | 1.627 | 0.017 | 1.742 | 1.442 | 1.698 | 79.8 | 1.749 | 1.711 |
| +growth | 1.638 | 0.024 | 1.794 | 1.432 | 1.688 | 79.1 | 1.749 | 1.718 |

Decision: not kept (+growth, gain -0.011, coverage change -0.7 points).

### a3. Features: same day last year, festival-aligned (level and shape)

Why: Error source 1: festivals occur once or twice in training; last year's festival sizes this one.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline + a1) | 1.627 | 0.017 | 1.742 | 1.442 | 1.698 | 79.8 | 1.749 | 1.711 |
| +lag365 festival-aligned | 1.612 | 0.034 | 1.723 | 1.424 | 1.688 | 79.5 | 1.749 | 1.672 |

Decision: kept (+lag365 festival-aligned, gain +0.016, coverage change -0.2 points).

### b. Target transform (LightGBM): day-over-day and week-over-week ratio vs the 7-day-mean ratio

Why: The baseline already predicts a ratio (to the last 7 days); test the other natural bases for drift.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline + a1 + a3) | 1.612 | 0.034 | 1.723 | 1.424 | 1.688 | 79.5 | 1.749 | 1.672 |
| target / origin day | 1.621 | 0.032 | 1.762 | 1.386 | 1.715 | 80.2 | 1.749 | 1.690 |
| target / same weekday last week | 1.651 | 0.025 | 1.788 | 1.469 | 1.695 | 79.3 | 1.749 | 1.818 |

Decision: not kept (target / origin day, gain -0.009, coverage change +0.6 points).

### c1. Recency weighting, LightGBM (half-life tuned)

Why: Track the latest demand level by weighting recent training months higher.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline + a1 + a3) | 1.612 | 0.034 | 1.723 | 1.424 | 1.688 | 79.5 | 1.749 | 1.672 |
| lgbm half-life 90 d | 1.634 | 0.012 | 1.832 | 1.390 | 1.678 | 80.5 | 1.749 | 1.655 |
| lgbm half-life 180 d | 1.590 | 0.029 | 1.727 | 1.387 | 1.658 | 80.5 | 1.749 | 1.614 |
| lgbm half-life 365 d | 1.586 | 0.014 | 1.700 | 1.388 | 1.670 | 79.6 | 1.749 | 1.634 |
| lgbm half-life 730 d | 1.598 | 0.026 | 1.692 | 1.426 | 1.677 | 80.0 | 1.749 | 1.651 |

Decision: kept (lgbm half-life 365 d, gain +0.026, coverage change +0.1 points).

### c2. Recency weighting, LSTM (half-life tuned)

Why: Same idea for the LSTM's training samples.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline + a1 + a3 + c1) | 1.586 | 0.014 | 1.700 | 1.388 | 1.670 | 79.6 | 1.749 | 1.634 |
| lstm half-life 90 d | 1.571 | 0.061 | 1.695 | 1.396 | 1.622 | 80.6 | 1.780 | 1.634 |
| lstm half-life 180 d | 1.609 | 0.014 | 1.774 | 1.393 | 1.661 | 80.5 | 1.825 | 1.634 |
| lstm half-life 365 d | 1.567 | 0.046 | 1.706 | 1.355 | 1.640 | 80.3 | 1.748 | 1.634 |
| lstm half-life 730 d | 1.580 | 0.057 | 1.734 | 1.360 | 1.646 | 80.0 | 1.796 | 1.634 |

Decision: kept (lstm half-life 365 d, gain +0.019, coverage change +0.7 points).

### d1. LSTM window length 45 / 90 days (current 60)

Why: Capacity/regularisation sweep around the Optuna best: look-back first.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline + a1 + a3 + c1 + c2) | 1.567 | 0.046 | 1.706 | 1.355 | 1.640 | 80.3 | 1.748 | 1.634 |
| LSTM window 45 d | 1.602 | 0.011 | 1.746 | 1.408 | 1.652 | 80.7 | 1.798 | 1.634 |
| LSTM window 90 d | 1.593 | 0.022 | 1.759 | 1.377 | 1.642 | 80.3 | 1.826 | 1.634 |

Decision: not kept (LSTM window 90 d, gain -0.026, coverage change -0.0 points).

### d2. LSTM hidden size 64 / 128 (current 96)

Why: Capacity.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline + a1 + a3 + c1 + c2) | 1.567 | 0.046 | 1.706 | 1.355 | 1.640 | 80.3 | 1.748 | 1.634 |
| LSTM hidden 64 | 1.578 | 0.036 | 1.685 | 1.388 | 1.661 | 79.6 | 1.754 | 1.634 |
| LSTM hidden 128 | 1.561 | 0.033 | 1.677 | 1.376 | 1.632 | 80.3 | 1.739 | 1.634 |

Decision: not kept (LSTM hidden 128, gain +0.005, coverage change -0.1 points).

### d3. LSTM dropout 0.2 / 0.5 (current 0.34)

Why: Regularisation.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline + a1 + a3 + c1 + c2) | 1.567 | 0.046 | 1.706 | 1.355 | 1.640 | 80.3 | 1.748 | 1.634 |
| LSTM dropout 0.2 | 1.576 | 0.042 | 1.723 | 1.372 | 1.632 | 79.9 | 1.799 | 1.634 |
| LSTM dropout 0.5 | 1.605 | 0.030 | 1.765 | 1.396 | 1.654 | 80.5 | 1.827 | 1.634 |

Decision: not kept (LSTM dropout 0.2, gain -0.009, coverage change -0.4 points).

### e. Ensemble: add TFT, weights per target × horizon on a 3-member simplex

Why: The baseline blends LSTM + LightGBM per target × horizon; TFT is trained but unused.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline + a1 + a3 + c1 + c2) | 1.567 | 0.046 | 1.706 | 1.355 | 1.640 | 80.3 | 1.748 | 1.634 |
| LSTM + LightGBM + TFT | 1.632 | 0.071 | 1.723 | 1.463 | 1.712 | 78.7 | 1.748 | 1.634 |

Decision: not kept (LSTM + LightGBM + TFT, gain -0.066, coverage change -1.6 points).

### f. Residual correction: small LightGBM on the ensemble's validation residuals

Why: Horizon/weekday/weather patterns left in the ensemble's errors.

| recipe | ensemble MAPE % | sd over seeds | solar-hour | evening | energy | coverage % | LSTM MAPE % | LightGBM MAPE % |
|---|---|---|---|---|---|---|---|---|
| before (baseline + a1 + a3 + c1 + c2) | 1.567 | 0.046 | 1.706 | 1.355 | 1.640 | 80.3 | 1.748 | 1.634 |
| +residual correction | 1.749 | 0.110 | 1.995 | 1.369 | 1.882 | 65.6 | 1.748 | 1.634 |

Decision: not kept (+residual correction, gain -0.182, coverage change -14.7 points).

Incumbent after the last experiment: **baseline + a1 + a3 + c1 + c2**.

