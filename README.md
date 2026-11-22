# India Power Demand Forecast

Deep-learning forecast of India's electricity demand, rebuilt every day from public
Grid-India (NLDC) reports, and benchmarked against Grid-India's own week-ahead forecast.

**Status: Phase 2 complete — probabilistic 7-day forecasts** (P10/P50/P90) of the solar-hour peak,
evening peak and daily energy from an ensemble of an LSTM encoder–decoder and LightGBM (plus a
compact Temporal Fusion Transformer), driven by all-weather inputs, with a rolling backtest and a
fairness audit against Grid-India, and a Streamlit app.
Results: [`reports/results.md`](reports/results.md) · data checks: [`reports/data_quality.md`](reports/data_quality.md).

## Results (12-month rolling backtest, forecasts issued 1 Nov 2025 – 8 Nov 2026)

| model | solar-hour peak MAPE | evening peak MAPE | energy MAPE | P10–P90 coverage |
|---|---|---|---|---|
| **Ensemble (LSTM + LightGBM)** | **2.40 %** | **1.79 %** | **2.00 %** | 80.1 % |
| LSTM (compact weather, rare-weather weights) | 2.39 % | 1.91 % | 2.06 % | 80.4 % |
| LightGBM (compact weather) | 2.78 % | 1.81 % | 2.26 % | 81.0 % |
| TFT (v1 weather) | 2.80 % | 1.96 % | 2.32 % | 79.2 % |
| Seasonal naive | 4.74 % | 3.15 % | 3.93 % | 81.9 % |
| Grid-India week-ahead forecast | 4.73 % | 3.08 % | 3.87 % | – |

* Against Grid-India on a **matched basis** (our forecast made with the information Grid-India had
  when it prepared its week-ahead forecast, same lead per day) our error is **19–25 % lower**, and
  17–26 % lower after removing Grid-India's systematic over-forecast — not the 45–53 % the naive
  comparison suggests ([`reports/fairness_audit.md`](reports/fairness_audit.md)).
* P10–P90 coverage is 76–83 % at every horizon after per-horizon adaptive conformal calibration
  (before: 69–90 %).
* The record-peak probability is labelled **Experimental**: its Brier skill stays below 0.05.

Full tables, by weather regime and season: [`reports/results.md`](reports/results.md).
Compute: NVIDIA RTX 3050 Ti Laptop (4 GB); a full 18-month backtest of six model variants takes about 17 min.

## Setup (incl. GPU)

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
# PyTorch with CUDA first (pick the wheel matching your driver; check with `nvidia-smi`).
# Any cu12x/cu13x wheel works with a CUDA 13.x driver; the large wheels may need a long timeout:
UV_HTTP_TIMEOUT=600 uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest                                   # offline tests (parsers, leakage, models, app pages)
```

The code picks the GPU automatically (`src/utils/device.py`) and falls back to CPU; training
uses fp16 autocast on CUDA, fixed seeds, and halves the batch size on CUDA out-of-memory.
Developed and trained on an NVIDIA RTX 3050 Ti Laptop (4 GB). On a laptop, plug in the charger before training.

## Phase 2 workflow

```bash
python -m src.data.quality --full        # coverage, gaps (<=2 d imputed + flagged), outliers -> reports/data_quality.md
                                         # (the daily pipeline writes the same summary to reports/latest/, git-ignored)
python -m src.data.weather               # Open-Meteo: observed + archived 1..7-day-ahead forecasts (cached)
python -m src.features.build             # features.parquet, weather_fc.parquet, weather_normals.json, weather_city.parquet
python scripts/train.py --model all --tune        # Optuna (30 trials/model, one at a time on the GPU) + final fit
python scripts/backtest.py               # 6 warm-up + 12 reported months, all models + ensemble -> reports/results.md
python scripts/audit.py                  # matched comparison with Grid-India -> reports/fairness_audit.md
python scripts/train.py --model all      # refit the final models with the variant the backtest chose
python scripts/predict.py                # live 7-day forecast -> data/processed/latest_forecast.parquet
streamlit run app/Home.py                # the app (presentation mode; POWER_PEAK_MODE=full for every page)
```

`scripts/train.py --model {lgbm,lstm,tft,all} [--tune] [--trials N] [--regime-boost X]` saves each model
with its scaler and a config YAML to `models/<name>/<timestamp>/` (`latest.txt` points to the newest);
loss curves go to `reports/training/`. Without `--tune` it reuses `models/<name>/best_params.yaml` if
present; without `--regime-boost` it uses the variant chosen by the backtest (`model_selection.json`).

### How the forecast works
* **Targets** (from the daily PSP report): solar-hour peak (MW), non-solar-hour/evening peak (MW),
  energy met (MU). April 2023 has no solar/evening split in the source; those days are masked.
* **Origin-based samples, no leakage.** A forecast made at origin *t* (last day with actuals) for
  *t+h* only uses actuals and observed weather up to *t*, weather **forecasts issued at t**
  (Open-Meteo Previous-Runs archive, so training and backtests see real forecast error), and the
  calendar. Lags are relative to the origin (`last`=y(t), `lag7`/`lag14`=same weekday 1/2 weeks
  before t+h, `lag364`=same weekday last year). Tests corrupt everything after *t* and assert the
  features don't change (`tests/test_features.py`).
* **All-weather inputs.** Per city (11 cities), per grid region and as the demand-weighted all-India
  value: max/min/mean temperature, cooling-degree days (base 24 °C) and heating-degree days (base
  18 °C), heat index (from temperature and dew point), rain, rain-day flag, 3-day rain sum (observed
  days up to the origin + forecast days after it), daytime and mean cloud cover, shortwave radiation
  (rooftop solar is behind the meter, so cloudy days raise daytime grid demand), humidity and dew
  point, wind speed and gusts, and a North-India fog proxy (early-morning dew-point depression on a
  cool night — Open-Meteo has no archived visibility forecast). **Anomalies** = value minus a smooth
  day-of-year normal per place, fitted only on observed weather up to 2025-10-24 (before every
  backtest origin; `tests/test_weather_features.py`). Archived forecasts of dew point, clouds and
  wind only exist from 2024; for earlier training samples those cells fall back to persistence of
  observed weather up to the origin (never hindsight).
* **Ratio targets.** Models predict y(t+h) / mean(y[t-6..t]) — this lets trees and nets follow the
  ~5 %/yr growth and forecast new records.
* **Quantiles.** Pinball loss for P10/P50/P90; the neural quantile head is monotone by construction;
  every model is conformally calibrated (CQR) on a time-ordered validation slice, and the backtest
  adds a **per-target, per-horizon split-conformal correction** fitted on the previous 180 days of
  out-of-sample residuals only (`src/postprocess.py`), so P10–P90 covers ~80 % at every horizon.
* **Ensemble.** LSTM and LightGBM quantiles blended with a weight per target and horizon, re-learned
  each month on earlier backtest months only (6 warm-up months precede the 12 reported months).
* **Rare-weather sample weights** (`src/models/regime_weights.py`, `*_w` models in the backtest):
  hot, cold/fog and heavy-rain/storm samples (judged from the forecast weather the sample sees)
  count double in the loss; kept only if the overall MAPE does not get worse.
* **Why a hand-written TFT** (`src/models/nets.py`, ~150 lines) instead of pytorch-forecasting /
  neuralforecast: the leakage-tested window builder, ratio targets, 3×3 multi-output and CQR
  would have to be bent around those libraries' dataset/scaling abstractions; a compact
  implementation keeps the inputs exactly under test and exposes the variable-selection weights.

### App (`streamlit run app/Home.py`)

Two modes, chosen by the environment variable `POWER_PEAK_MODE` (default `presentation`; anything
other than `full` means `presentation`). Nothing is removed in either mode: the hidden pages and tabs stay
in the code and are built only in full mode.

```bash
streamlit run app/Home.py                          # presentation mode (the default)
POWER_PEAK_MODE=presentation streamlit run app/Home.py
POWER_PEAK_MODE=full streamlit run app/Home.py     # every page and tab
```

| | Presentation | Full |
|---|---|---|
| Sidebar | Overview · 7-day forecast · Raw to processed · Data pipeline · Our forecast vs Grid-India · Model lab | all ten pages in four sections |
| Model lab tabs | Leaderboard · Evaluation design | Leaderboard · By horizon · By weather · Calibration · Evaluation design · Drift |
| Also in full only | | What-if simulator · Demand explorer · Weather impact · Evening peak (and the old `/Seasonality`, `/States` URLs) |

In presentation mode the other pages are not registered, so their URLs do not open. The **Update data**
button and the sidebar toggle are in both modes. Full mode, by section: **Forecast** — Overview, 7-day
forecast (Solar-hour peak | Evening peak | Energy), What-if simulator · **Performance** — Our forecast vs
Grid-India (Live scorecard | Backtest | Bias-corrected), Model lab · **Analysis** — Demand explorer
(Trends | Seasonality | States), Weather impact (By condition | By season | Next 7 days), Evening peak ·
**Data** — Raw to processed (Raw file | Parse | Clean | Weather | Model input | Files), Data pipeline (Update | History | Data quality). Old URLs still work in full mode.

**Update data (what you see).** The run happens in a background thread of the app server, so the page
stays usable. The panel lists the seven steps with a live state (Waiting, Running, Done, No new data,
Failed) and the elapsed time of each; the weather step carries the note "Fetching weather for 11 cities,
about 4 minutes in total." A step that finds nothing new shows "No new data", not an error. When the
run ends, a **What changed** panel shows the latest data date before and after, the new forecast dates,
and the days added to the live scorecard. Refreshing the page, or opening it in another tab, during a
run re-attaches to the same run (state is read from `logs/pipeline.lock` and `logs/pipeline_progress.json`,
not from the browser session); pressing the button again joins it, never starts a second run. The
run is appended to `logs/pipeline_runs.jsonl` whether or not anyone is watching.

Minimal editorial style defined in one place (`app/ui.py`, `.streamlit/config.toml`, `src/viz.py`):
headline sentence, at most four KPI cards, charts with insight headlines and direct labels, fixed
series colours (Actual #111827, our model #2563EB, Grid-India #D97706 dashed, P10–P90 band). Every
page and tab is covered by AppTest with and without data (no emoji, no per-page styling). The sidebar
button **Update data** runs the daily pipeline; `ENABLE_FETCH_BUTTON=false` hides it.

## Daily pipeline

`src/pipeline.py:run_daily_pipeline(progress_callback)` is the one implementation used by the app
button, `scripts/daily_run.sh` and the systemd timer:

1. Download new Grid-India PSP reports (the last 7 days, plus any gap since the last processed day)
2. Download the latest Grid-India week-ahead forecast
3. Update weather (Open-Meteo observed and archived forecasts; the live forecast is fetched in step 6)
4. Parse files and update the datasets (`scripts/build_dataset.py`, then the data-quality pass)
5. Rebuild features
6. Generate the new 7-day forecast (`src/predict.py`)
7. Update the live scorecard: archived forecasts whose days now have actuals, ours vs Grid-India

If yesterday's report is not published yet, step 1 reports "no new data" and the run continues.
Download steps 1–3 are best-effort; a failure in steps 4–7 stops the run and the remaining steps are
skipped (earlier results stay saved). A lock file (`logs/pipeline.lock`) prevents overlapping runs,
and every run is appended to `logs/pipeline_runs.jsonl`. The live state of a run in progress is
`logs/pipeline_progress.json`, which the app reads to show it.

```bash
scripts/daily_run.sh                       # or: python -m src.pipeline --trigger manual

# tests: the page tests run in full mode (tests/conftest.py); presentation mode has its own tests
python -m pytest -q

# daily timer (user unit, 11:30 local time)
cp deploy/systemd/india-power-forecast.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now india-power-forecast.timer

# one-off history (≈1,300 files, ~35 min at the default 1.5 s polite delay)
python scripts/backfill.py --start 2023-04-01 --forecasts --build
```

All steps are idempotent: existing files are skipped and `build_dataset.py` only parses dates it
has not processed yet (`--rebuild` re-parses everything).

## Data sources

| Source | Grid-India listing type | Content | Coverage |
|---|---|---|---|
| Daily PSP report (`DD.MM.YY_NLDC_PSP_<n>.xls`) | `DAILY_PSP_REPORT` | all-India/regional/state daily energy & peak; 15-min SCADA series | daily from ≥2013; **15-min `TimeSeries` sheet only from 2024-11-04** |
| Week-ahead demand forecast PDF | `DEMAND_FORECAST` | Grid-India's AI forecast for the next 7 days (benchmark) | ~weekly from 2023 |

How downloading works (verified 2026-11-02): grid-india.in is a JS app backed by a JSON API.
`POST https://webapi.grid-india.in/api/v1/file` with
`{"_source":"GRDW","_type":"DAILY_PSP_REPORT","_fileDate":"2026-27","_month":"00"}` lists a whole
financial year, including each file's real path (file names have random suffixes and older years
use a different folder, so URLs can't be constructed). Files are served from
`https://webcdn.grid-india.in/<FilePath>`. The old `report.grid-india.in/ReportData/...` host no
longer resolves. Listings are cached in `data/raw/_index/`.

## Datasets (`data/processed/`)

| File | Grain | Key columns |
|---|---|---|
| `daily_summary.parquet` | 1 row / day | `<metric>_<region>` for region ∈ nr, wr, sr, er, ner, all_india; metrics: `energy_met_mu`, `max_demand_mw`, `max_demand_time`, `evening_peak_demand_mw`, `evening_peak_shortage_mw`, `hydro/wind/solar_gen_mu`, `energy_shortage_mu`; plus `peak_demand_solar_hr_*` / `peak_demand_non_solar_hr_*` (all-India) |
| `state_demand.parquet` | day × state | `region, state, max_demand_mw, shortage_at_max_demand_mw, energy_met_mu, drawal_schedule_mu, od_ud_mu, energy_shortage_mu` |
| `timeseries_15min.parquet` | 15 min | `timestamp, frequency_hz, demand_met_mw, nuclear/wind/solar/hydro/gas/thermal/others_mw, storage…, net_demand_met_mw, total_generation_mw, net_transnational_exchange_mw` |
| `forecast_daily.parquet` | forecast week × day | `forecast_start, date, low_mw, high_mw` (daily peak) + `solar_hr_*`, `non_solar_hr_*`, `label_kind`, `curve_ok` |
| `forecast_curve.parquet` | forecast week × 15 min | `forecast_start, timestamp, forecast_mw, band_low_mw, band_high_mw` |

`date` is always the **data date** (from the filename); the report is published the next day.
Units: MW = instantaneous power, MU = million units = GWh of energy per day.

## Parsing notes

* **No hard-coded rows.** Every block is found by matching label text (case/whitespace/footnote
  tolerant). This matters: 2023 files have rows shifted by one, labels like "Peak Shortage (MW)"
  instead of "Shortage (MW) during Evening Peak", some days are `.xlsx`, and there is no
  TimeSeries sheet before 2024-11-04. TimeSeries columns also grow over time (14 → 16).
* **State regions.** In section C the region label sits on the *middle* row of each group (it
  looks merged but isn't), so forward-filling would mislabel e.g. Punjab. Regions come from a
  canonical state→region map, `Bulk Consumer_XX`/`Railways_XX` suffixes, and in-sheet labels as a
  cross-check; unknown names fall back to the nearest label with a warning.
* **Forecast PDFs** are vector charts, read exactly with pdfplumber and calibrated from the chart's
  own tick labels/gridlines. There are **two labels per day** (solar-hour ≈ 10:00 and
  non-solar-hour ≈ 18:30 peaks), each cross-checked against the drawn curve (`curve_ok`).
  Three generations exist: GW ranges `"234-238"` (Oct-2026), MW points `"246699"` (Jan-2026), and
  2024–25 charts whose labels are vector outlines (not text) → daily peaks are taken from the
  digitised curve (`label_kind="curve"`, accurate to ~0.1 GW). Archive titles contain typos, so
  the start date is cross-checked against the footer date.

## Layout

```
data/raw/psp/        daily PSP .xls/.xlsx (git-ignored)
data/raw/forecast/   week-ahead forecast PDFs (git-ignored)
data/raw/weather/    Open-Meteo caches: obs_/fc_ (base), obsx_/fcx_ (tmin, dew point, clouds, wind, fog proxy)
data/processed/      parquet datasets, backtest outputs, live forecast (git-ignored)
src/fetch/           grid_india.py — listing API client + downloader
src/parse/           psp_parser.py, forecast_parser.py
src/data/            quality.py (gaps, outliers), weather.py (Open-Meteo)
src/features/        build.py (leakage-safe samples, all-weather features, normals), calendar.py
src/models/          baselines.py (seasonal naive, LightGBM, Grid-India), nets.py + deep.py (LSTM, TFT),
                     regime_weights.py, tune.py (Optuna), registry.py
src/postprocess.py   ensemble weights + per-horizon conformal calibration (past-only)
src/evaluate.py      rolling backtest, leaderboards, weather regimes, seasons
src/audit.py         fairness audit vs Grid-India (issue dates, long-lead model, bias)
src/insights.py      record probability (+ recalibration), explanations, what-if, evening stress
src/report.py        reports/results.md
src/pipeline.py      daily pipeline (app button, scripts/daily_run.sh, systemd timer)
app/                 Streamlit app (Home + 10 pages)
scripts/             backfill, build_dataset, train, backtest, audit, predict, daily_run.sh
reports/             results.md, fairness_audit.md, data_quality.md, training/ (loss curves)
tests/               pytest + small real fixtures
```

## Roadmap

* **Phase 2 (done)** — baselines, LightGBM, LSTM, TFT, ensemble; all-weather features; rolling
  backtest against Grid-India's week-ahead forecast with a matched-issue-time audit; Streamlit app;
  daily pipeline.
* **Phase 3** — see "Future work" in reports/results.md.
