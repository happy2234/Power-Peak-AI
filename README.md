# Power Peak (clean project)

Forecasts India's daily electricity demand for the next 7 days from Grid-India's daily PSP reports and
weather: three targets (solar-hour peak MW, evening peak MW, energy met MU), each as P10 / P50 / P90.
Models: LightGBM, an LSTM and a TFT, blended into an ensemble. A Streamlit dashboard shows the results; a PyQt6
walkthrough app explains the pipeline step by step on one real day.

## Folder map

| Folder | What is in it |
|---|---|
| `data/raw/psp/` | 1,320 daily Grid-India PSP Excel reports |
| `data/raw/weather/` | observed (`obs_`, `obsx_`) and archived forecast (`fc_`, `fcx_`) weather, 11 cities |
| `data/raw/forecast/` | Grid-India week-ahead forecast PDFs (the benchmark) |
| `data/processed/` | parsed and cleaned tables, `features.parquet`, forecasts, back-tests |
| `models/{lgbm,lstm,tft}/` | the latest production run of each (`latest.txt` points to it) + tuned `best_params*.yaml` |
| `reports/training/` | training histories (loss per epoch) of those runs |
| `src/ scripts/ app/ tests/` | the pipeline, command-line scripts, Streamlit dashboard, tests |
| `walkthrough/` | the PyQt6 walkthrough app (`app.py`, data layer `wt_data.py`, `screens/` screenshots) |
| `_archive_data/` | git-ignored extras: NASA weather, older model runs, baseline snapshots, other processed files |

`data/`, `models/` and `_archive_data/` are git-ignored (kept local only).

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-walkthrough.txt        # includes requirements-pipeline.txt (torch CPU, lightgbm, streamlit ...)

POWER_PEAK_READONLY=1 streamlit run streamlit_app.py     # dashboard, read-only (no "Update data")
streamlit run streamlit_app.py                           # dashboard with "Update data" (needs network)
pytest -q                                                # tests

python walkthrough/app.py                  # walkthrough on the latest day with actuals
python walkthrough/app.py --date 2026-11-01
QT_QPA_PLATFORM=offscreen python walkthrough/app.py --selftest      # builds all steps, writes walkthrough/screens/*.png
```

Walkthrough keys: Left / Right (or PgUp / PgDn) = previous / next step, F11 = full screen, Esc = leave full
screen, Ctrl + / Ctrl - = zoom (or the A- / A+ buttons). It never writes to `models/` or `data/`; live
training writes only to a temp folder.

## Rehearsal checklist (5 lines)

1. Run `--selftest` once on the projector machine: every step must print `OK`; glance at `walkthrough/screens/`.
2. Start the app, press F11, then set the zoom (A+) until the back row can read the tables.
3. Click through steps 1-10 with the arrow keys; step 8: play Replay, then run Live once (about 10-30 s) and keep Cancel in mind.
4. If Live is slow or fails it falls back to Replay by itself after 90 s; no network is needed at any point.
5. Close other windows and notifications; keep `--date` for a second example day (e.g. a holiday) ready.
