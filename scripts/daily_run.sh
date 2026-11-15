#!/usr/bin/env bash
# Daily pipeline for the systemd timer / cron. Same implementation as the app's
# "Fetch latest data and update forecast" button: src/pipeline.py:run_daily_pipeline.
#   scripts/daily_run.sh                    # trigger=timer
#   PIPELINE_TRIGGER=manual scripts/daily_run.sh --no-live-weather
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" -m src.pipeline --trigger "${PIPELINE_TRIGGER:-timer}" "$@"
