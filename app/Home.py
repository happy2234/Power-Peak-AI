"""Power Peak AI: app entry point and router.

    streamlit run app/Home.py

Nine pages in four sections, drawn by ui.sidebar (wordmark, "Jump to…" search, grouped links,
"Update data", data timestamp); sub-views are tabs inside each page. st.navigation only
routes (position="hidden"), so the sidebar can filter its links. Old page URLs (/Forecast,
/Scorecard, /What_If, ...) keep working; /Seasonality and /States redirect to Demand explorer,
whose tabs replaced them.

POWER_PEAK_MODE chooses what is shown (ui.app_mode): "presentation" (the default) registers and
lists only Overview, 7-day forecast, Data pipeline, Our forecast vs Grid-India and Model lab;
"full" adds What-if simulator, Demand explorer, Weather impact and Evening peak (and the old
/Seasonality and /States URLs). The hidden pages stay in the code, unchanged.
"""
import os
from pathlib import Path

import common  # noqa: F401  (sets sys.path)
import streamlit as st

import ui

# Streamlit Community Cloud keeps settings in the app's "Secrets" (TOML) and does not document them as
# environment variables, so copy the two switches across. Real environment variables win.
for _k in ("POWER_PEAK_READONLY", "POWER_PEAK_MODE"):
    try:
        if _k not in os.environ and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:       # no secrets file (laptop, tests): nothing to copy
        pass

st.set_page_config(page_title="Power Peak AI", page_icon=":material/bolt:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()
st.session_state["_pp_script_run"] = st.session_state.get("_pp_script_run", 0) + 1   # see ui.pipeline_panel

_VIEWS = Path(__file__).resolve().parent / "views"       # absolute, so it works from app/ and from streamlit_app.py

P = {
    "overview": st.Page(str(_VIEWS / "overview.py"), title="Overview", icon=":material/home:", default=True),
    "forecast": st.Page(str(_VIEWS / "forecast.py"), title="7-day forecast", icon=":material/timeline:", url_path="Forecast"),
    "what_if": st.Page(str(_VIEWS / "what_if.py"), title="What-if simulator", icon=":material/tune:", url_path="What_If"),
    "scorecard": st.Page(str(_VIEWS / "vs_grid_india.py"), title="Our forecast vs Grid-India", icon=":material/compare_arrows:",
                         url_path="Scorecard"),
    "model_lab": st.Page(str(_VIEWS / "model_lab.py"), title="Model lab", icon=":material/science:", url_path="Model_Lab"),
    "explorer": st.Page(str(_VIEWS / "demand_explorer.py"), title="Demand explorer", icon=":material/show_chart:",
                        url_path="Demand_Explorer"),
    "weather": st.Page(str(_VIEWS / "weather.py"), title="Weather impact", icon=":material/partly_cloudy_day:",
                       url_path="Weather"),
    "evening": st.Page(str(_VIEWS / "evening_peak.py"), title="Evening peak", icon=":material/nights_stay:",
                       url_path="Evening_Peak"),
    "pipeline": st.Page(str(_VIEWS / "data_pipeline.py"), title="Data pipeline", icon=":material/storage:",
                        url_path="Data_Pipeline"),
    # old URLs of pages that became tabs (routed, never listed)
    "old_seasonality": st.Page(str(_VIEWS / "redirect_seasonality.py"), title="Seasonality", url_path="Seasonality"),
    "old_states": st.Page(str(_VIEWS / "redirect_states.py"), title="States", url_path="States"),
}

ALL_PAGES = P
if ui.full_mode():
    SECTIONS = {
        "Forecast": [P["overview"], P["forecast"], P["what_if"]],
        "Performance": [P["scorecard"], P["model_lab"]],
        "Analysis": [P["explorer"], P["weather"], P["evening"]],
        "Data": [P["pipeline"]],
    }
    ROUTES = [*[p for ps in SECTIONS.values() for p in ps], P["old_seasonality"], P["old_states"]]
else:                                          # presentation: the five pages, in the order they are narrated
    P = {k: ALL_PAGES[k] for k in ui.PRESENTATION_PAGES}
    SECTIONS = {"Forecast": [P["overview"], P["forecast"]], "Data": [P["pipeline"]],
                "Performance": [P["scorecard"], P["model_lab"]]}
    ROUTES = list(P.values())                  # the other pages are not registered: their URLs do not open
st.session_state["pp_pages"] = P               # only registered pages can be linked to
# extra words the "Jump to…" search matches, beyond the page and section names (mostly tab names)
KEYWORDS = {
    "overview": "home outlook kpi record",
    "forecast": "solar-hour peak evening peak energy download csv drivers",
    "what_if": "scenario temperature holiday simulator",
    "scorecard": "live scorecard backtest bias-corrected grid-india compare win rate",
    "model_lab": ("leaderboard by horizon by weather calibration evaluation design drift mape metrics"
                  if ui.full_mode() else "leaderboard evaluation design mape metrics"),
    "explorer": "trends seasonality states festivals weekday",
    "weather": "by condition by season next 7 days heat rain fog",
    "evening": "stress alert solar gap",
    "pipeline": "update history data quality freshness runs",
}
ui.settle_run()                                # an update this session was watching has ended
nav = st.navigation(ROUTES, position="hidden")
clicked = ui.sidebar(SECTIONS, {P[k].url_path: v for k, v in KEYWORDS.items() if k in P})
if clicked or ui.update_running():             # a run in progress is shown on every page, also after a refresh
    ui.pipeline_panel(start=clicked)
nav.run()
