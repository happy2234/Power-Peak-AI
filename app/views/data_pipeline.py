"""Data pipeline: the daily update flow, run history, and data quality."""
import common  # noqa: F401
import pandas as pd
import streamlit as st

import ui
from src.config import readonly
from src import fmt
from src import pipeline as pl

ui.page("Data pipeline", "How the daily update runs, how fresh each source is, and the result of every run.", badges=[ui.updated_badge()])
runs = pl.read_runs(n=10)
tabs = st.tabs(["Update", "History", "Data quality"])


def _age(d) -> str:
    if d is None or pd.isna(d):
        return "–"
    n = (pd.Timestamp(d).normalize() - pd.Timestamp.today().normalize()).days
    unit = "day" if abs(n) == 1 else "days"
    return "today" if n == 0 else f"{-n} {unit} behind" if n < 0 else f"{n} {unit} ahead"


with tabs[0]:
    if runs:
        last = runs[0]
        ok = last["status"] in ("success", "partial")
        ui.headline(f"Last update {ui.stamp(last['started'])} ({ui.TRIGGER.get(last['trigger'], last['trigger']).lower()}): "
                    f"{'completed' if ok else 'failed'}" + (" with warnings." if last["status"] == "partial" else "."))
        ui.flow(last["steps"])
        st.caption("State of each step in the latest run. Download steps 1–3 are best-effort: if one fails, the run "
                   "continues with the data already on disk. A failure in steps 4–7 stops the run; results of the "
                   "steps already finished stay saved.")
    else:
        ui.flow([pl.StepResult(k, label, short) for k, label, short in pl.STEPS])
        st.caption("No runs recorded in this snapshot yet." if readonly() else
                   "No runs yet. Use Update data in the sidebar to start the first one.")
    with st.expander("What each step does"):
        st.markdown("\n".join(f"{i}. **{label}**" for i, (_, label, _) in enumerate(pl.STEPS, 1)))
        st.markdown("The same function, `src.pipeline.run_daily_pipeline`, runs from the button in the sidebar, "
                    "from `scripts/daily_run.sh` and from the systemd timer in `deploy/systemd/`. A lock file "
                    "prevents overlapping runs.")
    if readonly():
        st.caption("This is a read-only data snapshot: the app cannot run updates. A scheduled GitHub Actions "
                   "job runs the daily pipeline and refreshes the data.")
    elif not ui.fetch_enabled():
        st.caption("Updates from the app are turned off on this deployment (ENABLE_FETCH_BUTTON=false); the "
                   "scheduled run keeps the data current.")
    st.subheader("Data freshness")
    fr = pl.data_freshness()
    fr["age"] = [_age(x) for x in fr["latest_data"]]
    ui.table(pd.DataFrame({"Source": fr["source"], "Latest data": pd.to_datetime(fr["latest_data"]),
                           "Relative to today": fr["age"], "File updated": [fmt.datetime(t) for t in fr["file_updated"]],
                           "Meaning": fr["meaning"]}), max_height=None)
    st.caption("Grid-India publishes each day's report the following day, so actuals are normally one to two days "
               "behind today. Our forecast covers the 7 days after the latest actuals.")

with tabs[1]:
    if not runs:
        ui.empty_state("No runs recorded in this snapshot yet. The scheduled job adds one each day.", None) if readonly() \
            else ui.empty_state("No runs recorded yet. The first run starts the history:", "pipeline")
    else:
        RESULT = {"success": "Success", "partial": "Completed with warnings", "failed": "Failed", "running": "Running",
                  "interrupted": "Interrupted"}
        rows = []
        for r in runs:
            bad = next((s for s in r["steps"] if s["status"] == pl.FAILED), None)
            rows.append({"Started": fmt.datetime(r["started"]), "Trigger": ui.TRIGGER.get(r["trigger"], r["trigger"]),
                         "Result": RESULT.get(r["status"], r["status"]), "Duration": fmt.duration(r.get("duration_s")),
                         "Failed step": bad["label"] if bad else "–", "Summary": r.get("summary", "")})
        n_ok = sum(r["status"] in ("success", "partial") for r in runs)
        ui.headline(f"{n_ok} of the last {len(runs)} runs completed.")
        ui.table(pd.DataFrame(rows), max_height=None)
        labels = [f"{fmt.datetime(r['started'])} ({r['trigger']})" for r in runs]
        pick = st.selectbox("Run details", range(len(runs)), format_func=lambda i: labels[i])
        r = runs[pick]
        ui.flow(r["steps"])
        ui.steps_table(r["steps"])
        bad = next((s for s in r["steps"] if s.get("traceback")), None)
        if bad:
            with st.expander(f"Traceback: {bad['label']}"):
                st.code(bad["traceback"], language="text")

with tabs[2]:
    q = common.quality_md()
    if q is None:
        ui.empty_state("No data-quality report in this snapshot.", None) if readonly() \
            else ui.empty_state("No data-quality report yet. The daily pipeline writes it:", "pipeline")
    else:
        body = q.split("\n", 1)[1] if q.startswith("# ") else q          # the page already has a title
        st.markdown(body.replace("\n## ", "\n#### "))
ui.footer()
