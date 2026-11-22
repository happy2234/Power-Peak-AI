"""Raw to processed: one real day, from the raw Grid-India Excel file to the model input.

Six tabs follow the data: Raw file -> Parse -> Clean -> Weather -> Model input -> Files. Pure display:
every number is read from a file by src/walkthrough.py (no write, no network). Without the raw file,
tabs 3 to 6 still show the latest day of the cleaned data."""
import html

import common  # noqa: F401
import pandas as pd
import streamlit as st

import ui
from src import fmt
from src import walkthrough as wt
from src.config import TARGET_LABELS, TARGETS
from src.data.weather import CITIES
from src.viz import recent_target_chart

ui.page("Raw to processed", "One real day, followed from the raw Grid-India Excel file to the numbers the models read.",
        badges=[ui.updated_badge()])

def num(v, nd: int = 2) -> str:
    return "–" if v is None or pd.isna(v) else (f"{v:,.0f}" if float(v) % 1 == 0 else f"{v:,.{nd}f}")


def show(v) -> str:
    if isinstance(v, (bool,)) or (hasattr(v, "dtype") and str(getattr(v, "dtype", "")) == "bool"):
        return "yes" if v else "no"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return num(v, 4)
    if isinstance(v, pd.Timestamp):
        return fmt.date(v) if pd.notna(v) else "–"
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


raw = wt.latest_raw_file()
overview = wt.raw_overview(raw) if raw else None
cells = wt.target_cells(raw) if raw else None
day = pd.Timestamp(overview["data_date"]) if overview else wt.latest_clean_date()
RAW_MISSING = ("The raw Grid-India file is not in this snapshot (data/raw/psp_latest is empty). "
               "The scheduled snapshot job copies the newest report there.")

if day is None:
    ui.empty_state("No cleaned data in this snapshot yet. The daily pipeline creates it:", "pipeline")
    ui.stop()

if overview:
    ui.headline(f"Following *{fmt.date(day)}* from {overview['name']} to the model input.")
else:
    ui.headline(f"No raw file here: showing the latest cleaned day, *{fmt.date(day)}*.")

tabs = st.tabs(["Raw file", "Parse", "Clean", "Weather", "Model input", "Files"])

# ------------------------------------------------------------------------------------------ 1 Raw file
with tabs[0]:
    if overview is None:
        ui.empty_state(RAW_MISSING, None)
    else:
        ui.kpis([{"label": "Workbook", "value": f"{len(overview['sheets'])} sheets", "note": overview["name"]},
                 {"label": "Size", "value": f"{overview['size_bytes'] / 1024:,.0f}", "unit": "KB"},
                 {"label": "Data date", "value": fmt.date(overview["data_date"]), "note": "from the file name"},
                 {"label": "Report date", "value": fmt.date(overview["report_date"]), "note": "Date of Reporting, in MOP_E"}])
        ui.section("The five sheets of the workbook")
        ui.table(pd.DataFrame([{"Sheet": s["sheet"], "Rows": s["rows"], "Columns": s["columns"]}
                               for s in overview["sheets"]]), max_height=None, index=False, name="raw-sheets", download=False)
        ui.note("Grid-India publishes the report the day after: the data date is the day described, the report date "
                "is the day it was issued. The three targets sit on the MOP_E sheet.")
        ui.section("MOP_E: sections A and I, as stored in the file")
        ui.chip_line("highlighted", "the three cells the pipeline reads: "
                     + ", ".join(f"{c['name']} {c['cell']}" for c in cells))
        for s in wt.mop_e_sections(raw):
            ui.kicker(s["title"])
            ui.grid_table(s["rows"], s["n_cols"], [wt.col_letter(i) for i in range(s["n_cols"])])
        st.download_button("Download the raw Excel file", raw.read_bytes(), file_name=raw.name,
                           mime="application/vnd.ms-excel", icon=":material/download:", key="rp_dl_raw")

# ------------------------------------------------------------------------------------------ 2 Parse
with tabs[1]:
    if cells is None:
        ui.empty_state(RAW_MISSING, None)
    else:
        stored = wt.daily_summary_row(day)
        ui.headline("The parser finds each number by its *label*, not by a fixed row or column.")
        ui.section("Raw cell → column → value")
        ui.table(pd.DataFrame([{
            "Target": c["name"], "Sheet": c["sheet"], "Row label": c["label"], "Excel cell": c["cell"],
            "Raw value": f"{num(c['value'])} {c['unit']}", "Column in daily_summary": c["column"],
            "Stored value": "–" if stored is None else f"{num(stored[c['column']])} {c['unit']}"} for c in cells]),
            max_height=None, index=False, name="parse-targets", download=False)
        if stored is not None and "evening_peak_demand_mw_all_india" in stored:
            ui.note(f"Not the evening-peak target: the row 'Demand Met during Evening Peak hr (19:00hrs)' "
                    f"({num(stored['evening_peak_demand_mw_all_india'])} MW) is only the 19:00 reading. The target is "
                    f"the Non-Solar hr row of section I.")
        ui.section("Check: parse the file again, compare with the stored row")
        cmp = wt.compare_with_stored(raw)
        if cmp is None:
            ui.empty_state(f"There is no stored daily_summary row for {fmt.date(day)}.", "pipeline")
        else:
            if cmp["match"]:
                ui.check(True, f"MATCH: the re-parsed row equals the stored daily_summary row, {cmp['n_match']} of "
                               f"{cmp['n_columns']} columns identical.")
            else:
                bad = [r[0] for r in cmp["rows"] if not r[3]]
                ui.check(False, f"MISMATCH: {len(bad)} of {cmp['n_columns']} columns differ: {', '.join(bad)}")
            with st.expander(f"All {cmp['n_columns']} columns: re-parsed vs stored"):
                ui.table(pd.DataFrame([{"Column": c, "Re-parsed": show(a), "Stored": show(b),
                                        "Same": "yes" if ok else "NO"} for c, a, b, ok in cmp["rows"]]),
                         max_height=420, index=False, name="parse-compare", download=False)

# ------------------------------------------------------------------------------------------ 3 Clean
with tabs[2]:
    row, counts = wt.daily_clean_row(day), wt.clean_counts()
    if row is None or counts is None:
        ui.empty_state(f"There is no daily_clean row for {fmt.date(day)} in this snapshot.", "pipeline")
    else:
        flags = [c for c in row.index if c.startswith("is_") or c.endswith("_imputed")]
        ui.kpis([{"label": "Days in daily_clean", "value": f"{counts['days']:,}",
                  "note": f"{fmt.date(counts['first'])} to {fmt.date(counts['last'])}"},
                 {"label": "Imputed days", "value": f"{counts['imputed']:,}", "note": "filled in by the cleaner"},
                 {"label": "Outliers", "value": f"{counts['outliers']:,}", "note": "flagged, kept"},
                 {"label": f"{fmt.date(day)}", "value": "imputed" if row["is_imputed"] else "as reported",
                  "note": "outlier" if row["is_outlier"] else "not an outlier"}])
        ui.section(f"The cleaned row of {fmt.date(day)}")
        parsed = wt.daily_summary_row(day)
        cmp_rows = []
        for t, col in TARGETS.items():
            p = None if parsed is None else parsed.get(col)
            cmp_rows.append({"Target": TARGET_LABELS[t], "Parsed (daily_summary)": num(p),
                             "Cleaned (daily_clean)": num(row[t]),
                             "Imputed": show(bool(row[f"{t}_imputed"])),
                             "Same as parsed": "–" if p is None or pd.isna(p) else ("yes" if row[t] == p else "no")})
        ui.table(pd.DataFrame(cmp_rows), max_height=None, index=False, name="clean-targets", download=False)
        left, right = st.columns(2)
        with left:
            ui.kicker("Flags")
            ui.table(pd.DataFrame({"Flag": flags, "Value": [show(bool(row[f])) for f in flags]}),
                     max_height=None, index=False, name="clean-flags", download=False)
        with right:
            ui.kicker("Every column of the row")
            ui.table(pd.DataFrame({"Column": [c for c in row.index if c != "date"],
                                   "Value": [show(row[c]) for c in row.index if c != "date"]}),
                     max_height=420, index=False, name="clean-row", download=False)
        ui.section("The last 120 days of the three targets")
        rec = wt.recent_clean(day, 120)
        cols = st.columns(3)
        for col, (t, flag) in zip(cols, [(t, f"{t}_imputed") for t in TARGETS]):
            with col:
                ui.chart(recent_target_chart(rec, t, flag), half=True, key=f"rp_recent_{t}")
        n_imp = int(rec["is_imputed"].sum())
        ui.note(f"Red rings mark imputed days ({n_imp} in these 120 days). Outliers are flagged but kept." )

# ------------------------------------------------------------------------------------------ 4 Weather
with tabs[3]:
    cw, weights = wt.city_weather_table(day), wt.region_weights()
    if cw is None or weights is None:
        ui.empty_state(f"There is no city weather for {fmt.date(day)} in this snapshot.", "pipeline")
    else:
        ui.section(f"The 11 cities on {fmt.date(day)}")
        ui.table(cw.rename(columns={"city": "City", "region": "Region", "tmax": "Max temp (°C)", "tmean": "Mean temp (°C)",
                                    "app_tmax": "Feels-like max (°C)", "rh_mean": "Humidity (%)", "precip": "Rain (mm)",
                                    "rad": "Solar radiation (MJ/m²)"}).assign(City=lambda d: d["City"].str.title()),
                 max_height=None, index=False, name="weather-cities", download=False)
        ui.section("Region weights")
        ui.table(pd.DataFrame([{"Region": r.upper(), "Cities": ", ".join(c.title() for c, v in CITIES.items() if v[2] == r),
                                "Weight (%)": w * 100} for r, w in weights.items()]),
                 max_height=None, index=False, name="region-weights", download=False)
        ui.note("A region's weight is its share of all-India energy met over the first financial year "
                "(fixed, so it is known before any forecast).")
        ui.section("From cities to one all-India number")
        chk = wt.weather_check(day, "tmax")
        terms = " + ".join(f"{weights[r] * 100:.1f}% × {r.upper()}" for r in weights)
        ui.formula("<b>Formula.</b> First each region's value is the <i>average of its cities</i>. "
                   f"Then the all-India value is the weighted average of the five regions: <b>{html.escape(terms)}</b>.")
        if chk:
            ui.section("Worked example: maximum temperature")
            ui.table(pd.DataFrame([{"Region": r.upper(), "Average of its cities (°C)": f"{chk['regional'][r]:.2f}",
                                    "Weight (%)": f"{chk['weights'][r] * 100:.1f}",
                                    "Contribution (°C)": f"{chk['weights'][r] * chk['regional'][r]:.2f}"} for r in chk["regional"]]),
                     max_height=None, index=False, name="weather-example", download=False)
            ui.kpis([{"label": "Rebuilt all-India max temp", "value": f"{chk['rebuilt']:.2f}", "unit": "°C"},
                     {"label": "Stored tmax_ai (features.parquet)", "value": f"{chk['stored']:.2f}", "unit": "°C"},
                     {"label": "Check", "value": "match" if chk["match"] else "differs"}])
        feat = wt.features_row(day)
        if feat is not None:
            ui.section("Stored all-India values")
            names = [n for n in ("tmax_ai", "tmin_ai", "tmean_ai", "app_tmax_ai", "rh_mean_ai", "precip_ai", "rad_ai",
                                 "cdd_ai", "hdd_ai") if n in feat.index]
            ui.table(pd.DataFrame({"Column": names, "Value": [num(feat[n], 3) for n in names]}),
                     max_height=None, index=False, name="weather-ai", download=False)
            ui.note("cdd_ai is cooling degree days: how far the mean temperature is above the base temperature; "
                    "demand for cooling grows with it.")

# ------------------------------------------------------------------------------------------ 5 Model input
with tabs[4]:
    groups, ex = wt.feature_groups(day), wt.scaled_examples(day)
    if groups is None:
        ui.empty_state(f"There is no features row for {fmt.date(day)} in this snapshot.", "pipeline")
    else:
        n_feat = sum(len(g) for g in groups.values())
        ui.section(f"The features row of {fmt.date(day)}: {n_feat} columns")
        for name, g in groups.items():
            with st.expander(f"{name}: {len(g)} columns", expanded=False):
                if g.empty:
                    ui.note(f"No {name.lower()} columns are stored in features.parquet: they are built when the model "
                            "sequences are made (see the list below).")
                else:
                    ui.table(g.assign(value=[show(v) for v in g["value"]]).rename(columns={"feature": "Feature", "value": "Value"}),
                             max_height=360, index=False, name=f"features-{name.lower()}", download=False)
        seq = wt.sequence_only_features("lstm")
        if seq:
            ui.note("Built when sequences are made (ratios *_r, masks *_mask, lags, forecast-weather flags), not stored in "
                    "features.parquet: " + ", ".join(seq) + ".")
        ui.section("Raw → scaled (LSTM encoder)")
        if ex is None or ex.empty:
            ui.empty_state("The LSTM scaler is not in this snapshot.", None)
        else:
            show_all = st.toggle(f"Show all {len(ex)} encoder features stored in features.parquet", key="rp_all_scaled")
            pick = ex if show_all else ex[ex["feature"].isin(["cdd_ai", "tmax_ai", "dow_sin", "tmin_ai", "precip_ai", "is_sunday"])]
            ui.table(pd.DataFrame({"Feature": pick["feature"], "Raw value": [f"{v:,.4f}" for v in pick["raw"]],
                                   "Mean (enc_mu)": [f"{v:,.4f}" for v in pick["enc_mu"]],
                                   "Std (enc_sd)": [f"{v:,.4f}" for v in pick["enc_sd"]],
                                   "Scaled": [f"{v:,.4f}" + (" (clipped)" if c else "") for v, c in zip(pick["scaled"], pick["clipped"])]}),
                     max_height=420, index=False, name="scaled", download=False)
            ui.note("scaled = (raw − mean) ÷ std, then clipped to −10 … +10. The mean and std come from the model's "
                    "scaler.npz, fitted on the training period.")
        ui.section("Features per model")
        ms = wt.model_summary()
        if not ms:
            ui.empty_state("No trained models in this snapshot.", None)
        else:
            ui.table(pd.DataFrame([{"Model": m["model"].upper() if m["model"] != "lgbm" else "LightGBM", "Run": m["run"],
                                    "Encoder (past)": "–" if m["encoder"] is None else m["encoder"],
                                    "Decoder (known ahead)": "–" if m["decoder"] is None else m["decoder"],
                                    "Distinct": m["distinct"], "Train end": m["train_end"]} for m in ms]),
                     max_height=None, index=False, name="model-features", download=False)
            ui.note("LightGBM has one flat list of features. The LSTM and TFT read past features (encoder) and features "
                    "known in advance (decoder); distinct counts a feature once if it is in both.")
        w = wt.window_info()
        ui.section("The window and the output")
        ui.kpis([{"label": "Look-back (encoder)", "value": w["encoder_days"], "unit": "days"},
                 {"label": "Forecast (decoder)", "value": w["horizon"], "unit": "days"},
                 {"label": "Training ends", "value": fmt.date(w["train_end"]) if w["train_end"] else "–"},
                 {"label": "Outputs per forecast", "value": w["outputs"],
                  "note": f"{len(w['targets'])} targets × {w['horizon']} days × {len(w['quantiles'])} quantiles"}])

# ------------------------------------------------------------------------------------------ 6 Files
with tabs[5]:
    chain = wt.file_chain()
    ui.section("Every file in the chain")
    ui.table(pd.DataFrame([{"Step": c["step"], "File": c["file"] if c["exists"] else f"{c['file']} (missing)",
                            "Rows": c["rows"] if c["rows"] is not None else c["detail"] or "–",
                            "Columns": c["columns"] if c["columns"] is not None else "–",
                            "Size (KB)": "–" if c["size_kb"] is None else f"{c['size_kb']:,.1f}",
                            "Modified": fmt.datetime(c["modified"])} for c in chain]),
             max_height=None, index=False, name="file-chain", download=False)
    ui.note("Read from the files on disk: row and column counts come from each file's own metadata.")

ui.footer()
