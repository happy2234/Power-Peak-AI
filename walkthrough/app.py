#!/usr/bin/env python
"""Power Peak walkthrough: the pipeline step by step on ONE real day (PyQt6 + matplotlib).

    python walkthrough/app.py                    # latest day with actuals
    python walkthrough/app.py --date 2026-11-01
    QT_QPA_PLATFORM=offscreen python walkthrough/app.py --selftest

Every number comes from real files (data/, models/, reports/) or from the project's own code (src/).
A missing file only disables its own step. Nothing is written to models/ or data/ (live training writes
to a temp folder).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt6.QtCore import QEventLoop, QObject, Qt, QThread, QTimer, QUrl, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QBrush, QColor, QDesktopServices, QFont, QKeySequence, QShortcut  # noqa: E402
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QComboBox, QFrame, QHBoxLayout,  # noqa: E402
                             QHeaderView, QLabel, QListWidget, QMainWindow, QPlainTextEdit, QPushButton,
                             QRadioButton, QScrollArea, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
                             QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

import wt_data as D  # noqa: E402

BASE_PT = 14
BLUE, RED, TEAL, ORANGE, INK, GREY = "#0b3d91", "#c1121f", "#007f73", "#d97706", "#111111", "#8a8a8a"
YELLOW = "#ffe066"
LIVE_BUDGET_S = 90

matplotlib.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 13,
                            "ytick.labelsize": 13, "legend.fontsize": 13, "lines.linewidth": 2.4,
                            "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                            "axes.spines.right": False, "figure.facecolor": "white"})


# =============================================================================================== helpers
@dataclass
class Ctx:
    day: pd.Timestamp
    tmp: Path
    canvases: list = field(default_factory=list)
    zoom: float = 1.0
    lines: list = field(default_factory=list)
    keep: list = field(default_factory=list)       # threads / objects that must outlive a step build

    def log(self, msg: str):
        self.lines.append(msg)


def label(text="", bold=False, size=None, color=None, wrap=True, mono=False) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(wrap)
    lb.setTextFormat(Qt.TextFormat.RichText if "<" in text else Qt.TextFormat.PlainText)
    st = []
    if bold:
        st.append("font-weight:bold")
    if size:
        st.append(f"font-size:{size}pt")
    if color:
        st.append(f"color:{color}")
    if mono:
        st.append("font-family:monospace")
    if st:
        lb.setStyleSheet(";".join(st))
    return lb


def heading(text) -> QLabel:
    lb = label(text, bold=True, size=BASE_PT + 3)
    lb.setObjectName("h")
    return lb


class Canvas(FigureCanvasQTAgg):
    def __init__(self, ctx: Ctx, rows=1, cols=1, height=330, **kw):
        self.fig = Figure(figsize=(8, 4), dpi=100 * ctx.zoom, layout="constrained")
        super().__init__(self.fig)
        self.axes = self.fig.subplots(rows, cols, **kw)
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ctx.canvases.append(self)


def table(df: pd.DataFrame, fmt=None, highlight=None, max_col=420) -> QTableWidget:
    """Read-only table from a DataFrame. fmt: {col: format str or callable}."""
    t = QTableWidget(len(df), len(df.columns))
    t.setHorizontalHeaderLabels([str(c) for c in df.columns])
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    t.verticalHeader().setVisible(False)
    fmt = fmt or {}
    for i in range(len(df)):
        for j, c in enumerate(df.columns):
            v = df.iat[i, j]
            f = fmt.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                s = ""
            elif callable(f):
                s = f(v)
            elif f:
                try:
                    s = format(v, f)
                except (ValueError, TypeError):
                    s = str(v)
            elif isinstance(v, (float, np.floating)):
                s = f"{v:,.2f}"
            elif isinstance(v, pd.Timestamp):
                s = v.strftime("%Y-%m-%d")
            else:
                s = str(v)
            it = QTableWidgetItem(s)
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if highlight and highlight(i, j):
                it.setBackground(QBrush(QColor(YELLOW)))
            t.setItem(i, j, it)
    t.resizeColumnsToContents()
    for j in range(t.columnCount()):
        if t.columnWidth(j) > max_col:
            t.setColumnWidth(j, max_col)
    t.resizeRowsToContents()
    return t


def fit_height(t: QTableWidget, extra=6) -> QTableWidget:
    """Make a small table exactly as tall as its rows (no inner scroll bar)."""
    h = t.horizontalHeader().height() + sum(t.rowHeight(i) for i in range(t.rowCount())) + extra
    t.setMinimumHeight(h)
    t.setMaximumHeight(h)
    t.setMinimumWidth(sum(t.columnWidth(j) for j in range(t.columnCount())) + 8)
    t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return t


def message(text: str, error=True) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.addStretch()
    lb = label(text, size=BASE_PT + 2, color=RED if error else INK)
    lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(lb)
    lay.addStretch()
    return w


def vbox(*widgets, stretch=None) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(6, 6, 6, 6)
    for i, x in enumerate(widgets):
        lay.addWidget(x, (stretch or {}).get(i, 0))
    return w


def hbox(*widgets, stretch=None) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for i, x in enumerate(widgets):
        lay.addWidget(x, (stretch or {}).get(i, 1))
    return w


def scroller(inner: QWidget) -> QScrollArea:
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setFrameShape(QFrame.Shape.NoFrame)
    s.setWidget(inner)
    return s


def mw(v, nd=0) -> str:
    return f"{v:,.{nd}f}"


# =============================================================================================== steps
# Each step: build(ctx) -> QWidget. It may raise D.StepError (clean message) or any exception (shown + logged).

def step1(ctx: Ctx) -> QWidget:
    p = D.psp_file(ctx.day)
    rs = D.raw_sheet(p)
    ctx.log(f"file: {D.rel(p)}")
    ctx.log("function: src.parse.psp_parser.read_sheets / find_sheet('MOP_E') (pandas + calamine)")
    ctx.log(f"sheets: {', '.join(f'{k} {v[0]}x{v[1]}' for k, v in rs['shapes'].items())}")
    raw, cells = rs["raw"], rs["cells"]
    for (r, c), (name, key) in cells.items():
        ctx.log(f"highlighted {name}: row {r + 1}, column {chr(65 + c)} = {raw.iat[r, c]}")
    t = QTableWidget(raw.shape[0], raw.shape[1])
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    t.setHorizontalHeaderLabels([chr(65 + i) for i in range(raw.shape[1])])
    t.setVerticalHeaderLabels([str(i + 1) for i in range(raw.shape[0])])
    first = {}
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw.iat[i, j]
            s = "" if pd.isna(v) else (f"{v:,.2f}" if isinstance(v, float) else str(v))
            it = QTableWidgetItem(s)
            if (i, j) in cells:
                it.setBackground(QBrush(QColor(YELLOW)))
                it.setForeground(QBrush(QColor("black")))
                f = QFont(it.font())
                f.setBold(True)
                it.setFont(f)
                first[(i, j)] = it
            else:
                it.setForeground(QBrush(QColor("#9a9a9a")))
            t.setItem(i, j, it)
    t.resizeColumnsToContents()
    for j in range(t.columnCount()):
        t.setColumnWidth(j, min(t.columnWidth(j), 360))
    t.resizeRowsToContents()
    pos = list(cells)
    jump = []
    for (r, c), (name, key) in cells.items():
        b = QPushButton(f"Go to: {name}")
        b.clicked.connect(lambda _=False, it=first[(r, c)]: t.scrollToItem(it, QAbstractItemView.ScrollHint.PositionAtCenter))
        jump.append(b)
    if pos:
        QTimer.singleShot(0, lambda: t.scrollToItem(first[pos[0]], QAbstractItemView.ScrollHint.PositionAtCenter))
    top = label(f"<b>{p.name}</b> &nbsp; sheets: {' · '.join(rs['sheets'])} &nbsp;|&nbsp; showing <b>MOP_E</b> "
                f"({raw.shape[0]} rows × {raw.shape[1]} columns). Yellow = the 3 cells the project uses; grey = ignored.")
    return vbox(top, hbox(*jump), t, stretch={2: 1})


def step2(ctx: Ctx) -> QWidget:
    p = D.psp_file(ctx.day)
    rs = D.raw_sheet(p)
    pr = D.parse_row(p)
    row = pr["row"]
    ctx.log(f"function: src.parse.psp_parser.parse_daily_summary({p.name}) -> 1 row x {pr['full'].shape[1]} columns")
    left = []
    key_val = {"energy_mu": row["energy_mu"], "solar_peak_mw": row["solar_peak_mw"], "evening_peak_mw": row["evening_peak_mw"]}
    ok = True
    for (r, c), (name, key) in rs["cells"].items():
        v = float(rs["raw"].iat[r, c])
        match = abs(v - key_val[key]) < 1e-6
        ok &= match
        left.append({"cell": f"{chr(65 + c)}{r + 1}", "what": name, "raw value": v, "-> field": key, "parser says": key_val[key],
                     "same?": "yes" if match else "NO"})
    ldf = pd.DataFrame(left)
    rdf = pd.DataFrame([row])
    ctx.log("check: every highlighted cell equals the parser output" if ok else "check FAILED: parser differs from the cell")
    t1 = fit_height(table(ldf, fmt={"raw value": ",.2f", "parser says": ",.2f"}))
    t2 = fit_height(table(rdf, fmt={"solar_peak_mw": ",.0f", "evening_peak_mw": ",.0f", "energy_mu": ",.2f"}))
    verdict = label(("Parser output equals the highlighted cells: yes" if ok else "WARNING: parser output differs from the cells"),
                    bold=True, color=TEAL if ok else RED, size=BASE_PT + 2)
    return vbox(heading("Highlighted cells in the Excel sheet"), t1, heading("Row the project's parser extracts"), t2, verdict,
                label("This one row is what the pipeline appends to <b>data/processed/daily_summary.parquet</b> for this date. "
                      f"The full parser row has {pr['full'].shape[1]} columns (regions, generation, shortages, ...); "
                      "only these three become forecast targets."), QWidget(), stretch={6: 1})


def step3(ctx: Ctx) -> QWidget:
    h = D.history()
    clean = h["clean"]
    ctx.log("files: data/processed/daily_summary.parquet -> src.data.quality.build_clean() -> daily_clean.parquet")
    ctx.log(f"build_clean() re-run now reproduces the stored daily_clean targets: {h['rebuild_matches']}")
    ctx.log(f"src.data.quality.find_outliers(): {len(h['outliers'])} (date, target) hits, z > 6 vs 15-day rolling median")
    cv = Canvas(ctx, 3, 1, height=420, sharex=True)
    dts = clean["date"]
    for ax, t in zip(cv.axes, D.TGT):
        ax.plot(dts, clean[t], color=BLUE, lw=1.6, label="daily value")
        imp = clean[clean[f"{t}_imputed"]]
        ax.scatter(imp["date"], imp[t], s=110, color=ORANGE, edgecolor="black", zorder=5, label="imputed (gap <= 2 days)")
        out = [(pd.Timestamp(d), v) for d, tt, v, z in h["outliers"] if tt == t]
        if out:
            ax.scatter([a for a, _ in out], [b for _, b in out], s=90, marker="X", color=RED, zorder=6, label="flagged outlier (kept)")
        ax.axvline(ctx.day, color="black", ls="--", lw=1.4)
        ax.set_ylabel(D.TGT_LABEL[t].replace(" (", "\n("))
    cv.axes[0].legend(loc="upper left", ncols=3)
    cv.axes[0].set_title(f"All {len(clean)} days, 3 targets (dashed line = {ctx.day.date()})")
    n_out_days = int(clean["is_outlier"].sum())
    txt = label(f"<b>{len(clean)}</b> calendar days since {clean['date'].min().date()} &nbsp;|&nbsp; "
                f"<b>{h['n_missing']}</b> days without a report &rarr; <b>{h['n_imputed']}</b> imputed by time interpolation "
                f"(only interior gaps of at most 2 days) &nbsp;|&nbsp; <b>{n_out_days}</b> days flagged as outliers "
                f"({len(h['outliers'])} day x target hits) &mdash; flagged, <u>not</u> removed.")
    return vbox(txt, cv, stretch={1: 1})


def step4(ctx: Ctx) -> QWidget:
    w = D.weather_day(ctx.day)
    fr = D.features_row(ctx.day)
    ctx.log("files: data/raw/weather/obs_<city>.parquet + obsx_<city>.parquet (11 cities), data/processed/region_weights.json")
    ctx.log("functions: src.data.weather.load_observed_all, regional (city -> region mean), src.features.build.derive_weather")
    ctx.log("check against data/processed/features.parquet row for the same date (see table)")
    cities = w["cities"][["city", "region", "tmax", "tmean", "rh_mean", "precip"]].copy()
    cities["region"] = cities["region"].str.upper()
    t1 = table(cities, fmt={c: ".1f" for c in ("tmax", "tmean", "rh_mean", "precip")})
    vars_ = ["tmax", "tmean", "cdd", "precip", "rh_mean"]
    regs = ["nr", "wr", "sr", "er", "ner"]
    rows = []
    for r in regs:
        rows.append({"region": r.upper(), "weight w": w["weights"][r], **{v: float(w["regional"][f"{v}_{r}"]) for v in vars_}})
    reg = pd.DataFrame(rows)
    manual = {v: sum(w["weights"][r] * float(w["regional"][f"{v}_{r}"]) for r in regs) for v in vars_}
    allin = {"region": "ALL-INDIA = sum(w*x)", "weight w": sum(w["weights"].values()), **manual}
    proj = {"region": "project (derive_weather)", "weight w": None, **{v: float(w["ai"][f"{v}_ai"]) for v in vars_}}
    feat = {"region": "features.parquet", "weight w": None, **{v: float(fr[f"{v}_ai"]) for v in vars_}}
    reg = pd.concat([reg, pd.DataFrame([allin, proj, feat])], ignore_index=True)
    diff = max(abs(manual[v] - float(fr[f"{v}_ai"])) for v in vars_)
    ctx.log(f"max |my weighted sum - features.parquet| = {diff:.2e}")
    t2 = table(reg, fmt={"weight w": ".4f", **{v: ".3f" for v in vars_}},
               highlight=lambda i, j: i >= len(regs) and j >= 2)
    t2 = fit_height(t2)
    names = {"tmax": "tmax_ai", "tmean": "tmean_ai", "cdd": "cdd_ai", "precip": "precip_ai", "rh_mean": "rh_mean_ai"}
    formula = label("<b>Formula</b> &nbsp; x_ai = &Sigma;<sub>r</sub> w<sub>r</sub> &middot; x<sub>r</sub> / &Sigma;<sub>r</sub> w<sub>r</sub>, "
                    "r = NR, WR, SR, ER, NER. &nbsp; Each region value is the mean of its cities (Delhi, Lucknow, Jaipur = NR; Ahmedabad, "
                    "Mumbai, Nagpur = WR; Chennai, Bengaluru, Hyderabad = SR; Kolkata = ER; Guwahati = NER). "
                    "w = each region's share of all-India energy met in 2023-24 (region_weights.json). "
                    "Columns map to " + ", ".join(names.values()) + f". Check vs the stored row: max difference {diff:.1e}.")
    t1 = fit_height(t1)
    t1.setMaximumWidth(560)
    left = vbox(heading(f"1. Raw values, {w['n_cities']} cities"), t1)
    right = vbox(heading("2. Cities -> regions -> weighted all-India"), t2)
    return scroller(vbox(hbox(left, right, stretch={0: 4, 1: 6}), formula))


def step5(ctx: Ctx) -> QWidget:
    c = D.calendar_day(ctx.day)
    ctx.log("function: src.features.calendar.calendar_features([date]); holidays from the `holidays` package (IN); "
            "festival dates from src/features/calendar.py FESTIVALS")
    s = c["cal"]
    items = [("weekday", c["weekday"])] + [(k, s[k]) for k in s.index]
    items.append(("national holiday today", c["holiday"] or "none"))
    items.append(("festival today", ", ".join(c["festivals_today"]) or "none"))
    df = pd.DataFrame(items, columns=["feature", "value"])
    df["value"] = df["value"].map(lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) and not float(v).is_integer() else
                                  (str(int(v)) if isinstance(v, (float, np.floating)) else str(v)))
    near = pd.DataFrame(c["festivals_near"], columns=["festival", "date", "days from today"])
    left = table(df)
    right = vbox(heading("Festivals within 30 days"), table(near) if len(near) else label("none"),
                 label("<b>days_to_festival / days_from_festival</b> are capped at 30; diwali_window covers Dhanteras..Bhai Dooj "
                       "(demand dips for several days). Everything here depends only on the date, so it is known for any future day: no leakage."))
    return hbox(left, right, stretch={0: 1, 1: 1})


def step6(ctx: Ctx) -> QWidget:
    clean = D.read_parquet("daily_clean.parquet")
    fr = D.features_row(ctx.day)
    d = clean[clean["date"] == ctx.day].iloc[0]
    feats = D.read_parquet("features.parquet")
    ctx.log("files: data/processed/daily_clean.parquet (demand) + weather (step 4) + calendar (step 5) -> features.parquet")
    ctx.log(f"features.parquet: {feats.shape[0]} rows x {feats.shape[1]} columns")
    dem = pd.DataFrame([{"date": ctx.day, "solar_peak": d["solar_peak"], "evening_peak": d["evening_peak"], "energy": d["energy"],
                         "is_imputed": bool(d["is_imputed"])}])
    wx = pd.DataFrame([{k: float(fr[k]) for k in ("tmax_ai", "tmin_ai", "cdd_ai", "precip_ai", "rh_mean_ai")}])
    cal = pd.DataFrame([{k: fr[k] for k in ("dow", "is_weekend", "month", "is_holiday", "is_festival", "days_to_festival")}]).astype(float)
    joined = feats[feats["date"] == ctx.day]
    top = vbox(label("<b>Demand row</b> (step 3)"), fit_height(table(dem, fmt={"solar_peak": ",.0f", "evening_peak": ",.0f", "energy": ",.1f"})),
               label("<b>Weather row</b> (step 4)"), fit_height(table(wx, fmt={c: ".2f" for c in wx.columns})),
               label("<b>Calendar row</b> (step 5)"), fit_height(table(cal, fmt={c: ".0f" for c in cal.columns})))
    mid = label(f"&darr; &nbsp; <b>merge on date = {ctx.day.date()}</b> &nbsp; &darr; &nbsp; features.parquet row: "
                f"<b>{joined.shape[1]} columns</b> (the full row is a join of these three plus lags, see next step)")
    g = D.feature_groups()
    if not g:
        raise D.StepError("No model config.yaml found in models/*/latest.txt")
    kinds = list(g)
    groups = ["weather", "calendar", "lag", "rolling", "other"]
    rows = []
    for gr in groups:
        r = {"group": gr}
        for k in kinds:
            r[k] = len(g[k][gr])
        ex = next((g[k][gr] for k in kinds if g[k][gr]), [])
        r["examples"] = ", ".join(ex[:6]) + (" ..." if len(ex) > 6 else "")
        rows.append(r)
    tot = {"group": "TOTAL", **{k: sum(len(v) for v in g[k].values()) for k in kinds}, "examples": ""}
    gdf = pd.DataFrame(rows + [tot])
    for k in kinds:
        ctx.log(f"{k}: {D.rel(D.model_configs()[k]['dir'] / 'config.yaml')} -> "
                + ", ".join(f"{gr} {len(g[k][gr])}" for gr in groups))
    note = label("Counts are the model inputs listed in each model's <b>config.yaml</b> (LightGBM: <i>features</i>; LSTM/TFT: encoder + decoder inputs, "
                 "deduplicated). A feature belongs to the group its name says; LightGBM also gets rolling means/std; the sequence models "
                 "see the history itself instead.")
    return scroller(vbox(top, mid, heading("Which model uses which feature group"), fit_height(table(gdf, max_col=700)), note))


def step7(ctx: Ctx) -> QWidget:
    tabs = QTabWidget()
    day = ctx.day
    # --- lag and rolling
    le = D.lag_example(day)
    ctx.log("function: src.features.build.make_tabular(daily, fc, [origin], horizons=[1]) -> ratios; raw days read from features.parquet")
    lag_w = QWidget()
    lay = QVBoxLayout(lag_w)
    combo = QComboBox()
    combo.addItems(D.TGT)
    combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    cv = Canvas(ctx, height=300)
    tb_holder = QVBoxLayout()
    note = label("")
    note.setTextFormat(Qt.TextFormat.RichText)
    lay.addWidget(hbox(label("Target:"), combo, stretch={0: 0, 1: 0}))
    lay.addWidget(cv, 2)
    lay.addLayout(tb_holder)
    lay.addWidget(note)

    def draw(_=None):
        tgt = combo.currentText()
        e = D.lag_example(day, tgt)
        ax = cv.axes
        ax.clear()
        s = e["series"]
        s = s[(s.index >= day - pd.Timedelta(days=60)) & (s.index <= day)]
        ax.plot(s.index, s.values, color=BLUE, label="actual")
        ax.axvspan(e["roll_days"][0], e["roll_days"][-1], color=TEAL, alpha=0.12, label="30-day window (roll30)")
        ax.axvspan(e["ref_days"][0], e["ref_days"][-1], color=ORANGE, alpha=0.2, label="last 7 days (ref)")
        for k, (dt_, v) in e["pts"].items():
            if k != "lag364":
                ax.scatter([dt_], [v], s=120, zorder=5, color=RED if k != "last" else "black")
                ax.annotate(k, (dt_, v), textcoords="offset points", xytext=(4, 10), fontsize=14, fontweight="bold")
        ax.axvline(day, color="black", ls="--")
        ax.set_title(f"{D.TGT_LABEL[tgt]}: only days up to the origin {day.date()} are used (lag364 = {e['pts']['lag364'][0].date()} is off the left edge)")
        ax.legend(loc="upper left", ncols=3)
        rows = []
        for k, (dt_, v) in e["pts"].items():
            rows.append({"feature": f"{tgt}_{k}", "day used": dt_, "value": v, "ratio to ref": e["tab"][f"{tgt}_{k}"],
                         "days before origin" : (day - dt_).days})
        rows.append({"feature": f"{tgt}_roll30", "day used": f"{e['roll_days'][0].date()} .. {day.date()}", "value": e["roll30"],
                     "ratio to ref": e["tab"][f"{tgt}_roll30"], "days before origin": 0})
        df = pd.DataFrame(rows)
        while tb_holder.count():
            tb_holder.takeAt(0).widget().deleteLater()
        tb_holder.addWidget(fit_height(table(df, fmt={"value": ",.1f", "ratio to ref": ".4f", "days before origin": "d"})))
        used = [v[0] for v in e["pts"].values()] + [e["roll_days"][-1]]
        note.setText(f"Forecast for {e['target_day'].date()} (h=1) made on {day.date()}. ref = mean of the last 7 days = {e['ref']:,.1f}; "
                     f"every feature is value / ref. Latest day used: {max(used).date()} &le; origin: {'yes' if max(used) <= day else 'NO'}.")
        cv.draw_idle()

    combo.currentIndexChanged.connect(draw)
    draw()
    tabs.addTab(lag_w, "Lag and rolling")
    # --- scaling
    sc = D.scaling_example(day)
    ctx.log(f"file: {D.rel(sc['scaler'])} (mean, std per decoder feature), clip to +-10")
    sw = QWidget()
    sl = QVBoxLayout(sw)
    c2 = QComboBox()
    c2.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    pick = [n for n in ("solar_peak_lag7", "evening_peak_lag7", "energy_lag7", "tmax_ai", "cdd_ai", "solar_peak_lag364") if n in sc["names"]]
    c2.addItems(pick)
    cv2 = Canvas(ctx, 1, 2, height=240)
    tb2 = QVBoxLayout()
    ft = label("")
    ft.setAlignment(Qt.AlignmentFlag.AlignTop)
    ft.setTextFormat(Qt.TextFormat.RichText)
    row2 = QHBoxLayout()
    row2.addLayout(tb2, 0)
    row2.addWidget(ft, 1)
    sl.addWidget(hbox(label("Feature:"), c2, stretch={0: 0, 1: 0}))
    sl.addWidget(cv2, 2)
    sl.addLayout(row2)

    def draw2(_=None):
        f = c2.currentText()
        i = sc["names"].index(f)
        mu, sd = float(sc["dec_mu"][i]), float(sc["dec_sd"][i])
        raw = sc["dec"][:, i]
        z = np.clip((raw - mu) / sd, -10, 10)
        a, b = cv2.axes
        a.clear()
        b.clear()
        x = np.arange(1, len(raw) + 1)
        a.bar(x, raw, color=BLUE)
        a.set_title("before scaling")
        a.set_xlabel("horizon day h")
        b.bar(x, z, color=TEAL)
        b.axhline(0, color="black", lw=1)
        b.set_title("after scaling")
        b.set_xlabel("horizon day h")
        df = pd.DataFrame({"h": x, "raw": raw, "(raw - mean) / std": z})
        while tb2.count():
            tb2.takeAt(0).widget().deleteLater()
        tb2.addWidget(fit_height(table(df, fmt={"h": "d", "raw": ".4f", "(raw - mean) / std": ".4f"})))
        ft.setText(f"<b>{f}</b>: z = (x - mean) / std with mean = {mu:.4f}, std = {sd:.4f} (fitted on the training window only; "
                   f"saved in scaler.npz). Example h=1: ({raw[0]:.4f} - {mu:.4f}) / {sd:.4f} = {z[0]:.3f}.")
        cv2.draw_idle()

    c2.currentIndexChanged.connect(draw2)
    draw2()
    tabs.addTab(sw, "Scaling")
    # --- windows
    ww = QWidget()
    wl = QVBoxLayout(ww)
    cv3 = Canvas(ctx, height=300)
    ax = cv3.axes
    s = le["series"]
    lo = day - pd.Timedelta(days=75)
    sub = s[(s.index >= lo) & (s.index <= day)]
    ax.plot(sub.index, sub.values, color=BLUE)
    e0, e1 = day - pd.Timedelta(days=sc["enc_days"] - 1), day
    ax.axvspan(e0, e1, color=TEAL, alpha=0.15)
    ax.axvspan(day + pd.Timedelta(days=1), day + pd.Timedelta(days=D.HORIZON), color=ORANGE, alpha=0.35)
    ymin, ymax = ax.get_ylim()
    ax.text(e0 + (e1 - e0) / 2, ymin + 0.05 * (ymax - ymin), f"{sc['enc_days']} days IN\n(encoder)", ha="center", fontsize=15, fontweight="bold")
    ax.text(day + pd.Timedelta(days=D.HORIZON / 2 + .5), ymin + 0.4 * (ymax - ymin), f"{D.HORIZON} days\nOUT", ha="center", fontsize=15, fontweight="bold")
    ax.axvline(day, color="black", ls="--")
    ax.set_title(f"Origin {day.date()}: window of {sc['enc_days']} past days in, {D.HORIZON} future days out")
    ax.set_xlim(lo, day + pd.Timedelta(days=D.HORIZON + 2))
    wl.addWidget(cv3, 2)
    wl.addWidget(label(f"Encoder tensor per origin: <b>{sc['enc_shape'][0]} days x {sc['enc_shape'][1]} features</b> (history of the 3 targets, "
                       f"weather, calendar). Decoder tensor: <b>{sc['dec_shape'][0]} days x {sc['dec_shape'][1]} known-future features</b> "
                       f"(forecast weather, calendar, lags). Output: 7 days x 3 targets x P10/P50/P90. "
                       "Built by src.features.build.make_sequences."))
    tabs.addTab(ww, "60 in, 7 out")
    # --- split
    sp = D.splits()
    ctx.log("files: models/lstm/<latest>/config.yaml (train_end), data/processed/fixed_split_info.json; src.features.build.valid_origins")
    cv4 = Canvas(ctx, height=300)
    ax = cv4.axes
    rowsd = []
    pr, fx = sp["production"], sp["fixed"]
    if pr:
        rowsd += [("Production model", "train", *pr["train"], BLUE), ("Production model", "validation", *pr["validation"], ORANGE)]
    if fx:
        rowsd += [("Fixed split (evaluation)", "fit", *fx["fit"], BLUE), ("Fixed split (evaluation)", "validation", *fx["validation"], ORANGE),
                  ("Fixed split (evaluation)", "test", *fx["test"], RED)]
    ys = {n: i for i, n in enumerate(dict.fromkeys(r[0] for r in rowsd))}
    import matplotlib.dates as mdates

    seen = set()
    for name, part, a, b, col in rowsd:
        ax.barh(ys[name], (b - a).days + 1, left=mdates.date2num(a), color=col, height=0.5, edgecolor="black",
                label=part if part not in seen else None)
        seen.add(part)
        mid = mdates.date2num(a) + ((b - a).days) / 2
        if (b - a).days < 250:       # narrow bar: label above it, in black
            ax.text(mid, ys[name] + 0.32, f"{part}\n{a.date()} .. {b.date()}", ha="center", va="bottom", color="black", fontsize=12,
                    fontweight="bold")
        else:
            ax.text(mid, ys[name], f"{part}\n{a.date()} .. {b.date()}", ha="center", va="center", color="white", fontsize=13,
                    fontweight="bold")
    ax.axvline(mdates.date2num(sp["last"]), color="black", ls="--")
    ax.set_ylim(-0.5, len(ys) - 0.1)
    ax.set_yticks(list(ys.values()), list(ys))
    ax.xaxis_date()
    ax.set_title(f"Train / validation / test on a timeline (dashed = last actual day {sp['last'].date()})")
    sw2 = QWidget()
    s2 = QVBoxLayout(sw2)
    s2.addWidget(cv4, 2)
    txt = ""
    if pr:
        txt += (f"Production: {pr['n_train']} training origins, then a 7-day gap (so no label overlaps), then {pr['n_val']} validation origins "
                f"(the last 15 %), used for early stopping and calibration; afterwards the model is refitted on train + validation "
                f"up to {sp['production'] and D.model_configs()['lstm']['cfg']['train_end']}. ")
    if fx:
        txt += "Fixed split: models fitted only on 'fit', tuned on 'validation', scored on the untouched 'test' period."
    s2.addWidget(label(txt))
    tabs.addTab(sw2, "Train / val / test")
    return tabs


# ---- step 8: training ---------------------------------------------------------------------------
class Worker(QObject):
    progress = pyqtSignal(dict)
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, tmp: Path, target: str):
        super().__init__()
        self.tmp, self.target = tmp, target
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        t0 = time.time()
        deadline = t0 + LIVE_BUDGET_S
        out = {}
        try:
            out["lgbm"] = D.live_lgbm(self.target, self.progress.emit, lambda: self._cancel, deadline, self.tmp)
            try:
                out["lstm"] = D.live_lstm(self.progress.emit, lambda: self._cancel, deadline, self.tmp)
            except D.StepError as e:
                out["lstm_msg"] = str(e)
            out["seconds"] = time.time() - t0
            self.done.emit(out)
        except TimeoutError as e:
            self.failed.emit(f"{e} after {time.time() - t0:.1f} s")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


class TrainStep(QWidget):
    def __init__(self, ctx: Ctx):
        super().__init__()
        self.ctx = ctx
        self.hist = D.histories()
        self.lg = D.lgbm_info()
        try:
            self.counts = D.param_counts()
        except Exception as e:  # noqa: BLE001
            self.counts = {"error": str(e)}
        ctx.log(f"replay: {', '.join(D.rel(v['src']) for v in self.hist.values())}")
        ctx.log(f"LightGBM: {D.rel(self.lg['src'])} (best_iters per target|quantile)")
        ctx.log("live: src.features.build.make_tabular + lightgbm (1 target, P10/P50/P90) and src.models.deep.DeepForecaster('lstm'), "
                f"seed 42, CPU, {LIVE_BUDGET_S} s budget, outputs only in {ctx.tmp}")
        self.thread = None
        self.worker = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.anim = QTimer(self)
        self.anim.timeout.connect(self._anim_step)
        self.t0 = 0.0
        self.epoch_shown = 0
        self.live_hist: list[dict] = []
        self.live_lgb: list[tuple] = []

        self.rb_replay, self.rb_live = QRadioButton("Replay (stored histories)"), QRadioButton("Live (retrain small models now)")
        self.rb_replay.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_replay)
        grp.addButton(self.rb_live)
        self.target = QComboBox()
        self.target.addItems(D.TGT)
        self.target.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.btn = QPushButton("Start")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.clock = label("timer 0.0 s", bold=True)
        self.status = label("")
        self.btn.clicked.connect(self.start)
        self.btn_cancel.clicked.connect(self.cancel)
        self.rb_replay.toggled.connect(self._mode_changed)
        bar = hbox(self.rb_replay, self.rb_live, label("LightGBM target:"), self.target, self.btn, self.btn_cancel, self.clock,
                   stretch={i: 0 for i in range(7)})
        self.cv = Canvas(ctx, 1, 2, height=300)
        self.info = QVBoxLayout()
        lay = QVBoxLayout(self)
        lay.addWidget(bar)
        lay.addWidget(self.status)
        lay.addWidget(self.cv, 3)
        lay.addLayout(self.info)
        self._draw_static()
        self._show_info()

    # ---- drawing
    def _draw_static(self):
        a, b = self.cv.axes
        for ax in (a, b):
            ax.clear()
        a.set_title("LSTM: pinball loss per epoch")
        b.set_title("TFT: pinball loss per epoch")
        for ax in (a, b):
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss (scaled)")
        self.cv.draw_idle()

    def _draw_replay(self, n: int | None = None):
        a, b = self.cv.axes
        for ax, kind in ((a, "lstm"), (b, "tft")):
            ax.clear()
            ax.set_title(f"{kind.upper()}: pinball loss per epoch (stored run)")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss (scaled)")
            if kind not in self.hist:
                ax.text(0.5, 0.5, f"no history file for {kind}", transform=ax.transAxes, ha="center", fontsize=15, color=RED)
                continue
            h = self.hist[kind]["hist"]
            k = len(h) if n is None else min(n, len(h))
            ax.set_xlim(0, len(h) + 1)
            ax.set_ylim(0, float(max(h["train_loss"].max(), h["val_loss"].max())) * 1.05)
            ax.plot(h["epoch"][:k], h["train_loss"][:k], color=BLUE, label="train")
            ax.plot(h["epoch"][:k], h["val_loss"][:k], color=RED, label="validation")
            best = int(h["val_loss"].idxmin())
            if k > best:
                ax.scatter([h["epoch"][best]], [h["val_loss"][best]], s=260, marker="*", color="gold", edgecolor="black", zorder=6,
                           label=f"best epoch {int(h['epoch'][best])}")
            ax.legend(loc="upper right")
        self.cv.draw_idle()

    def _draw_live(self):
        a, b = self.cv.axes
        a.clear()
        b.clear()
        a.set_title("LSTM live: loss per epoch")
        a.set_xlabel("epoch")
        if self.live_hist:
            ep = [h["epoch"] for h in self.live_hist]
            a.plot(ep, [h["val"] for h in self.live_hist], color=RED, marker="o", label="validation")
            if self.live_hist and "train" in self.live_hist[0]:
                a.plot(ep, [h["train"] for h in self.live_hist], color=BLUE, marker="o", label="train")
            a.legend()
        b.set_title(f"LightGBM live ({self.target.currentText()}): validation pinball")
        b.set_xlabel("boosting round")
        by_q = {}
        for q, it, v in self.live_lgb:
            by_q.setdefault(q, []).append((it, v))
        for q, pts in by_q.items():
            b.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"P{int(q * 100)}")
        if by_q:
            b.legend()
        self.cv.draw_idle()

    def _show_info(self):
        while self.info.count():
            w = self.info.takeAt(0).widget()
            if w:
                w.deleteLater()
        it = self.lg["best_iters"]
        rows = []
        for t in D.TGT:
            rows.append({"target": t, **{f"P{int(float(q) * 100)}": it.get(f"{t}|{q}") for q in (0.1, 0.5, 0.9)}})
        lg = pd.DataFrame(rows)
        cfgs = D.model_configs()
        pc = []
        for kind in ("lstm", "tft"):
            if kind in cfgs:
                c = cfgs[kind]["cfg"]
                p = c["params"]
                arch = ", ".join(f"{k}={p[k]}" for k in ("hidden", "layers", "d_model", "heads") if k in p)
                pc.append({"model": kind.upper(), "architecture (config.yaml)": arch, "epochs run": c.get("epochs"),
                           "parameters": self.counts.get(kind, "n/a")})
        pdf = pd.DataFrame(pc)
        left = vbox(label(f"<b>LightGBM</b>: boosting rounds kept per model (early stopping); total <b>{self.lg['total_trees']:,}</b> trees "
                          "= 3 targets x 3 quantiles"), fit_height(table(lg, fmt={c: "d" for c in lg.columns[1:]})))
        right = vbox(label("<b>Neural nets</b> (parameter count: built from config.yaml, loaded from model.pt)"),
                     fit_height(table(pdf, fmt={"parameters": ",", "epochs run": "d"}, max_col=380)))
        self.info.addWidget(hbox(left, right))

    # ---- mode / replay
    def _mode_changed(self):
        self.btn.setText("Play replay" if self.rb_replay.isChecked() else "Start live training")

    def start(self):
        self._stop_all()
        if self.rb_replay.isChecked():
            self.status.setText("Replaying the stored training histories epoch by epoch.")
            self.epoch_shown = 0
            self.anim.start(120)
            self.btn_cancel.setEnabled(True)
        else:
            self._start_live()

    def _anim_step(self):
        n = max((len(v["hist"]) for v in self.hist.values()), default=0)
        self.epoch_shown += 1
        self._draw_replay(self.epoch_shown)
        if self.epoch_shown >= n:
            self.anim.stop()
            self.btn_cancel.setEnabled(False)
            self.status.setText("Replay finished: stars mark the epoch with the lowest validation loss (the weights that are kept).")

    def show_replay_final(self):
        self._draw_replay(None)

    # ---- live
    def _start_live(self):
        self.live_hist, self.live_lgb = [], []
        self.status.setText("Live: building samples, then LightGBM (P10/P50/P90) and an 8-epoch LSTM on CPU, fixed seed 42 ...")
        self.worker = Worker(self.ctx.tmp, self.target.currentText())
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.done.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.t0 = time.time()
        self.timer.start(200)
        self.btn.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.ctx.keep += [self.thread, self.worker]
        self.thread.start()

    def _tick(self):
        el = time.time() - self.t0
        self.clock.setText(f"timer {el:4.1f} s / {LIVE_BUDGET_S} s")

    def _on_progress(self, d: dict):
        if d.get("stage") == "lgbm":
            self.live_lgb.append((d["q"], d["iter"], d["val"]))
            self.status.setText(f"LightGBM P{int(d['q'] * 100)}: round {d['iter']}/{d['of']}, validation pinball {d['val']:.4f}")
        elif d.get("stage") == "lstm":
            self.live_hist.append({"epoch": d["epoch"], "val": d["val"]})
            self.status.setText(f"LSTM: epoch {d['epoch']}/{d['of']}, validation loss {d['val']:.4f}")
        else:
            self.status.setText(d.get("text", ""))
        self._draw_live()

    def _finish_thread(self):
        self.timer.stop()
        self._tick()
        self.btn.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(5000)

    def _on_done(self, out: dict):
        self._finish_thread()
        msg = [f"Live training finished in {out['seconds']:.1f} s."]
        if "lgbm" in out:
            r = out["lgbm"]
            msg.append(f"LightGBM {self.target.currentText()}: {r['n_train']:,} train rows, {r['n_val']:,} validation rows, "
                       f"median-forecast MAPE {r['mape']:.2f} %.")
            self.ctx.log(f"live LightGBM: {r['seconds']:.1f} s, MAPE {r['mape']:.2f} % -> {D.rel(self.ctx.tmp / 'live_lgbm_validation.csv')}")
        if "lstm" in out:
            h = out["lstm"]["history"]
            self.live_hist = [{"epoch": int(r.epoch), "val": float(r.val_loss), "train": float(r.train_loss)} for r in h.itertuples()]
            msg.append(f"LSTM: {len(h)} epochs in {out['lstm']['seconds']:.1f} s, best validation loss {out['lstm']['best_val']:.4f}.")
            self.ctx.log(f"live LSTM: {out['lstm']['seconds']:.1f} s -> {D.rel(self.ctx.tmp / 'live_lstm_history.csv')}")
        if "lstm_msg" in out:
            msg.append(out["lstm_msg"])
        self.ctx.log(f"live total {out['seconds']:.1f} s (nothing written to models/)")
        self._draw_live()
        self.status.setText(" ".join(msg))
        self.live_seconds = out["seconds"]

    def _on_failed(self, why: str):
        self._finish_thread()
        self.ctx.log(f"live stopped: {why}; fell back to Replay")
        self.rb_replay.setChecked(True)
        self.status.setText(f"Live training stopped ({why}). Switched to Replay.")
        self.live_fell_back = True
        self.epoch_shown = 0
        self.anim.start(120)

    def cancel(self):
        if self.anim.isActive():
            self.anim.stop()
            self._draw_replay(None)
            self.btn_cancel.setEnabled(False)
            self.status.setText("Replay stopped.")
        if self.worker is not None and self.thread is not None and self.thread.isRunning():
            self.worker.cancel()
            self.status.setText("Cancelling ...")

    def _stop_all(self):
        self.anim.stop()

    def stop_everything(self):
        self.anim.stop()
        self.timer.stop()
        if self.worker is not None:
            self.worker.cancel()
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(3000)


def step8(ctx: Ctx) -> QWidget:
    w = TrainStep(ctx)
    w._draw_replay(None)
    ctx.keep.append(w)
    return w


# ---- step 9 --------------------------------------------------------------------------------------
def step9(ctx: Ctx) -> QWidget:
    e = D.ensemble_data(ctx.day)
    ctx.log("files: data/processed/backtest_processed.parquet (rolling out-of-sample forecasts), postprocess.json (blend weights), "
            f"latest_forecast.parquet, {len(D.week_ahead_pdfs())} Grid-India week-ahead PDFs in data/raw/forecast/ (parsed into forecast_daily.parquet)")
    ctx.log(f"ensemble members: {e['members']} (weight w on the first = LSTM): ensemble = w*LSTM + (1-w)*LightGBM per target and horizon")
    ctx.log(f"7-day forecast origin {ctx.day.date()} read from {e['forecast_src']}")
    w = QWidget()
    lay = QVBoxLayout(w)
    tgt = QComboBox()
    tgt.addItems(D.TGT)
    hc = QComboBox()
    hc.addItems([str(i) for i in range(1, 8)])
    for c in (tgt, hc):
        c.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    cv = Canvas(ctx, 1, 2, height=330)
    holder = QVBoxLayout()
    cap = label("")
    cap.setTextFormat(Qt.TextFormat.RichText)
    lay.addWidget(hbox(label("Target:"), tgt, label("backtest horizon h:"), hc, stretch={0: 0, 1: 0, 2: 0, 3: 0}))
    lay.addWidget(cv, 3)
    lay.addLayout(holder)
    lay.addWidget(cap)
    names = {e["members"][0]: ("LSTM", TEAL), e["members"][1]: ("LightGBM", ORANGE), "ensemble": ("Blend", BLUE)}

    def draw(_=None):
        t, h = tgt.currentText(), int(hc.currentText())
        a, b = cv.axes
        a.clear()
        b.clear()
        bt = e["backtest"]
        bt = bt[(bt["target"] == t) & (bt["h"] == h)]
        end = bt["date"].max()
        bt = bt[bt["date"] > end - pd.Timedelta(days=60)]
        for m, (nm, col) in names.items():
            s = bt[bt["model"] == m].sort_values("date")
            a.plot(s["date"], s["p50"], color=col, lw=3.2 if m == "ensemble" else 1.8, label=nm)
        act = bt[bt["model"] == "ensemble"].sort_values("date")
        a.scatter(act["date"], act["actual"], color="black", s=26, zorder=5, label="actual")
        a.set_title(f"Backtest, last 60 days, h={h} ({t})")
        a.legend(loc="best", ncols=2)
        a.tick_params(axis="x", rotation=30)
        mape = (act["p50"] - act["actual"]).abs().div(act["actual"]).mean() * 100
        # 7-day forecast
        fc = e["forecast"]
        info = ""
        if fc is None:
            b.text(0.5, 0.5, f"no stored forecast for origin {ctx.day.date()}", transform=b.transAxes, ha="center", fontsize=15, color=RED)
        else:
            f = fc[fc["target"] == t]
            ens = f[f["model"] == "ensemble"].sort_values("date")
            b.fill_between(ens["date"], ens["p10"], ens["p90"], color=BLUE, alpha=0.25, label="blend P10-P90")
            b.plot(ens["date"], ens["p50"], color=BLUE, lw=3.2, marker="o", label="blend P50")
            for m, col, nm in (("lstm", TEAL, "LSTM"), ("lgbm", ORANGE, "LightGBM")):
                s = f[f["model"] == m].sort_values("date")
                if len(s):
                    b.plot(s["date"], s["p50"], color=col, lw=1.4, ls=":", label=nm)
            g = f[f["model"] == "grid_india"].sort_values("date")
            if g["p50"].notna().any():
                b.plot(g["date"], g["p50"], color=RED, lw=2.4, marker="s", ms=8, label="Grid-India week-ahead")
            else:
                info = " (Grid-India publishes no week-ahead value for this target)"
            if f["actual"].notna().any():
                b.scatter(f["date"], f["actual"], color="black", zorder=6, label="actual")
            b.legend(loc="best", ncols=2, fontsize=12)
            b.tick_params(axis="x", rotation=30)
        b.set_title(f"7-day forecast issued {ctx.day.date()} ({t})")
        wt = e["weights"][t]
        wdf = pd.DataFrame([{"h": int(k), "weight on LSTM w": v, "weight on LightGBM 1-w": 1 - v} for k, v in wt.items()])
        wdf = wdf.set_index("h").T.reset_index().rename(columns={"index": f"{t}"})
        while holder.count():
            holder.takeAt(0).widget().deleteLater()
        holder.addWidget(fit_height(table(wdf, fmt={str(i): ".2f" for i in range(1, 8)} | {c: ".2f" for c in wdf.columns[1:]})))
        cap.setText(f"Learned blend weights for {t} (postprocess.json, refit from the last 180 days of out-of-sample errors). "
                    f"Backtest MAPE of the blend at h={h}: {mape:.2f} %.{info}")
        cv.draw_idle()

    tgt.currentIndexChanged.connect(draw)
    hc.currentIndexChanged.connect(draw)
    draw()
    return w


# ---- step 10 -------------------------------------------------------------------------------------
STEP_FILES = lambda day, tmp: {  # noqa: E731
    1: [D.psp_file_safe(day)],
    2: [D.PROCESSED / "daily_summary.parquet"],
    3: [D.PROCESSED / "daily_clean.parquet"],
    4: [D.PROCESSED / "region_weights.json", D.PROCESSED / "weather_city.parquet", D.PROCESSED / "weather_fc.parquet",
        D.PROCESSED / "weather_normals.json"],
    5: [D.ROOT / "src" / "features" / "calendar.py"],
    6: [D.PROCESSED / "features.parquet"],
    7: [D.MODELS / "lstm" / (D.MODELS / "lstm" / "latest.txt").read_text().strip() / "scaler.npz"]
       if (D.MODELS / "lstm" / "latest.txt").exists() else [],
    8: sorted(D.TRAINING.glob("*.csv")) + sorted(D.MODELS.glob("*/latest.txt")) + sorted(tmp.glob("*")),
    9: [D.PROCESSED / n for n in ("backtest_processed.parquet", "postprocess.json", "latest_forecast.parquet", "forecast_daily.parquet")],
}


def step10(ctx: Ctx) -> QWidget:
    files = {k: v for k, v in STEP_FILES(ctx.day, ctx.tmp).items()}
    df = D.outputs_table(ctx.day, files)
    ctx.log("table built by wt_data.outputs_table from the files each step reads or writes (live-training outputs are in the temp folder)")
    t = table(df, max_col=560)
    sel = {"p": D.ROOT / "data" / "processed"}

    def on_sel():
        r = t.currentRow()
        if r >= 0:
            sel["p"] = (D.ROOT / df.iloc[r]["path"]).parent if not Path(df.iloc[r]["path"]).is_absolute() else Path(df.iloc[r]["path"]).parent

    t.itemSelectionChanged.connect(on_sel)
    b1 = QPushButton("Open folder of selected file")
    b2 = QPushButton("Open data/processed")
    b3 = QPushButton("Open temp folder of live training")
    b1.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(sel["p"]))))
    b2.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(D.ROOT / "data" / "processed"))))
    b3.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ctx.tmp))))
    miss = int((df["size"] == "missing").sum())
    top = label(f"<b>{len(df)}</b> files, grouped by the step that reads or writes them; {miss} missing. "
                f"Select a row, then <i>Open folder</i>.")
    return vbox(top, t, hbox(b1, b2, b3, stretch={0: 0, 1: 0, 2: 0}), stretch={1: 1})


STEPS = [
    ("Raw Excel", step1,
     "Grid-India publishes one Excel report per day (the PSP report). The sheet MOP_E holds the national power-supply position. "
     "The project reads only three numbers from it.",
     "Everything downstream starts from these three cells; seeing them in the sheet shows the data is real and what we ignore."),
    ("Parse", step2,
     "The project's parser finds the cells by their labels (not fixed positions) and returns one tidy row: date, solar-hour peak, "
     "evening peak and energy met.",
     "Labels move between report versions; a parser that searches by label keeps 1,300+ files consistent. The row is checked against the cells."),
    ("History and cleaning", step3,
     "Every day's row is stacked into a daily history. Missing report days of up to 2 days are interpolated; unusual days are flagged.",
     "Models learn from a continuous calendar. Flagged days (holidays, storms) stay in, because they are real demand."),
    ("Weather", step4,
     "Raw weather of 11 cities is averaged into the 5 grid regions, then blended into all-India values using each region's share of demand.",
     "Demand follows weather where the electricity is used, so a demand-weighted all-India weather is a much better signal than a simple mean."),
    ("Calendar", step5,
     "Weekday, season, national holidays and festival distances are built from the date alone.",
     "Demand drops on Sundays and festivals (Diwali). These are known for any future date, so they help every forecast day."),
    ("Merge on date", step6,
     "Demand, weather and calendar rows are joined on the date into features.parquet. Each model picks its own feature groups.",
     "One row per day is the single source for training and prediction; the model configs show who uses what."),
    ("Lags, scaling, windows", step7,
     "Lag and rolling features summarise the past only. A scaler standardises each input. The sequence models see 60 days in and predict 7 days out.",
     "Past-only features prevent leakage; scaling makes inputs comparable; the date split tests the model on days it has never seen."),
    ("Train", step8,
     "LightGBM (trees) and an LSTM / TFT (neural nets) learn to predict the next 7 days with P10, P50 and P90.",
     "Early stopping keeps the epoch with the lowest validation loss, which guards against over-fitting; replay shows the stored production runs."),
    ("Ensemble and forecast", step9,
     "The LSTM and LightGBM forecasts are blended with weights learned from recent out-of-sample errors, then turned into a 7-day P10-P50-P90 forecast.",
     "A blend is more accurate than either model alone; the P10-P90 band shows the uncertainty; Grid-India's own week-ahead is the benchmark."),
    ("What we get", step10,
     "Every file the steps read or produce, with its size and shape.",
     "These files are the pipeline's memory: re-running a step rewrites only its own outputs."),
]


# =============================================================================================== window
class Main(QMainWindow):
    def __init__(self, day: pd.Timestamp, tmp: Path):
        super().__init__()
        self.ctx = Ctx(day, tmp)
        self.setWindowTitle(f"Power Peak walkthrough · {day.date()}")
        self.resize(1700, 1000)
        self.pt = BASE_PT
        self.built: dict[int, QWidget] = {}
        self.logs: dict[int, list[str]] = {}
        self.status: dict[int, str] = {}
        self.side = QListWidget()
        for i, (name, *_rest) in enumerate(STEPS, 1):
            self.side.addItem(f"{i}.  {name}")
        self.side.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.side.setFixedWidth(330)
        self.side.currentRowChanged.connect(self.show_step)
        self.stack = QWidget()
        self.stack_l = QVBoxLayout(self.stack)
        self.stack_l.setContentsMargins(0, 0, 0, 0)
        self.title = label("", bold=True, size=BASE_PT + 6)
        self.zoom_lbl = label("100 %", bold=True)
        bm, bp, bf = QPushButton("A -"), QPushButton("A +"), QPushButton("Full screen (F11)")
        bm.clicked.connect(lambda: self.set_zoom(self.ctx.zoom - 0.1))
        bp.clicked.connect(lambda: self.set_zoom(self.ctx.zoom + 0.1))
        bf.clicked.connect(self.toggle_full)
        for b in (bm, bp, bf):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        bprev, bnext = QPushButton("< Back"), QPushButton("Next >")
        bprev.clicked.connect(lambda: self.go(-1))
        bnext.clicked.connect(lambda: self.go(1))
        for b in (bprev, bnext):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        top = hbox(self.title, label("Zoom"), bm, self.zoom_lbl, bp, bprev, bnext, bf, stretch={0: 1, **{i: 0 for i in range(1, 8)}})
        self.what, self.why = QTextBrowser(), QTextBrowser()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for tb in (self.what, self.why):
            tb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        panels = QSplitter(Qt.Orientation.Horizontal)
        for ttl, w in (("What this step does", self.what), ("Why it matters", self.why), ("Where it came from (files / functions)", self.log)):
            panels.addWidget(vbox(label(ttl, bold=True), w))
        right = QSplitter(Qt.Orientation.Vertical)
        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(top)
        cl.addWidget(self.stack, 1)
        right.addWidget(centre)
        right.addWidget(panels)
        right.setStretchFactor(0, 4)
        right.setStretchFactor(1, 1)
        right.setSizes([700, 260])
        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(self.side)
        main.addWidget(right)
        main.setStretchFactor(1, 1)
        self.setCentralWidget(main)
        for keys, fn in ((Qt.Key.Key_Right, lambda: self.go(1)), (Qt.Key.Key_Left, lambda: self.go(-1)),
                         (Qt.Key.Key_PageDown, lambda: self.go(1)), (Qt.Key.Key_PageUp, lambda: self.go(-1)),
                         (Qt.Key.Key_F11, self.toggle_full), (Qt.Key.Key_Escape, self.leave_full)):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)
        for seq, d in (("Ctrl++", 0.1), ("Ctrl+=", 0.1), ("Ctrl+-", -0.1)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(lambda d=d: self.set_zoom(self.ctx.zoom + d))
        self.apply_style()

    # ---- style / zoom / navigation
    def apply_style(self):
        p = self.pt * self.ctx.zoom
        QApplication.instance().setStyleSheet(
            f"* {{ font-size: {p:.1f}pt; }} QMainWindow, QWidget {{ background: white; color: #111; }}"
            f"QListWidget {{ background: #f2f4f8; border: 0; }} QListWidget::item {{ padding: 10px 8px; }}"
            f"QListWidget::item:selected {{ background: {BLUE}; color: white; }}"
            f"QPushButton {{ background: #e8ecf5; border: 2px solid {BLUE}; border-radius: 6px; padding: 6px 12px; color: #111; }}"
            f"QPushButton:disabled {{ color: #888; border-color: #aaa; }} QPushButton:pressed {{ background: #c9d3ea; }}"
            f"QTableWidget {{ gridline-color: #c8c8c8; }} QHeaderView::section {{ background: #dde3f0; padding: 4px; border: 1px solid #b8c0d4; font-weight: bold; }}"
            f"QTextBrowser, QPlainTextEdit {{ border: 1px solid #b8c0d4; }} QPlainTextEdit {{ font-family: monospace; font-size: {max(p - 1, 12):.1f}pt; }}"
            f"QTabBar::tab {{ padding: 8px 16px; background: #e8ecf5; }} QTabBar::tab:selected {{ background: {BLUE}; color: white; }}")

    def set_zoom(self, z: float):
        self.ctx.zoom = round(min(2.0, max(0.7, z)), 2)
        self.zoom_lbl.setText(f"{int(self.ctx.zoom * 100)} %")
        self.apply_style()
        for cv in self.ctx.canvases:
            try:
                cv.fig.set_dpi(100 * self.ctx.zoom)
                cv.draw_idle()
            except RuntimeError:  # canvas already deleted
                pass

    def toggle_full(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def leave_full(self):
        if self.isFullScreen():
            self.showNormal()

    def go(self, d: int):
        r = min(len(STEPS) - 1, max(0, self.side.currentRow() + d))
        self.side.setCurrentRow(r)

    # ---- steps
    def build(self, i: int) -> QWidget:
        name, fn, what, why = STEPS[i]
        self.ctx.lines = []
        try:
            w = fn(self.ctx)
            self.status[i] = "ok"
        except D.StepError as e:
            self.ctx.log(f"MISSING: {e}")
            self.status[i] = f"missing: {e}"
            w = message(f"Step {i + 1} cannot run.\n\n{e}")
        except Exception as e:  # noqa: BLE001
            self.ctx.log("ERROR: " + "".join(traceback.format_exception_only(type(e), e)).strip())
            self.ctx.log(traceback.format_exc(limit=4))
            self.status[i] = f"error: {type(e).__name__}: {e}"
            w = message(f"Step {i + 1} failed:\n\n{type(e).__name__}: {e}\n\nThe other steps still work.")
        self.logs[i] = list(self.ctx.lines)
        return w

    def show_step(self, i: int):
        if i < 0:
            return
        if i not in self.built:
            self.built[i] = self.build(i)
        w = self.built[i]
        while self.stack_l.count():
            it = self.stack_l.takeAt(0)
            if it.widget():
                it.widget().hide()
        self.stack_l.addWidget(w)
        w.show()
        name, _, what, why = STEPS[i]
        self.title.setText(f"Step {i + 1} of {len(STEPS)}: {name}   ({self.ctx.day.date()})")
        self.what.setPlainText(what)
        self.why.setPlainText(why)
        self.log.setPlainText("\n".join(self.logs.get(i, [])))
        if i == 7 and hasattr(w, "show_replay_final"):
            pass

    def closeEvent(self, ev):
        w = self.built.get(7)
        if w is not None and hasattr(w, "stop_everything"):
            w.stop_everything()
        super().closeEvent(ev)


# =============================================================================================== selftest
def selftest(win: Main, live: bool) -> int:
    out = HERE / "screens"
    out.mkdir(exist_ok=True)
    app = QApplication.instance()
    bad = 0
    win.show()
    app.processEvents()
    for i, (name, *_r) in enumerate(STEPS):
        win.side.setCurrentRow(i)
        for _ in range(5):
            app.processEvents()
        st = win.status.get(i, "?")
        win.grab().save(str(out / f"step_{i + 1:02d}.png"))
        if st == "ok":
            print(f"OK       step {i + 1:2d} {name}")
        elif st.startswith("missing"):
            print(f"MISSING  step {i + 1:2d} {name}: {st[9:].splitlines()[0]}")
            bad += 1
        else:
            print(f"FAIL     step {i + 1:2d} {name}: {st}")
            bad += 1
    win.side.setCurrentRow(6)
    tabs = win.built.get(6)
    if isinstance(tabs, QTabWidget):
        for k in range(1, tabs.count()):
            tabs.setCurrentIndex(k)
            app.processEvents()
            win.grab().save(str(out / f"step_07_tab{k + 1}.png"))
        tabs.setCurrentIndex(0)
    if live:
        win.side.setCurrentRow(7)
        w = win.built[7]
        w.rb_live.setChecked(True)
        loop = QEventLoop()
        w.worker_done = False
        t0 = time.time()
        w.start()
        done = QTimer()
        done.setInterval(200)

        def chk():
            if (w.thread is not None and not w.thread.isRunning()) or time.time() - t0 > LIVE_BUDGET_S + 20:
                loop.quit()
        done.timeout.connect(chk)
        done.start()
        loop.exec()
        done.stop()
        app.processEvents()
        win.grab().save(str(out / "step_08_live.png"))
        print(f"LIVE     step 8 finished in {time.time() - t0:.1f} s; status: {w.status.text()}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="YYYY-MM-DD (default: latest day with actuals)")
    ap.add_argument("--selftest", action="store_true", help="build every step, save screenshots to walkthrough/screens/")
    ap.add_argument("--live", action="store_true", help="with --selftest: also run the live training once")
    a = ap.parse_args()
    try:
        latest = D.latest_day()
    except D.StepError as e:
        print(e)
        return 2
    day = pd.Timestamp(a.date) if a.date else latest
    clean = D.read_parquet("daily_clean.parquet")
    row = clean[clean["date"] == day]
    if row.empty or bool(row["is_missing_report"].iloc[0]) or bool(row["is_imputed"].iloc[0]):
        print(f"{day.date()} has no real actuals in daily_clean.parquet (range {clean['date'].min().date()} .. {latest.date()}).")
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="pp_walkthrough_"))
    app = QApplication(sys.argv[:1])
    f = QFont()
    f.setPointSize(BASE_PT)
    app.setFont(f)
    win = Main(day, tmp)
    if a.selftest:
        bad = selftest(win, a.live)
        win.close()
        print("SELFTEST " + ("OK: all 10 steps built" if not bad else f"{bad} step(s) not OK"))
        return 1 if bad else 0
    win.show()
    win.side.setCurrentRow(0)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
