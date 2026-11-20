"""Plotly figure builders for the app (all chart logic and every chart colour live here; pages
only display).

One template for every chart: transparent background, no border, light horizontal gridlines only,
no vertical gridlines, direct labels at line ends instead of legends where possible, a headline
that states the insight plus a one-line subtitle with units and source. Hovers: GW with 1 decimal,
MU with 0, dates as '11 Nov 2026'. Fixed series colours on every chart:

    Actual        #09090B solid (#FAFAFA in dark mode)
    Our model     #2563EB accent
    Grid-India    #D97706 dashed, diamond markers (identified by more than colour)
    P10–P90 band  #2563EB at 12 % opacity (light) / 26 % (dark, same visual weight)
    Alerts        #DC2626 only;  success #16A34A

When several of our models are compared, the best one keeps the accent and the others use a
quiet zinc ladder (OTHER_MODELS). Neutral fills and reference marks come from NEUTRAL.
`set_mode("dark")` switches all mode-dependent colours (the app calls it once per run).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from src import fmt
from src.config import TARGET_LABELS

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
ACCENT = "#2563EB"
ALERT = "#DC2626"
GOOD = "#16A34A"
GRID_INDIA = "#D97706"
PALETTE = {
    "light": {"text": "#09090B", "muted": "#71717A", "grid": "#EDEDEF", "axis": "#D4D4D8", "actual": "#09090B",
              "hover": "#FFFFFF", "faint": "#D4D4D8", "mid": "#A1A1AA", "strong": "#52525B", "band": 0.12,
              "models": ["#3F3F46", "#71717A", "#A1A1AA", "#D4D4D8", "#E4E4E7"]},
    "dark": {"text": "#FAFAFA", "muted": "#A1A1AA", "grid": "#232326", "axis": "#3F3F46", "actual": "#FAFAFA",
             "hover": "#18181B", "faint": "#3F3F46", "mid": "#71717A", "strong": "#A1A1AA", "band": 0.26,
             "models": ["#D4D4D8", "#A1A1AA", "#71717A", "#52525B", "#3F3F46"]},
}
MODE = "light"
MUTED = PALETTE["light"]["muted"]
GRID = PALETTE["light"]["grid"]
AXIS = PALETTE["light"]["axis"]
REF = "rgba(113,113,122,0.7)"            # reference lines (zero, 50 %, target)

COLORS = {"actual": PALETTE["light"]["actual"], "ours": ACCENT, "grid_india": GRID_INDIA}
NEUTRAL = {k: PALETTE["light"][k] for k in ("faint", "mid", "strong", "muted")}   # mode-aware greys
BAND_ALPHA = 0.12
_MODEL_ORDER = ["ensemble", "lgbm", "lstm", "tft", "seasonal_naive"]
# other models: a quiet zinc ladder, so the best model (accent) and Grid-India (amber) stand out
OTHER_MODELS = dict(zip(_MODEL_ORDER, PALETTE["light"]["models"]))
CATEGORICAL = [ACCENT, "#0F766E", "#7C3AED", "#BE185D", "#4D7C0F", "#0E7490", "#A16207", "#71717A"]
UP, DOWN = ACCENT, NEUTRAL["mid"]        # diverging: pushes up / down (red is reserved for alerts)
STATUS = {"good": GOOD, "warning": "#B45309", "critical": ALERT}
SEQ_BLUE = ["#EFF6FF", "#DBEAFE", "#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#1D4ED8"]
SOURCE = "Source: Grid-India (NLDC), Open-Meteo; own forecasts"
BACKTEST_SOURCE = "own backtest, Grid-India (NLDC) actuals"
HISTORY_SOURCE = "Grid-India (NLDC) daily reports"


def set_mode(mode: str) -> None:
    """'light' or 'dark': text, grid, neutrals, the model ladder and the 'Actual' series colour
    follow the viewer's theme."""
    global MODE, MUTED, GRID, AXIS, BAND_ALPHA, DOWN
    MODE = mode if mode in PALETTE else "light"
    p = PALETTE[MODE]
    MUTED, GRID, AXIS, BAND_ALPHA = p["muted"], p["grid"], p["axis"], p["band"]
    COLORS["actual"] = p["actual"]
    NEUTRAL.update({k: p[k] for k in NEUTRAL})
    OTHER_MODELS.update(zip(_MODEL_ORDER, p["models"]))
    DOWN = p["mid"]
    _register_template()


RECIPE_LABELS = {"nov26": "Nov 2026 recipe"}


class _Labels(dict):
    """Display names; variants like 'lstm_w@compact' -> 'LSTM (weighted, compact weather)'."""

    def get(self, key, default=None):
        if key in self:
            return self[key]
        if not isinstance(key, str):
            return default
        key, _, tag = key.partition("+")
        name, _, fs = key.partition("@")
        base, w = (name[:-2], True) if name.endswith("_w") else (name, False)
        if base not in self:
            return default if default is not None else key
        extra = [x for x in (("weighted" if w else ""), (f"{fs} weather" if fs else ""),
                             (RECIPE_LABELS.get(tag, f"recipe {tag}") if tag else "")) if x]
        return f"{self[base]} ({', '.join(extra)})" if extra else self[base]


MODEL_LABELS = _Labels({"lgbm": "LightGBM", "tft": "TFT", "lstm": "LSTM", "seasonal_naive": "Seasonal naive",
                        "grid_india": "Grid-India", "actual": "Actual", "ensemble": "Ensemble"})


def _register_template() -> None:
    p = PALETTE[MODE]
    pio.templates["pp"] = go.layout.Template(layout=dict(
        font=dict(family=FONT, size=12, color=p["text"]),
        colorway=CATEGORICAL,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(font=dict(family=FONT, size=12, color=p["text"]), bordercolor=p["axis"], bgcolor=p["hover"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None,
                    font=dict(size=12, color=p["muted"]), bgcolor="rgba(0,0,0,0)"),
        title=dict(x=0, xanchor="left", font=dict(size=15, color=p["text"])),
        xaxis=dict(showgrid=False, linecolor=p["axis"], zeroline=False, ticks="outside", tickcolor=p["axis"],
                   ticklen=4, tickfont=dict(color=p["muted"]), title=dict(font=dict(color=p["muted"], size=12))),
        yaxis=dict(showgrid=True, gridcolor=p["grid"], gridwidth=1, zeroline=False, showline=False,
                   tickfont=dict(color=p["muted"]), title=dict(font=dict(color=p["muted"], size=12))),
    ))
    pio.templates.default = "plotly_white+pp"


_register_template()


def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


BAND = rgba(ACCENT, BAND_ALPHA)


def scale(target: str) -> tuple[float, str]:
    """Display peaks in GW, energy in MU."""
    div, unit, _ = fmt.scale(target)
    return div, unit


def model_color(model: str, best: str | None = None) -> str:
    if model == "grid_india":
        return COLORS["grid_india"]
    if model == best or best is None:
        return COLORS["ours"]
    return OTHER_MODELS.get(model, OTHER_MODELS.get(model.partition("@")[0].removesuffix("_w"), MUTED))


NUM_FMT = {"GW": ",.1f", "MU": ",.0f", "%": ".1f", "°C": ".1f", "mm": ".1f", "km/h": ".0f", "MAPE (%)": ".2f"}


def _is_date_axis(fig) -> bool:
    for t in fig.data:
        x = getattr(t, "x", None)
        if x is not None and len(x):
            v = x[0]
            return isinstance(v, (pd.Timestamp, np.datetime64)) or (isinstance(v, str) and len(v) >= 10 and v[4] == "-")
    return False


def _jsonable_x(v):
    if isinstance(v, (np.datetime64, pd.Timestamp)):
        return pd.Timestamp(v).isoformat()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _direct_labels(fig, height: int) -> int:
    """Name each line at its last point, in the line's colour, nudging labels apart vertically so
    they never overlap; returns the right margin the labels need."""
    items = []
    for t in fig.data:
        if t.type != "scatter" or t.showlegend is False or not t.name or (t.fill or "") == "toself":
            continue
        x, y = list(t.x), np.asarray(t.y, dtype=float)
        ok = np.where(~np.isnan(y))[0]
        if not len(ok):
            continue
        i = ok[-1]
        color = (t.line.color if t.line and t.line.color else None) or (t.marker.color if t.marker else None) or MUTED
        items.append([_jsonable_x(x[i]), float(y[i]), t.name, color, t.meta == "label-left"])
    ys = [v for t in fig.data if t.type == "scatter" and t.y is not None
          for v in np.asarray(t.y, dtype=float) if not np.isnan(v)]
    if not items or not ys:
        fig.update_layout(showlegend=False)
        return 0
    span = (max(ys) - min(ys)) or 1.0
    gap = span * 16 / max(height - 110, 120)            # ~16 px between label baselines
    rights = sorted([it for it in items if not it[4]], key=lambda it: it[1])
    for k in range(1, len(rights)):                     # push up until separated
        if rights[k][1] - rights[k - 1][1] < gap:
            rights[k][1] = rights[k - 1][1] + gap
    for x, y, name, color, left in items:
        fig.add_annotation(x=x, y=y, text=name, showarrow=False, xanchor="right" if left else "left",
                           xshift=-8 if left else 8, yshift=-12 if left else 0, yanchor="middle",
                           font=dict(size=12, color=color), align="left")
    fig.update_layout(showlegend=False)
    right = [it for it in items if not it[4]]
    return min(max(7 * len(it[2]) for it in right) + 18, 230) if right else 10


DEFAULT_SOURCE = "Grid-India (NLDC), own models"


def with_source(subtitle: str | None, source: str | None = DEFAULT_SOURCE) -> str | None:
    """'GW, P50 with P10–P90 range' -> 'GW, P50 with P10–P90 range · Source: Grid-India (NLDC), own models'."""
    if not subtitle or not source or "source" in subtitle.lower():
        return subtitle
    return f"{subtitle.rstrip(' .;')} · Source: {source}"


def style(fig: go.Figure, title: str | None = None, height: int = 380, ytitle: str | None = None,
          legend: bool | str = True, subtitle: str | None = None, source: str | None = DEFAULT_SOURCE) -> go.Figure:
    """Shared template + headline (`title`, states the insight, 15px semibold), subtitle (units;
    the source is appended unless it already names one), height and y-axis title. legend: True
    (legend under the chart), False (none) or "direct" (labels at the end of each line). Hover
    values get the unit's precision (GW 1 decimal, MU 0); dates read '11 Nov 2026' in hovers and
    '11 Nov' / 'Nov 2026' on the axis."""
    p = PALETTE[MODE]
    t = None
    subtitle = with_source(subtitle, source)
    if title:
        t = dict(text=title, x=0, xanchor="left", font=dict(size=15, color=p["text"], weight=600))
        if subtitle:
            t["subtitle"] = dict(text=subtitle, font=dict(size=12, color=p["muted"]))
    right = 10
    if legend == "direct":
        right = _direct_labels(fig, height) or 10
    top = (66 if subtitle else 44) if title else 16
    leg_below = legend is True and len(fig.data) > 1
    fig.update_layout(template="plotly_white+pp", height=height + (36 if leg_below else 0),
                      showlegend=bool(legend) and legend != "direct",
                      title=t, margin=dict(l=8, r=right, t=top, b=8 + (36 if leg_below else 0)),
                      # legend pinned to the bottom edge of the figure, clear of tick labels and axis titles
                      legend=dict(orientation="h", yref="container", y=0, yanchor="bottom", x=0, xanchor="left"),
                      modebar=dict(bgcolor="rgba(0,0,0,0)", color=p["muted"], activecolor=p["text"]),
                      font=dict(family=FONT, color=p["text"]),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, linecolor=p["axis"], zeroline=False, tickfont=dict(color=p["muted"]))
    fig.update_yaxes(showgrid=True, gridcolor=p["grid"], zeroline=False, showline=False,
                     tickfont=dict(color=p["muted"]),
                     title=dict(text=ytitle, font=dict(color=p["muted"], size=12)) if ytitle else None)
    if _is_date_axis(fig):
        xs = [pd.Timestamp(v) for t in fig.data if t.x is not None for v in (t.x[0], t.x[-1]) if len(t.x)]
        span = (max(xs) - min(xs)).days if xs else 0
        fig.update_xaxes(hoverformat="%d %b %Y", tickformat="%d %b" if span <= 120 else "%b %Y")
        if xs and legend == "direct":       # keep the axis on the data; end labels sit in the margin
            pad = (max(xs) - min(xs)) * 0.01
            fig.update_xaxes(range=[min(xs) - pad, max(xs) + pad])
    num = NUM_FMT.get(ytitle or "")
    if num:
        unit = "" if ytitle in ("%", "MAPE (%)") else f" {ytitle}"
        pct = "%" if ytitle in ("%", "MAPE (%)") else ""
        for tr in fig.data:
            if tr.type in ("scatter", "bar") and tr.hovertemplate is None and tr.hoverinfo != "skip":
                tr.hovertemplate = f"%{{y:{num}}}{pct}{unit}<extra>{tr.name or ''}</extra>"
    return fig


def axis_title(fig: go.Figure, x: str | None = None, y: str | None = None) -> go.Figure:
    if x:
        fig.update_xaxes(title=dict(text=x, font=dict(color=MUTED, size=12)))
    if y:
        fig.update_yaxes(title=dict(text=y, font=dict(color=MUTED, size=12)))
    return fig


def empty(msg: str = "Not available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=MUTED), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=220, legend=False)


# ============================================================================ standard series
def actual_trace(x, y, unit: str, name: str = "Actual") -> go.Scatter:
    nf = NUM_FMT.get(unit, ",.1f")
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=COLORS["actual"], width=2),
                      hovertemplate=f"%{{y:{nf}}} {unit}<extra>{name}</extra>")


def band_traces(x, lo, hi, color: str = ACCENT, name: str = "P10–P90") -> go.Scatter:
    x = list(x)
    alpha = BAND_ALPHA                                        # 12 % light, 26 % dark: same visual weight
    return go.Scatter(x=x + x[::-1], y=list(hi) + list(lo)[::-1], fill="toself", mode="lines",
                      fillcolor=rgba(color, alpha), line=dict(width=0), hoverinfo="skip", name=name,
                      showlegend=False)


def grid_india_trace(x, y, unit: str, name: str = "Grid-India", line: bool = True) -> go.Scatter:
    nf = NUM_FMT.get(unit, ",.1f")
    return go.Scatter(x=x, y=y, name=name, mode="lines+markers" if line else "markers",
                      line=dict(color=COLORS["grid_india"], width=2, dash="dash"),
                      marker=dict(symbol="diamond", size=7, color=COLORS["grid_india"]),
                      hovertemplate=f"%{{y:{nf}}} {unit}<extra>{name}</extra>")


# ============================================================================ forecasts
def forecast_fan(hist: pd.Series, fc: pd.DataFrame, target: str, models: list[str] | None = None,
                 best: str | None = None, gi: pd.DataFrame | None = None, title: str | None = None,
                 height: int = 420, subtitle: str | None = None) -> go.Figure:
    """History (actuals) + P10–P90 band of the best model + P50 lines of the selected models,
    labelled directly at their ends; a thin rule marks where the forecast starts."""
    div, unit = scale(target)
    nf = NUM_FMT[unit]
    fig = go.Figure()
    fig.add_trace(actual_trace(hist.index, hist / div, unit).update(meta="label-left"))
    f = fc[fc["target"] == target]
    models = models or [m for m in f["model"].unique() if m != "grid_india"]
    if best in models:
        b = f[f["model"] == best].sort_values("date")
        fig.add_trace(band_traces(b["date"], b["p10"] / div, b["p90"] / div,
                                  name=f"{MODEL_LABELS.get(best, best)} P10–P90"))
    for m in models:
        g = f[f["model"] == m].sort_values("date")
        if g.empty:
            continue
        lab = MODEL_LABELS.get(m, m)
        fig.add_trace(go.Scatter(
            x=g["date"], y=g["p50"] / div, name=lab if len(models) > 1 else "Forecast", mode="lines+markers",
            line=dict(color=model_color(m, best), width=2.5 if m == best else 1.5), marker=dict(size=6 if m == best else 4),
            customdata=np.stack([g["p10"] / div, g["p90"] / div], -1),
            hovertemplate=f"%{{y:{nf}}} {unit} (P10 %{{customdata[0]:{nf}}} – P90 %{{customdata[1]:{nf}}})"
                          f"<extra>{lab}</extra>"))
    if gi is not None and len(gi):
        g = gi[gi["target"] == target].sort_values("date")
        if len(g):
            fig.add_trace(grid_india_trace(g["date"], g["p50"] / div, unit))
    if len(hist):
        fig.add_vline(x=hist.index.max(), line=dict(color=AXIS, width=1, dash="dot"))
        fig.add_annotation(x=hist.index.max(), y=1, yref="paper", text="forecast", showarrow=False,
                           xanchor="left", xshift=4, yanchor="top", font=dict(size=11, color=MUTED))
    return style(fig, title or TARGET_LABELS[target], height, ytitle=unit, legend="direct",
                 subtitle=subtitle or f"{unit}, P50 with P10–P90 range. {SOURCE}")


def hbar(labels: list[str], values: list[float], text: list[str] | None = None, colors=None,
         title: str | None = None, subtitle: str | None = None, source: str | None = DEFAULT_SOURCE,
         hover: str = "%{y}: %{x:+.1f}", xtitle: str | None = None, height: int | None = None) -> go.Figure:
    """Horizontal bars, first item on top. Value labels sit at the end of positive bars and just
    right of the zero line for negative ones, so they never collide with the category names."""
    labels, values = list(labels)[::-1], [float(v) for v in values][::-1]
    text = list(text)[::-1] if text is not None else [f"{v:+.1f}" for v in values]
    if colors is None:
        colors = [UP if v > 0 else DOWN for v in values]
    elif isinstance(colors, str):
        colors = [colors] * len(values)
    else:
        colors = list(colors)[::-1]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors, cornerradius=3),
                           hovertemplate=hover + "<extra></extra>"))
    for y, v, t in zip(labels, values, text):
        fig.add_annotation(x=max(v, 0), y=y, text=t, showarrow=False, xanchor="left", xshift=6,
                           font=dict(color=MUTED, size=11))
    m = max([abs(v) for v in values] or [1]) or 1
    chars = max([len(t) for t in text] or [4])
    lo = min(min(values), 0) * 1.04
    hi = max(max(values), 0) + m * (0.035 * chars) * (0.6 if min(values) >= 0 else 1)   # room for the labels
    if min(values) < 0:
        fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(range=[lo, hi], showgrid=False, zeroline=False, showticklabels=xtitle is not None,
                     title=dict(text=xtitle, font=dict(color=MUTED, size=12)) if xtitle else None)
    fig.update_layout(hovermode="closest", bargap=0.3)
    h = height or (84 if subtitle else 64) + 30 * len(labels)
    fig = style(fig, title, height=h, legend=False, subtitle=subtitle, source=source)
    fig.update_yaxes(showgrid=False, ticksuffix="  ")
    return fig


def drivers_bar(drivers: list[tuple[str, float]], unit: str, title: str | None = None,
                subtitle: str | None = None) -> go.Figure:
    """Signed effects of each driver: blue raises the forecast, grey lowers it."""
    return hbar([d[0] for d in drivers], [d[1] for d in drivers], title=title, subtitle=subtitle,
                hover=f"%{{y}}: %{{x:+.1f}} {unit}")


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "<br>".join(textwrap.wrap(text, width=width, break_long_words=False)) if len(text) > width else text


def fit(fig: go.Figure, half: bool = False) -> go.Figure:
    """Wrap the headline and subtitle to the chart's column (full or half width) and grow the
    top margin to match, so nothing is clipped at the right edge."""
    t = fig.layout.title
    if not t or not t.text:
        return fig
    tw, sw = (60, 80) if half else (120, 150)
    title = _wrap(t.text.replace("<br>", " "), tw)
    sub = t.subtitle.text if t.subtitle and t.subtitle.text else None
    sub = _wrap(sub.replace("<br>", " "), sw) if sub else None
    extra = 21 * title.count("<br>") + 18 * (sub.count("<br>") if sub else 0)
    extra += 12 if extra else 0
    if extra:
        fig.update_layout(margin=dict(t=fig.layout.margin.t + extra), height=fig.layout.height + extra)
    fig.update_layout(title=dict(text=title, subtitle=dict(text=sub) if sub else None))
    return fig


def gauge(p: float, title: str) -> go.Figure:
    pct = 100 * (p if p == p else 0)
    color = STATUS["critical"] if pct >= 50 else STATUS["warning"] if pct >= 20 else ACCENT
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct, number=dict(suffix=" %", font=dict(size=34), valueformat=".0f"),
        title=dict(text=title, font=dict(size=13)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED)),
                   bar=dict(color=color, thickness=0.35), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, 100], color=GRID)])))
    fig.update_layout(template="plotly_white+pp", height=210, margin=dict(l=36, r=36, t=40, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT))
    return fig


# ============================================================================ evaluation
def our_models(models) -> list[str]:
    """Our models (not Grid-India) in display order: ensemble, LightGBM, LSTM, TFT, seasonal naive."""
    rank = {"ensemble": 0, "lgbm": 1, "lstm": 2, "tft": 3, "seasonal_naive": 4}
    base = lambda m: m.partition("@")[0].removesuffix("_w")  # noqa: E731
    return sorted([m for m in set(models) if base(m) in rank], key=lambda m: (rank[base(m)], m))


def main_models(lb: pd.DataFrame) -> list[str]:
    """The best variant of each model family by overall MAPE (leaderboard rows), in display order:
    what horizon charts show so their end labels stay readable."""
    overall = lb[lb["model"] != "grid_india"].groupby("model")["MAPE"].mean()
    fam = overall.groupby(lambda m: m.partition("@")[0].removesuffix("_w")).idxmin()
    return our_models([m for f, m in fam.items() if f != "seasonal_naive"])


def horizon_lines(bt: pd.DataFrame, target: str, metric_fn, ytitle: str, title: str,
                  ref_line: float | None = None, best: str | None = None, models=None,
                  subtitle: str | None = None) -> go.Figure:
    fig = go.Figure()
    for m in models or our_models(bt["model"].unique()):
        g = bt[(bt["model"] == m) & (bt["target"] == target)]
        vals = g.groupby("h").apply(metric_fn)
        fig.add_trace(go.Scatter(x=vals.index, y=vals.values, name=MODEL_LABELS.get(m, m), mode="lines+markers",
                                 line=dict(color=model_color(m, best or "lgbm"), width=2.5 if m == best else 1.5),
                                 marker=dict(size=6 if m == best else 4),
                                 hovertemplate="%{y:.2f}<extra>" + MODEL_LABELS.get(m, m) + "</extra>"))
    if ref_line is not None:
        fig.add_hline(y=ref_line, line=dict(color=REF, dash="dot"), annotation_text=f"target {ref_line:g}",
                      annotation_font_color=MUTED, annotation_position="top left")
    fig.update_xaxes(dtick=1)
    axis_title(fig, x="days ahead")
    return style(fig, title, 360, ytitle, legend="direct", subtitle=subtitle, source=BACKTEST_SOURCE)


def reliability_plot(rel: pd.DataFrame, title: str, subtitle: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=REF, dash="dot"),
                             name="Perfect calibration", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rel["mean_predicted"], y=rel["observed_freq"], mode="lines+markers+text",
                             text=[f"n={n}" for n in rel["n"]], textposition="top left",
                             textfont=dict(color=MUTED, size=10), marker=dict(size=10, color=ACCENT),
                             line=dict(color=ACCENT, width=2), name="Model",
                             hovertemplate="predicted %{x:.2f}, observed %{y:.2f}<extra></extra>"))
    fig.update_xaxes(range=[-0.02, 1.02])
    fig.update_yaxes(range=[-0.02, 1.02])
    axis_title(fig, x="predicted probability")
    return style(fig, title, 380, "observed frequency", legend=False, subtitle=subtitle, source=BACKTEST_SOURCE)


def training_curve(h: pd.DataFrame, title: str, subtitle: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["train_loss"], name="Training", line=dict(color=MUTED, width=1.5)))
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_loss"], name="Validation", line=dict(color=ACCENT, width=2)))
    best = h["val_loss"].idxmin()
    fig.add_vline(x=h.loc[best, "epoch"], line=dict(color=REF, dash="dot"),
                  annotation_text="best epoch", annotation_font_color=MUTED)
    axis_title(fig, x="epoch")
    title = f"{title}: best at epoch {int(h.loc[best, 'epoch'])} of {int(h['epoch'].max())}"
    return style(fig, title, 300, "pinball loss", legend="direct", subtitle=subtitle, source="own training runs")


def month_year_heatmap(s: pd.Series, title: str, unit: str, subtitle: str | None = None) -> go.Figure:
    t = s.groupby([s.index.year, s.index.month]).mean().unstack()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    t = t.reindex(columns=range(1, 13))
    dec = 1 if unit == "GW" else 0
    fig = go.Figure(go.Heatmap(z=t.values, x=months, y=[str(y) for y in t.index],
                               colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
                               xgap=2, ygap=2, colorbar=dict(title=unit, tickfont=dict(color=MUTED)),
                               text=np.where(np.isnan(t.values), "",
                                             np.vectorize(lambda v: f"{v:,.{dec}f}")(np.nan_to_num(t.values))),
                               texttemplate="%{text}", textfont=dict(size=10),
                               hovertemplate="%{y} %{x}: %{z:,." + str(dec) + "f} " + unit + "<extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(showgrid=False)
    return style(fig, title, 300, legend=False, subtitle=subtitle, source=HISTORY_SOURCE)


# ============================================================================ weather impact
def mape_bars(t: pd.DataFrame, models: list[str], order: list[str], title: str, best: str | None = None,
              subtitle: str | None = None) -> go.Figure:
    """Grouped bars: x = regime/season label, one bar per model; hover shows the day count."""
    fig = go.Figure()
    for m in models:
        g = t[t["model"] == m].set_index("label").reindex(order)
        if g["MAPE"].isna().all():
            continue
        kw = dict(marker=dict(color=model_color(m, best), cornerradius=3))
        if m == "grid_india":
            kw = dict(marker=dict(color=COLORS["grid_india"], pattern=dict(shape="/"), cornerradius=3))
        fig.add_trace(go.Bar(x=order, y=g["MAPE"], name=MODEL_LABELS.get(m, m), customdata=g["days"],
                             hovertemplate="%{x}: %{y:.2f} % (%{customdata} days)<extra>" + MODEL_LABELS.get(m, m)
                                           + "</extra>", **kw))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05, hovermode="closest")
    return style(fig, title, 400, "MAPE (%)", subtitle=subtitle, source=BACKTEST_SOURCE)


def coverage_before_after(cov: pd.DataFrame, target: str, model: str, title: str,
                          subtitle: str | None = None) -> go.Figure:
    g = cov[(cov["target"] == target) & (cov["model"] == model)].sort_values("h")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["h"], y=g["before"], name="Before", mode="lines+markers",
                             line=dict(color=MUTED, width=1.5, dash="dash"), marker=dict(size=6, symbol="circle-open"),
                             hovertemplate="%{y:.1f} %<extra>Before</extra>"))
    fig.add_trace(go.Scatter(x=g["h"], y=g["after"], name="After", mode="lines+markers",
                             line=dict(color=ACCENT, width=2.5), marker=dict(size=6),
                             hovertemplate="%{y:.1f} %<extra>After</extra>"))
    fig.add_hline(y=80, line=dict(color=REF, dash="dot"), annotation_text="target 80 %", annotation_font_color=MUTED,
                  annotation_position="bottom left")
    fig.update_xaxes(dtick=1)
    axis_title(fig, x="days ahead")
    return style(fig, title, 340, "% inside P10–P90", legend="direct", subtitle=subtitle, source=BACKTEST_SOURCE)


WX_PANELS = [  # (column, label, unit, kind)
    ("tmax_ai", "Max temperature, all India", "°C", "line"),
    ("tmin_nr", "Night temperature, North India", "°C", "line"),
    ("precip_ai", "Rain, all India (demand-weighted)", "mm", "bar"),
    ("cloud_day_ai", "Daytime cloud cover, all India", "%", "bar"),
]


def weather_week(wx: pd.DataFrame, panels=WX_PANELS, height: int = 230) -> list[go.Figure]:
    """Small multiples of the 7-day weather forecast the demand forecast was made with."""
    figs = []
    for col, lab, unit, kind in panels:
        if col not in wx:
            continue
        x = wx["date"]
        anom = f"{col}_anom"
        hover = f"%{{x|%a %d %b %Y}}: %{{y:.1f}} {unit}"
        if anom in wx:
            hover += " (%{customdata:+.1f} vs normal)"
        tr = (go.Bar(x=x, y=wx[col], marker=dict(color=ACCENT, cornerradius=3)) if kind == "bar" else
              go.Scatter(x=x, y=wx[col], mode="lines+markers", line=dict(color=ACCENT, width=2), marker=dict(size=7)))
        tr.update(customdata=wx[anom] if anom in wx else None, hovertemplate=hover + "<extra></extra>", name=lab)
        fig = go.Figure(tr)
        fig.update_layout(hovermode="closest")
        v = wx[col].astype(float)
        if v.notna().any():
            top = wx.loc[v.idxmax(), "date"]
            dec = 0 if unit in ("%", "km/h") else 1
            ttl = (f"{lab}: highest on {pd.Timestamp(top):%a %d %b}, {v.max():.{dec}f} {unit}" if kind == "line"
                   else f"{lab}: {v.max():.{dec}f} {unit} at most, on {pd.Timestamp(top):%a %d %b}"
                   if v.max() > 0 else f"{lab}: none expected")
        else:
            ttl = lab
        fig = style(fig, ttl, height, unit, legend=False, subtitle=f"{unit}, daily forecast", source="Open-Meteo")
        fig.update_xaxes(tickformat="%a %d")
        figs.append(fig)
    return figs
