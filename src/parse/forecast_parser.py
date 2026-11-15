"""Parser for Grid-India "Week ahead all India demand forecast" PDFs (benchmark forecasts).

The PDF is a vector chart (not an image), so pdfplumber can read it exactly:
  * title: "26-Oct-2026 to 01-Nov-2026"
  * chart page: y tick labels (MW), one vertical gridline per midnight, a red line = point
    forecast, (newer files) a grey filled polygon = uncertainty band, and text labels on each
    day's peaks.

Three chart generations exist (verified on real files):
  * Oct-2026:           text labels "234-238"  -> range in GW (low_mw=234000, high_mw=238000)
  * Jan-2026:           text labels "246699"   -> point in MW (low_mw = high_mw)
  * Nov-2024..May-2025: labels drawn as glyph outlines (not text) -> no labels; daily peaks
                        are derived from the curve and marked label_kind="curve".

There are TWO labels per day, not one: the solar-hour (daytime, ~09:45) peak and the
non-solar-hour (evening, ~18:30) peak — the same split as PSP report section I.
Every label is cross-checked against the max of the red curve in its own window
(`curve_ok`), so a mis-read label is flagged instead of silently trusted.

Calibration comes from the chart's own tick labels and gridlines; nothing about page
coordinates is hard-coded. If the expected structure is missing we raise, rather than guess.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pdfplumber

log = logging.getLogger(__name__)

_TITLE = re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{4})\s*to\s*(\d{1,2}-[A-Za-z]{3}-\d{4})")
_RANGE = re.compile(r"^(\d{2,3})-(\d{2,3})$")      # "234-238" (GW)
_NUMBER = re.compile(r"^\d{5,6}$")                 # "240000" tick, or "246699" point label
RED = (1.0, 0.0, 0.0)
SOLAR_HOURS = (6.0, 18.0)  # label inside this window -> solar-hour peak, else non-solar-hour


class ForecastParseError(ValueError):
    pass


@dataclass
class _Chart:
    start: dt.date
    n_days: int
    y_to_mw: Callable
    x_to_h: Callable
    words: list[dict]
    tick_x1: float
    red: np.ndarray | None
    band: np.ndarray | None


def _rgb(color) -> tuple:
    """pdfplumber colours may be a scalar grey, (g,), (r,g,b) or (c,m,y,k)."""
    if color is None:
        return ()
    if isinstance(color, (int, float)):
        return (float(color),) * 3
    color = tuple(float(c) for c in color)
    if len(color) == 1:
        return color * 3
    if len(color) == 4:
        c, m, y, k = color
        return tuple((1 - v) * (1 - k) for v in (c, m, y))
    return color


_FOOTER_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")  # PowerPoint footer, M/D/YYYY


def _resolve_period(text: str, filename: str, path: Path) -> tuple[dt.date, int | None]:
    """Forecast start date and (if trustworthy) number of days.

    Titles contain typos in the archive (e.g. PDF text "04-Dec-2025 to 10-Jan-2025", file
    name "04-Jan-2024 to 10-Jan-2025" for the week 04..10-Jan-2025), so all candidates are
    collected and the one consistent with the footer 'prepared on' date wins.
    """
    pairs = []
    for src in (text, filename):
        m = _TITLE.search(src)
        if m:
            pairs.append(tuple(pd.to_datetime(x, format="%d-%b-%Y").date() for x in m.groups()))
    if not pairs:
        raise ForecastParseError(f"{path.name}: period 'DD-Mon-YYYY to DD-Mon-YYYY' not found")
    good = [(s, e) for s, e in pairs if 5 <= (e - s).days <= 8]
    candidates = {s for s, _ in pairs} | {e - dt.timedelta(days=6) for _, e in pairs}
    prepared = [dt.date(int(y), int(mo), int(d)) for mo, d, y in _FOOTER_DATE.findall(text)
                if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31]
    if prepared:
        ok = sorted(c for c in candidates if 0 <= (c - prepared[0]).days <= 10)
        if ok:
            start = (good[0][0] if good and good[0][0] in ok else ok[0])
            n = (good[0][1] - good[0][0]).days + 1 if good and good[0][0] == start else None
            if not good or good[0][0] != start:
                log.warning("%s: inconsistent title dates; using start %s (prepared %s)",
                            path.name, start, prepared[0])
            return start, n
    if good and all(p == good[0] for p in good):
        return good[0][0], (good[0][1] - good[0][0]).days + 1
    raise ForecastParseError(f"{path.name}: inconsistent forecast dates {pairs}; not guessing")


def _load_chart(path: Path) -> _Chart:
    with pdfplumber.open(path) as pdf:
        text = " ".join((p.extract_text() or "") for p in pdf.pages)
        start, n_days = _resolve_period(text, path.name, path)

        page = max(pdf.pages, key=lambda p: len(p.curves))
        if not page.curves:
            raise ForecastParseError(f"{path.name}: no vector chart found (scanned/image PDF?)")
        words = page.extract_words()

        # ---- y axis: numeric tick labels in the left-most column; fit value = a*y + b
        nums = [w for w in words if _NUMBER.match(w["text"])]
        if not nums:
            raise ForecastParseError(f"{path.name}: no y-axis tick labels")
        tick_x1 = min(w["x1"] for w in nums)
        ticks = [w for w in nums if w["x1"] - tick_x1 < 3]  # right-aligned column
        if len(ticks) < 3:
            raise ForecastParseError(f"{path.name}: fewer than 3 y-axis tick labels")
        ty = np.array([(w["top"] + w["bottom"]) / 2 for w in ticks])
        tv = np.array([float(w["text"]) for w in ticks])
        # Text centres sit ~1-2pt off the gridline they label (=~1 GW bias), so snap each tick
        # to the nearest horizontal gridline when one is close.
        hlines = np.array([l["top"] for l in page.lines
                           if abs(l["top"] - l["bottom"]) < 0.5 and l["x1"] - l["x0"] > 200])
        if len(hlines):
            near = np.abs(hlines[None, :] - ty[:, None])
            snap = near.min(axis=1) < 0.4 * np.median(np.diff(np.sort(ty)))
            ty = np.where(snap, hlines[near.argmin(axis=1)], ty)
        a, b = np.polyfit(ty, tv, 1)
        if np.abs(a * ty + b - tv).max() > 0.01 * np.ptp(tv):
            raise ForecastParseError(f"{path.name}: y-axis ticks are not linear")

        reds = [c for c in page.curves if np.allclose(_rgb(c.get("stroking_color")), RED, atol=0.01)
                if len(c["pts"]) > 50]
        red = np.array(max(reds, key=lambda c: len(c["pts"]))["pts"], float) if reds else None
        fills = [c for c in page.curves if c.get("fill") and len(c["pts"]) > 100]
        band = np.array(max(fills, key=lambda c: len(c["pts"]))["pts"], float) if fills else None

        # ---- x axis: evenly spaced vertical gridlines, one per midnight. Newer charts draw
        # both outer midnights, older ones only the interior ones, so anchor day 0 at the
        # gridline where the forecast curve starts.
        min_h = 0.5 * np.ptp(ty)
        xs = sorted({round(l["x0"], 1) for l in page.lines
                     if abs(l["x0"] - l["x1"]) < 0.5 and l["bottom"] - l["top"] > min_h})
        grid = _longest_even_run(xs)
        if grid is None or len(grid) < 3:
            raise ForecastParseError(f"{path.name}: day gridlines not found (x={xs})")
        day_w = float(np.mean(np.diff(grid)))
        x0 = grid[0]
        if red is not None:
            x0 = min(grid, key=lambda g: abs(g - red[:, 0].min()))
            if abs(x0 - red[:, 0].min()) > 0.1 * day_w:
                raise ForecastParseError(f"{path.name}: forecast curve does not start at midnight")
            curve_days = round((red[:, 0].max() - x0) / day_w)
            if n_days is None:
                n_days = curve_days
            elif abs(curve_days - n_days) > 1:
                raise ForecastParseError(
                    f"{path.name}: title says {n_days} days but curve spans {curve_days}")
        if n_days is None:
            raise ForecastParseError(f"{path.name}: cannot determine number of forecast days")

    return _Chart(start=start, n_days=n_days,
                  y_to_mw=lambda y: a * np.asarray(y, float) + b,
                  x_to_h=lambda x: (np.asarray(x, float) - x0) / day_w * 24.0,
                  words=words, tick_x1=tick_x1, red=red, band=band)


def _longest_even_run(xs: list[float], tol: float = 0.02) -> list[float] | None:
    """Longest run of consecutive values with (nearly) equal spacing (skips frame edges)."""
    best = None
    for i in range(len(xs) - 1):
        gap = xs[i + 1] - xs[i]
        if gap < 5:
            continue
        run = [xs[i], xs[i + 1]]
        for x in xs[i + 2:]:
            if abs((x - run[-1]) - gap) < tol * gap:
                run.append(x)
            elif x - run[-1] > gap:
                break
        if best is None or len(run) > len(best):
            best = run
    return best


def parse_forecast_labels(path: str | Path) -> pd.DataFrame:
    """Every peak label on the chart (normally 2 per day) in MW, with its day, peak type,
    and a cross-check against the red curve."""
    path = Path(path)
    ch = _load_chart(path)
    rows = []
    for w in ch.words:
        if (m := _RANGE.match(w["text"])):
            low, high, kind = int(m[1]) * 1000, int(m[2]) * 1000, "range"
        elif _NUMBER.match(w["text"]) and w["x0"] > ch.tick_x1 + 10:
            low = high = int(w["text"])
            kind = "point"
        else:
            continue
        if low > high:
            raise ForecastParseError(f"{path.name}: bad range label {w['text']}")
        hours = float(ch.x_to_h((w["x0"] + w["x1"]) / 2))
        day = int(hours // 24)
        if not 0 <= day < ch.n_days:
            raise ForecastParseError(f"{path.name}: label {w['text']} outside the forecast days")
        label_h = hours - 24 * day
        rec = {"date": pd.Timestamp(ch.start) + pd.Timedelta(days=day), "low_mw": low,
               "high_mw": high, "label_kind": kind, "label_hour": round(label_h, 2)}
        if ch.red is not None:
            # Labels are drawn at the peak they annotate: compare with the curve max within
            # +-1h of the label's x position.
            h = ch.x_to_h(ch.red[:, 0]) - 24 * day
            mask = np.abs(h - label_h) <= 1.0
            if mask.any():
                vals = np.where(mask, ch.y_to_mw(ch.red[:, 1]), -np.inf)
                i = int(np.argmax(vals))
                rec["curve_peak_mw"] = round(float(vals[i]))
                rec["curve_peak_hour"] = round(float(h[i]), 2)
                slack = 1000 if kind == "range" else 1500  # GW rounding / drawing precision
                rec["curve_ok"] = low - slack <= rec["curve_peak_mw"] <= high + slack
        rows.append(rec)

    if not rows:
        raise ForecastParseError(f"{path.name}: no peak labels found on the chart")
    df = pd.DataFrame(rows).sort_values(["date", "label_hour"]).reset_index(drop=True)
    # Two labels per day: earlier = solar-hour peak, later = non-solar-hour (evening) peak.
    # (A fixed clock cut-off fails: winter evening peaks are labelled before 18:00.)
    n = df.groupby("date")["label_hour"].transform("size")
    order = df.groupby("date").cumcount()
    by_clock = np.where((df["label_hour"] >= SOLAR_HOURS[0]) & (df["label_hour"] < SOLAR_HOURS[1]),
                        "solar_hr", "non_solar_hr")
    df["peak_type"] = np.where(n == 2, np.where(order == 0, "solar_hr", "non_solar_hr"), by_clock)
    df.insert(0, "forecast_start", pd.Timestamp(ch.start))
    return df


def parse_forecast_pdf(path: str | Path) -> pd.DataFrame:
    """One row per forecast day (normally 7): date, low_mw, high_mw of the day's *peak*
    (the higher of its labels), plus the solar-hour and non-solar-hour values.

    `curve_ok` is False if any label disagrees with the drawn forecast curve; treat such
    rows as suspect rather than trusting them."""
    try:
        labels = parse_forecast_labels(path)
    except ForecastParseError as e:
        if "no peak labels" not in str(e):
            raise
        return _daily_peaks_from_curve(path)
    out = []
    for date, g in labels.groupby("date"):
        top = g.loc[g["high_mw"].idxmax()]
        rec = {"forecast_start": g["forecast_start"].iloc[0], "date": date,
               "low_mw": top["low_mw"], "high_mw": top["high_mw"]}
        for kind in ("solar_hr", "non_solar_hr"):
            k = g[g["peak_type"] == kind]
            rec[f"{kind}_low_mw"] = k["low_mw"].iloc[0] if len(k) else np.nan
            rec[f"{kind}_high_mw"] = k["high_mw"].iloc[0] if len(k) else np.nan
        rec["label_kind"] = top["label_kind"]
        rec["n_labels"] = len(g)
        rec["curve_ok"] = bool(g["curve_ok"].all()) if "curve_ok" in g else None
        out.append(rec)
    df = pd.DataFrame(out)
    df["source_file"] = Path(path).name
    return df


def _daily_peaks_from_curve(path: str | Path) -> pd.DataFrame:
    """For charts whose labels are drawn as glyph outlines (unreadable as text, e.g. 2024-25
    files): daily max of the digitised red curve. Approximate (~0.3 GW), marked as such."""
    c = parse_forecast_curve(path).dropna(subset=["forecast_mw"])
    log.warning("%s: no text labels; daily peaks derived from the curve", Path(path).name)
    g = c.groupby(c["timestamp"].dt.normalize())["forecast_mw"].max().round()
    df = pd.DataFrame({"forecast_start": c["forecast_start"].iloc[0], "date": g.index,
                       "low_mw": g.values, "high_mw": g.values,
                       "solar_hr_low_mw": np.nan, "solar_hr_high_mw": np.nan,
                       "non_solar_hr_low_mw": np.nan, "non_solar_hr_high_mw": np.nan,
                       "label_kind": "curve", "n_labels": 0, "curve_ok": None})
    df["source_file"] = Path(path).name
    return df


def parse_forecast_curve(path: str | Path, freq: str = "15min") -> pd.DataFrame:
    """The red point-forecast curve (and grey band, if drawn) resampled to `freq`.

    Precision is limited by the drawing (~0.2-0.5 GW), so treat this as an approximate
    benchmark; the text labels are the exact published numbers.
    """
    path = Path(path)
    ch = _load_chart(path)
    if ch.red is None:
        raise ForecastParseError(f"{path.name}: red forecast curve not found")

    grid_h = np.arange(0, 24 * ch.n_days, pd.Timedelta(freq).total_seconds() / 3600)
    h, v = ch.x_to_h(ch.red[:, 0]), ch.y_to_mw(ch.red[:, 1])
    o = np.argsort(h, kind="stable")
    df = pd.DataFrame({
        "timestamp": pd.Timestamp(ch.start) + pd.to_timedelta(grid_h, unit="h"),
        "forecast_mw": np.interp(grid_h, h[o], v[o]),
        "band_low_mw": np.nan, "band_high_mw": np.nan,
    })
    df.loc[(grid_h < h.min()) | (grid_h > h.max()), "forecast_mw"] = np.nan
    if ch.band is not None:
        # Polygon traced forward along one edge, then back along the other.
        bh, bv = ch.x_to_h(ch.band[:, 0]), ch.y_to_mw(ch.band[:, 1])
        # Split at the right-most x; skip the vertical closing segments at both ends.
        at_max = np.flatnonzero(np.isclose(bh, bh.max()))
        t1, t2 = at_max[0], at_max[-1]
        tail_min = t2 + int(np.argmin(bh[t2:]))
        edges = []
        for eh, ev in ((bh[:t1 + 1], bv[:t1 + 1]), (bh[t2:tail_min + 1], bv[t2:tail_min + 1])):
            oo = np.argsort(eh, kind="stable")
            edges.append(np.interp(grid_h, eh[oo], ev[oo]))
        inside = (grid_h >= bh.min()) & (grid_h <= bh.max())
        df.loc[inside, "band_low_mw"] = np.minimum(*edges)[inside]
        df.loc[inside, "band_high_mw"] = np.maximum(*edges)[inside]
    df.insert(0, "forecast_start", pd.Timestamp(ch.start))
    return df
