"""Power Peak AI: page chrome and the visual system shared by every page. Pages never style
anything themselves; they call these helpers.

    Router (app/Home.py):  ui.inject_css(); ui.sidebar(nav pages); page.run()
    Every page:            ui.page("Title", "One-line description.")
                           ... ui.headline / ui.kpis / ui.card / ui.chart / ui.table / ui.badge ...
                           ui.footer()        (ui.stop() ends a page early but keeps the footer)

Design tokens (light / dark), defined once here as CSS variables and mirrored in
.streamlit/config.toml and src/viz.py:
    background  #FAFAFA / #09090B      surface (cards)  #FFFFFF / #18181B
    border      #E4E4E7 / #27272A      text             #09090B / #FAFAFA
    secondary   #71717A / #A1A1AA      accent (ours)    #2563EB
    alert       #DC2626 (alerts only)  success          #16A34A
Cards: 1px border, 12px radius, shadow 0 1px 2px rgba(0,0,0,.05). Inter 400/500/600 with tabular
figures for every number; JetBrains Mono for code and IDs. 8px spacing grid, content max 1180px.
"""
from __future__ import annotations

import html
import os

import pandas as pd
import streamlit as st

import common  # noqa: F401  (sets sys.path)
from src import app_data, fmt
from src.config import readonly
from src import pipeline as pl
from src import viz

APP_NAME = "Power Peak AI"
TAGLINE = "7-day electricity demand forecasts for India"
MODE_ENV = "POWER_PEAK_MODE"
# the sidebar in presentation mode, in this order (keys of the page table in Home.py)
PRESENTATION_PAGES = ("overview", "forecast", "pipeline", "scorecard", "model_lab")
PRESENTATION_LAB_TABS = ("Leaderboard", "Evaluation design")


def app_mode() -> str:
    """POWER_PEAK_MODE: "presentation" (default; five pages, a short Model lab) or "full" (everything).
    Anything other than "full" means presentation, so a typo never exposes the unfinished pages."""
    return "full" if os.environ.get(MODE_ENV, "").strip().lower() == "full" else "presentation"


def full_mode() -> bool:
    return app_mode() == "full"


SOURCES = ("Sources: Grid-India (NLDC) daily Power Supply Position reports and week-ahead forecasts; Open-Meteo weather. "
           "Not an official forecast.")

_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --pp-bg: #FAFAFA; --pp-card: #FFFFFF; --pp-text: #09090B; --pp-muted: #71717A; --pp-border: #E4E4E7;
        --pp-subtle: #F4F4F5; --pp-accent: #2563EB; --pp-accent-soft: rgba(37,99,235,.08); --pp-alert: #DC2626;
        --pp-good: #16A34A; --pp-warn: #B45309; --pp-shadow: 0 1px 2px rgba(0,0,0,.05);
        --pp-font: Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
        --pp-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-pp-theme="light"]) {
        --pp-bg: #09090B; --pp-card: #18181B; --pp-text: #FAFAFA; --pp-muted: #A1A1AA; --pp-border: #27272A;
        --pp-subtle: #27272A; --pp-accent-soft: rgba(37,99,235,.18); --pp-alert: #EF4444; --pp-good: #22C55E;
        --pp-warn: #F59E0B; --pp-shadow: 0 1px 2px rgba(0,0,0,.4); } }
html, body, .stApp { font-family: var(--pp-font); }
/* tabular figures wherever numbers line up (not globally: Inter's tabular set also widens hyphens) */
.pp-kpi .val, .pp-badge, table.pp-table td, .pp-list .v, .js-plotly-plot text, .pp-tl-ticks,
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums lining-nums; }
code, kbd, .pp-mono { font-family: var(--pp-mono) !important; }

/* chrome: hide only the Deploy button, the main menu, the footer and the decoration line. Never
   hide the header or the toolbar: since Streamlit 1.46 the button that reopens a collapsed sidebar
   (stExpandSidebarButton) lives inside the toolbar (tests/test_app_design.py guards this). */
#MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
[data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; pointer-events: none; }
header[data-testid="stHeader"] [data-testid="stToolbar"], header[data-testid="stHeader"] button { pointer-events: auto; }
[data-testid="stExpandSidebarButton"] { display: inline-flex !important; visibility: visible !important;
        background: var(--pp-card); border: 1px solid var(--pp-border); border-radius: 8px; box-shadow: var(--pp-shadow); }
/* the collapse control stays visible, not only on hover */
[data-testid="stSidebarCollapseButton"] { display: block !important; visibility: visible !important; opacity: 1 !important; }

/* width and rhythm: 8px grid, max 1180px */
.block-container, [data-testid="stMainBlockContainer"] { max-width: 1180px; padding: 40px 40px 64px; }
[data-testid="stVerticalBlock"] { gap: 16px; }
h1 { font-size: 1.75rem !important; font-weight: 600 !important; letter-spacing: -0.025em; line-height: 1.2 !important;
     padding: 0 !important; margin: 0 !important; }
h2, h3 { font-weight: 600 !important; letter-spacing: -0.015em; }
h3 { font-size: 1.05rem !important; padding: 8px 0 0 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--pp-muted) !important; opacity: 1 !important;
     font-size: .82rem; line-height: 1.5; }
[data-testid="stHeaderActionElements"] { display: none; }

/* page header: title + description on the left, status badges on the right */
.pp-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin: 0 0 8px; }
.pp-head h1 { font-size: 1.75rem; font-weight: 600; letter-spacing: -0.025em; line-height: 1.2; margin: 0; color: var(--pp-text); }
.pp-head p { margin: 6px 0 0; color: var(--pp-muted); font-size: .92rem; line-height: 1.5; max-width: 46rem; }
.pp-head .pp-badges { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; padding-top: 6px; }
.pp-stale { font-size: .9rem; color: var(--pp-text); }

/* sidebar: wordmark, search, grouped links, fetch button, timestamp */
[data-testid="stSidebarHeader"] { height: 8px; min-height: 0; padding: 0; }
[data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] { position: absolute; right: 12px; top: 20px; }
[data-testid="stSidebarUserContent"] { padding-top: 16px; }
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap: 2px; }
.pp-brand { padding: 4px 8px 16px; }
.pp-brand .name { font-weight: 600; font-size: 1.05rem; letter-spacing: -0.02em; color: var(--pp-text); }
.pp-brand .tag { font-size: .8rem; color: var(--pp-muted); margin-top: 2px; line-height: 1.4; }
[data-testid="stSidebarUserContent"] [data-testid="stTextInput"] { margin-bottom: 8px; }
.pp-nav-sec { font-size: .72rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; color: var(--pp-muted);
        padding: 16px 8px 6px; }
[data-testid="stPageLink-NavLink"] { border-radius: 8px; padding: 6px 8px !important; }
[data-testid="stPageLink-NavLink"] p, [data-testid="stPageLink-NavLink"] span { font-size: .9rem; color: var(--pp-text); }
[data-testid="stPageLink-NavLink"]:hover { background: var(--pp-subtle); }
[data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] { color: var(--pp-muted); font-size: 18px; }
.pp-nav-gap { height: 24px; }
.pp-side-foot { font-size: .75rem; color: var(--pp-muted); line-height: 1.5; padding: 8px 8px 0; font-variant-numeric: tabular-nums; }

/* tabs: minimal underline, accent indicator on the active tab, no boxes */
[data-testid="stTabs"] [role="tablist"] { gap: 24px; border-bottom: 1px solid var(--pp-border); }
[data-testid="stTab"] { padding: 8px 0 10px !important; background: transparent !important; }
[data-testid="stTab"] p { font-size: .9rem !important; font-weight: 500; color: var(--pp-muted); }
[data-testid="stTab"][aria-selected="true"] p { color: var(--pp-text); }
[data-testid="stTab"]:hover p { color: var(--pp-text); }
[data-testid="stTabs"] .react-aria-SelectionIndicator { background: var(--pp-accent) !important; height: 2px !important; }
[data-testid="stTabPanel"] { padding-top: 16px; }

/* cards: ui.card() containers, KPI cards, table frames */
[class*="st-key-card_"] { background: var(--pp-card); border: 1px solid var(--pp-border) !important; border-radius: 12px !important;
        box-shadow: var(--pp-shadow); padding: 20px !important; }
.pp-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 8px 0; }
.pp-kpi { background: var(--pp-card); border: 1px solid var(--pp-border); border-radius: 12px; padding: 16px 20px;
        box-shadow: var(--pp-shadow); min-width: 0; }
.pp-kpi .lbl { font-size: .8rem; font-weight: 500; color: var(--pp-muted); display: flex; gap: 6px; align-items: center; }
.pp-kpi .val { font-size: 30px; font-weight: 600; color: var(--pp-text); line-height: 1.15; letter-spacing: -0.03em;
        margin-top: 8px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pp-kpi .val small { font-size: .9rem; font-weight: 500; color: var(--pp-muted); margin-left: 4px; letter-spacing: 0; }
.pp-kpi .chg { font-size: .78rem; margin-top: 8px; color: var(--pp-muted); line-height: 1.4; }
.pp-kpi .chg b { font-weight: 600; }
.pp-kpi .chg.up b { color: var(--pp-good); } .pp-kpi .chg.down b { color: var(--pp-alert); }
.pp-tip { display: inline-flex; width: 14px; height: 14px; border-radius: 50%; border: 1px solid var(--pp-border);
        font-size: 9px; font-weight: 600; align-items: center; justify-content: center; color: var(--pp-muted); cursor: help; }

/* headline sentence, section labels, badges, notes, insights */
.pp-headline { font-size: 1.35rem; line-height: 1.4; font-weight: 500; color: var(--pp-text); margin: 8px 0;
        max-width: 56rem; letter-spacing: -0.02em; }
.pp-headline em { font-style: normal; color: var(--pp-accent); }
.pp-kicker { font-size: .75rem; letter-spacing: .04em; color: var(--pp-muted); font-weight: 600; margin: 8px 0 0;
        text-transform: uppercase; }
.pp-badge { display: inline-flex; align-items: center; gap: 6px; font-size: .75rem; font-weight: 500; line-height: 1;
        padding: 5px 10px; border: 1px solid var(--pp-border); border-radius: 999px; color: var(--pp-muted);
        background: var(--pp-card); white-space: nowrap; font-variant-numeric: tabular-nums; }
.pp-badge.live::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--pp-good); }
.pp-badge.warn { color: var(--pp-warn); }
.pp-badges-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0; }
.pp-badge.alert { color: var(--pp-alert); }
.pp-note { font-size: .86rem; color: var(--pp-muted); border-left: 2px solid var(--pp-border); padding: 2px 0 2px 12px; margin: 4px 0;
        line-height: 1.5; }
.pp-note.alert { border-left-color: var(--pp-alert); color: var(--pp-text); }
.pp-insights { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin: 8px 0; }
.pp-insight { border: 1px solid var(--pp-border); border-radius: 12px; background: var(--pp-card); padding: 14px 16px;
        box-shadow: var(--pp-shadow); font-size: .88rem; line-height: 1.5; color: var(--pp-text); }
.pp-insight .k { display: block; font-size: .72rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
        color: var(--pp-muted); margin-bottom: 4px; }
.pp-list { margin: 0 !important; padding: 0 !important; list-style: none; }
.pp-list li { margin: 0 !important; padding: 10px 0; border-bottom: 1px solid var(--pp-subtle); font-size: .9rem; line-height: 1.5;
        display: flex; justify-content: space-between; gap: 16px; }
.pp-list li:last-child { border-bottom: none; }
.pp-list li span.v { font-variant-numeric: tabular-nums; font-weight: 600; white-space: nowrap; }
.pp-list li span.v.up { color: var(--pp-accent); } .pp-list li span.v.down { color: var(--pp-muted); }

/* empty state: centred card, one sentence, the exact command */
.pp-empty { text-align: center; border: 1px dashed var(--pp-border); border-radius: 12px; background: var(--pp-card);
        padding: 32px 24px; margin: 8px 0; }
.pp-empty p { margin: 0 0 12px; color: var(--pp-text); font-size: .95rem; }
.pp-empty code { display: inline-block; font-size: .82rem; background: var(--pp-subtle); color: var(--pp-text);
        border-radius: 6px; padding: 6px 10px; }

/* tables: no index, right-aligned tabular numbers, light row dividers */
.pp-table-wrap { overflow: auto; border: 1px solid var(--pp-border); border-radius: 12px; background: var(--pp-card);
        box-shadow: var(--pp-shadow); }
table.pp-table { width: 100%; border-collapse: collapse; font-size: .86rem; margin: 0; }
table.pp-table th { text-align: left; font-weight: 500; color: var(--pp-muted); font-size: .78rem; padding: 10px 16px;
        border-bottom: 1px solid var(--pp-border); position: sticky; top: 0; background: var(--pp-card);
        white-space: normal; vertical-align: bottom; line-height: 1.3; }
table.pp-table td { padding: 8px 16px; border-bottom: 1px solid var(--pp-subtle); color: var(--pp-text);
        vertical-align: top; line-height: 1.4; }
table.pp-table th, table.pp-table td { border-left: none !important; border-right: none !important; border-top: none; }
table.pp-table tr:last-child td { border-bottom: none; }
table.pp-table .date { white-space: nowrap; font-variant-numeric: tabular-nums; }
table.pp-table td:first-child, table.pp-table td.nw { white-space: nowrap; }
table.pp-table .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.pp-table th.num { text-align: right; }

/* widgets */
[data-testid="stBaseButton-primary"] { box-shadow: var(--pp-shadow); }
[data-testid="stBaseButton-tertiary"] { color: var(--pp-muted) !important; font-size: .82rem; min-height: 28px; padding: 0 4px; }
[data-testid="stBaseButton-tertiary"]:hover { color: var(--pp-text) !important; }
[data-testid="stExpander"] details { border-radius: 12px; border-color: var(--pp-border); background: var(--pp-card); }

/* pipeline step indicator */
.pp-flow {display: flex; align-items: flex-start; overflow-x: auto; padding: 8px 2px 16px;}
.pp-node {display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; width: 92px;}
.pp-dot {width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center;
          justify-content: center; font-size: .78rem; font-weight: 600; border: 1.5px solid var(--pp-border); color: var(--pp-muted);}
.pp-lbl {font-size: .75rem; margin-top: 6px; text-align: center; line-height: 1.2; color: var(--pp-text);}
.pp-state {font-size: .7rem; color: var(--pp-muted); margin-top: 2px; text-align: center;}
.pp-bar {flex: 1 1 20px; min-width: 14px; height: 1px; margin-top: 14px; background: var(--pp-border);}
.pp-bar.on {background: var(--pp-accent);}
.s-done .pp-dot, .s-nodata .pp-dot {background: var(--pp-accent); border-color: var(--pp-accent); color: #fff;}
.s-running .pp-dot {border-color: var(--pp-accent); color: var(--pp-accent); animation: pp-pulse 1.2s ease-in-out infinite;}
.s-failed .pp-dot {background: var(--pp-alert); border-color: var(--pp-alert); color: #fff;}
.s-skipped .pp-dot {border-style: dashed; opacity: .5;}
@keyframes pp-pulse {0%, 100% {box-shadow: 0 0 0 0 rgba(37,99,235,.35);} 50% {box-shadow: 0 0 0 5px rgba(37,99,235,0);}}

/* live update panel: one row per step with state and elapsed time */
.pp-live {margin: 8px 0 4px;}
.pp-live-row {display: grid; grid-template-columns: 28px 1fr auto 72px; gap: 4px 12px; align-items: center; padding: 8px 0;
          border-bottom: 1px solid var(--pp-subtle);}
.pp-live-row:last-child {border-bottom: none;}
.pp-live-row .n {width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
          font-size: .78rem; font-weight: 600; border: 1.5px solid var(--pp-border); color: var(--pp-muted);}
.pp-live-row .t {font-size: .95rem; color: var(--pp-text);}
.pp-live-row .st {font-size: .8rem; color: var(--pp-muted); white-space: nowrap;}
.pp-live-row .el {font-size: .85rem; font-variant-numeric: tabular-nums; text-align: right; color: var(--pp-muted);}
.pp-live-row .sub {grid-column: 2 / -1; font-size: .8rem; color: var(--pp-muted); line-height: 1.4; margin-top: -2px;}
.pp-live-row.s-running .t {font-weight: 600;} .pp-live-row.s-running .st, .pp-live-row.s-running .el {color: var(--pp-accent);}
.pp-live-row.s-done .n, .pp-live-row.s-nodata .n {background: var(--pp-accent); border-color: var(--pp-accent); color: #fff;}
.pp-live-row.s-running .n {border-color: var(--pp-accent); color: var(--pp-accent); animation: pp-pulse 1.2s ease-in-out infinite;}
.pp-live-row.s-failed .n {background: var(--pp-alert); border-color: var(--pp-alert); color: #fff;}
.pp-live-row.s-failed .st {color: var(--pp-alert);}
.pp-live-row.s-skipped .n {border-style: dashed; opacity: .5;}
.pp-live-row.s-pending .t {color: var(--pp-muted);}

/* evaluation-design timeline */
.pp-tl { margin: 8px 0 4px; }
.pp-tl-bar { display: flex; height: 48px; border-radius: 12px; overflow: hidden; border: 1px solid var(--pp-border); }
.pp-tl-seg { display: flex; min-width: 96px; flex-direction: column; justify-content: center; padding: 0 12px; min-width: 0;
        font-size: .78rem; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--pp-text); }
.pp-tl-seg b { font-weight: 600; }
.pp-tl-seg.train { background: var(--pp-subtle); }
.pp-tl-seg.val { background: var(--pp-border); }
.pp-tl-seg.test { background: var(--pp-accent-soft); }
.pp-tl-seg.live { background: var(--pp-accent); color: #fff; }
.pp-tl-ticks { position: relative; height: 20px; font-size: .72rem; color: var(--pp-muted); font-variant-numeric: tabular-nums; }
.pp-tl-ticks span { position: absolute; transform: translateX(-50%); top: 4px; white-space: nowrap; }
.pp-tl-ticks span:first-child { transform: none; }
.pp-tl-ticks span:last-child { transform: translateX(-100%); }
.pp-footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--pp-border);
             font-size: .78rem; color: var(--pp-muted); line-height: 1.6; }
</style>"""


# ============================================================================ frame
GLOSSARY = {
    "MAPE": "Mean absolute percentage error: the average size of the forecast error as a share of actual demand. "
            "Lower is better.",
    "P10–P90": "The range we expect demand to fall inside on 8 days out of 10. P50 is the central forecast.",
    "pinball loss": "The error measure used to train quantile forecasts: it penalises a P10 or P90 that is on the "
                    "wrong side of the actual value in proportion to how far off it is.",
    "win rate": "Share of days on which our forecast was closer to the actual value than Grid-India's.",
    "coverage": "Share of days on which the actual value fell inside the P10–P90 range. The target is 80 %.",
    "Brier skill": "How much better our record probabilities are than always predicting the historical rate. "
                   "Above 0 is better than that baseline.",
}


def theme_mode() -> str:
    try:
        t = st.context.theme.type
        return t if t in ("light", "dark") else "light"
    except Exception:
        return "light"


def inject_css() -> None:
    """Once per run, from the router: CSS + chart colours for the viewer's theme."""
    st.markdown(_CSS, unsafe_allow_html=True)
    viz.set_mode(theme_mode())
    st.session_state["_pp_n"] = 0          # per-run counter for stable download-button keys


def _seq(prefix: str) -> str:
    st.session_state["_pp_n"] = st.session_state.get("_pp_n", 0) + 1
    return f"{prefix}_{st.session_state['_pp_n']}"


def badge(text: str, tone: str = "neutral") -> str:
    """Small pill status label (HTML string): badge('Live', 'live'), badge('Experimental', 'warn')."""
    cls = {"warn": " warn", "alert": " alert", "live": " live"}.get(tone, "")
    return f"<span class='pp-badge{cls}'>{html.escape(text)}</span>"


def badges(*items: str) -> None:
    """A row of badges in the page flow: ui.badges(ui.badge('Experimental', 'warn'))."""
    st.markdown("<div class='pp-badges-row'>" + "".join(items) + "</div>", unsafe_allow_html=True)


def terms(*names: str) -> None:
    """One caption line defining the jargon a view uses: ui.terms('MAPE', 'P10–P90')."""
    st.caption(" ".join(f"{n}: {GLOSSARY[n]}" for n in names))


def tip(term: str) -> str:
    """A small 'i' marker whose hover text explains a term from GLOSSARY (HTML string)."""
    return f"<span class='pp-tip' title='{html.escape(GLOSSARY.get(term, term))}'>i</span>"


def page(title: str, description: str, badges: list[str] | None = None) -> None:
    """Page header: title (sentence case), then the one-line description with status badges
    (HTML from badge()) aligned right. Also shows the result card after a data update."""
    st.title(title, anchor=False)
    right = "".join(badges or [])
    st.markdown(f"<div class='pp-head'><div><p>{html.escape(description)}</p></div>"
                f"<div class='pp-badges'>{right}</div></div>", unsafe_allow_html=True)
    _last_run_card()


def page_with_action(title: str, description: str, key: str = "fetch_page") -> bool:
    """Page header with the status badge and the primary 'Update data' button top right (the
    button only when fetching is enabled). Returns True when the button was pressed."""
    left, right = st.columns([2.2, 1.5], vertical_alignment="top")
    with left:
        st.title(title, anchor=False)
        st.markdown(f"<div class='pp-head'><div><p>{html.escape(description)}</p></div></div>", unsafe_allow_html=True)
    clicked = False
    with right, st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap="small"):
        st.markdown(updated_badge(), unsafe_allow_html=True)
        if fetch_enabled():
            clicked = update_button(key)
    _last_run_card()
    return clicked


def update_button(key: str) -> bool:
    return st.button(FETCH_LABEL, key=key, type="primary", icon=":material/refresh:", help=FETCH_HELP)


def updated_badge() -> str:
    ts = app_data.last_updated()
    if readonly() and ts is not None:
        return badge(f"Data snapshot: {pd.Timestamp(ts):%d %b %Y}")
    return badge(f"Updated {pd.Timestamp(ts):%d %b, %H:%M}") if ts is not None else ""


def headline(text: str) -> None:
    """The one-sentence summary at the top of a page. *word* is set in the accent colour."""
    import re

    t = html.escape(text)
    t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
    st.markdown(f"<div class='pp-headline'>{t}</div>", unsafe_allow_html=True)


def kicker(text: str) -> None:
    """Small section label above a panel."""
    st.markdown(f"<div class='pp-kicker'>{html.escape(text)}</div>", unsafe_allow_html=True)


def note(text: str, alert: bool = False) -> None:
    """A quiet explanatory note with a left rule (red rule only for alerts). `text` is plain text."""
    st.markdown(f"<div class='pp-note{' alert' if alert else ''}'>{html.escape(text)}</div>", unsafe_allow_html=True)


def card(name: str):
    """A surface: white (dark: zinc-900) card with a 1px border, 12px radius and a faint shadow.
    Use as a context manager: `with ui.card("drivers"): ...`."""
    return st.container(border=False, key=f"card_{name}")


def kpis(items: list[dict]) -> None:
    """KPI cards, max 4 per row: [{label, value, unit?, delta?, good?, note?, help?}].
    `delta` is a short change ('+2.1 GW'), coloured green when `good` is True and red when False;
    `note` follows it in grey ('vs last week'). `help` names a GLOSSARY term (or is free text)
    and adds an 'i' tooltip to the label."""
    cards = []
    for k in items[:4]:
        unit = f"<small>{html.escape(k['unit'])}</small>" if k.get("unit") else ""
        lbl = html.escape(k["label"])
        if k.get("help"):
            lbl += " " + tip(k["help"])
        chg = ""
        if k.get("change") and not k.get("delta"):          # older call sites: one grey/green/red line
            k = {**k, "delta": k["change"]}
        if k.get("delta") or k.get("note"):
            cls = {True: " up", False: " down"}.get(k.get("good"), "")
            d = f"<b>{html.escape(k['delta'])}</b> " if k.get("delta") else ""
            chg = f"<div class='chg{cls}'>{d}{html.escape(k.get('note') or '')}</div>"
        cards.append(f"<div class='pp-kpi'><div class='lbl'>{lbl}</div>"
                     f"<div class='val'>{html.escape(str(k['value']))}{unit}</div>{chg}</div>")
    st.markdown("<div class='pp-kpis'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def insights(items: list[tuple[str, str]]) -> None:
    """The 'Insights' strip: 2–3 short plain-English findings as small cards [(label, sentence)]."""
    if not items:
        return
    cells = "".join(f"<div class='pp-insight'><span class='k'>{html.escape(k)}</span>{html.escape(v)}</div>"
                    for k, v in items[:3])
    st.markdown(f"<div class='pp-insights'>{cells}</div>", unsafe_allow_html=True)


def value_list(rows: list[tuple[str, str, bool | None]]) -> None:
    """A divided list of (sentence, value, up?) rows; up True = accent, False = grey."""
    li = "".join(f"<li><span>{html.escape(a)}</span><span class='v{' up' if u else ' down' if u is False else ''}'>"
                 f"{html.escape(b)}</span></li>" for a, b, u in rows)
    st.markdown(f"<ul class='pp-list'>{li}</ul>", unsafe_allow_html=True)


def chart(fig, modebar: bool = False, key: str | None = None, half: bool = False) -> None:
    """Every chart goes through here: our template (not Streamlit's), headline and subtitle
    wrapped to the column (half=True inside two columns), mode bar hidden except where a page
    asks for it (Model lab)."""
    viz.fit(fig, half=half)
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displayModeBar": modebar, "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})


PERIODS = {"3 months": 91, "1 year": 365, "2 years": 730, "All": None}


def period(d: pd.DataFrame | pd.Series | None, key: str, default: str = "1 year"):
    """Compact period selector -> (start, end) timestamps (None without data)."""
    if d is None or len(d) == 0:
        return None
    sel = st.segmented_control("Period", list(PERIODS), default=default, key=key, label_visibility="collapsed")
    days = PERIODS.get(sel or default)
    end = pd.Timestamp(d.index.max())
    start = pd.Timestamp(d.index.min()) if days is None else max(pd.Timestamp(d.index.min()), end - pd.Timedelta(days=days))
    return start, end


def timeline(segments: list[tuple[str, str, str, str, str]], ticks: list[str] | None = None) -> None:
    """Horizontal timeline of the evaluation design. segments: (css class train|val|test|live,
    title, subtitle, start, end) with ISO dates; widths are proportional to the time span (with a
    minimum so short segments stay readable); ticks: ISO dates shown under the bar."""
    t0 = pd.Timestamp(segments[0][3])
    t1 = pd.Timestamp(segments[-1][4])
    span = (t1 - t0).days or 1
    widths = [max(((pd.Timestamp(e) - pd.Timestamp(s)).days + 1) / span, 0.12) for _, _, _, s, e in segments]
    tot = sum(widths)
    segs = "".join(f"<div class='pp-tl-seg {c}' style='flex:0 0 {100 * w / tot:.2f}%' title='{html.escape(a)}: "
                   f"{html.escape(b)}'><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>"
                   for (c, a, b, _, _), w in zip(segments, widths))
    pos, acc = {}, 0.0
    for (c, a, b, s, e), w in zip(segments, widths):            # tick positions follow the drawn segments
        pos[s] = acc
        acc += 100 * w / tot
        pos[e] = acc
    tick_html = "".join(f"<span style='left:{pos.get(d, 0):.2f}%'>{pd.Timestamp(d):%d %b %Y}</span>"
                        for d in (ticks or []) if d in pos)
    st.markdown(f"<div class='pp-tl'><div class='pp-tl-bar'>{segs}</div><div class='pp-tl-ticks'>{tick_html}</div></div>",
                unsafe_allow_html=True)


# what is missing -> the exact command that produces it
FIX = {
    "pipeline": "python -m src.pipeline --trigger manual",
    "backtest": "python scripts/backtest.py",
    "audit": "python scripts/audit.py",
    "fixed_split": "python scripts/fixed_split.py",
    "train_lgbm": "python scripts/train.py --model lgbm",
    "backfill": "python scripts/backfill.py --start 2023-04-01 --forecasts --build",
}


def empty_state(sentence: str, fix: str | None = "pipeline") -> None:
    """Centred card: one sentence on what is missing, then the exact command that fixes it
    (a FIX key or a literal command; None when there is nothing to run)."""
    cmd = FIX.get(fix, fix) if fix else None
    code = f"<code>{html.escape(cmd)}</code>" if cmd else ""
    st.markdown(f"<div class='pp-empty'><p>{html.escape(sentence)}</p>{code}</div>", unsafe_allow_html=True)


def not_available(what: str, fix: str | None = "pipeline"):
    """Backwards-compatible: '<what> is not available yet' + the command (see empty_state)."""
    empty_state(f"{what} is not available yet. Run this to create it:" if fix else f"{what} is not available yet.", fix)


def snapshot_text() -> str:
    """'Data snapshot: 16 Nov 2026 (actuals through 14 Nov 2026)': the snapshot date, then the last day of actuals."""
    ts = app_data.snapshot_time() or app_data.last_updated()
    last = app_data.summary()
    through = f" (actuals through {pd.Timestamp(last['date'].max()):%d %b %Y})" if last is not None and len(last) else ""
    return "Data snapshot: " + ("not available yet" if ts is None else f"{pd.Timestamp(ts):%d %b %Y}") + through


def footer():
    """On every page: sources, the 'not official' notice, and the snapshot date."""
    st.markdown(f"<div class='pp-footer'>{SOURCES} {snapshot_text()}.</div>", unsafe_allow_html=True)


def stamp(ts) -> str:
    """'11 Nov 2026, 10:02'"""
    return "–" if ts is None or pd.isna(ts) else f"{pd.Timestamp(ts):%d %b %Y, %H:%M}"


def stop():
    """End the page early but keep the footer."""
    footer()
    st.stop()


# ============================================================================ sidebar
def _matches(q: str, section: str, page, extra: str) -> bool:
    hay = f"{section} {page.title} {extra}".lower()
    return all(w in hay for w in q.lower().split())


def _jump(sections: dict, keywords: dict):
    """Enter in the search box: go straight to the page when exactly one matches."""
    q = (st.session_state.get("nav_q") or "").strip()
    hits = [p for sec, ps in sections.items() for p in ps if q and _matches(q, sec, p, keywords.get(p.url_path, ""))]
    if len(hits) == 1:
        st.session_state["_pp_jump"] = hits[0]
        st.session_state["nav_q"] = ""


def sidebar(sections: dict, keywords: dict | None = None) -> bool:
    """The whole sidebar: wordmark + tagline, 'Jump to…' search that filters the links, the
    grouped page links, the 'Update data' button, then the data timestamp. Returns True
    when the button was pressed (the router then runs the update above the page)."""
    keywords = keywords or {}
    jump = st.session_state.pop("_pp_jump", None)
    if jump is not None:
        st.switch_page(jump)
    clicked = False
    with st.sidebar:
        st.markdown(f"<div class='pp-brand'><div class='name'>{APP_NAME}</div><div class='tag'>{TAGLINE}</div></div>",
                    unsafe_allow_html=True)
        q = st.text_input("Jump to", key="nav_q", placeholder="Jump to…", label_visibility="collapsed",
                          icon=":material/search:", on_change=_jump, args=(sections, keywords))
        q = (q or "").strip()
        shown = 0
        for sec, pages in sections.items():
            hits = [p for p in pages if not q or _matches(q, sec, p, keywords.get(p.url_path, ""))]
            if not hits:
                continue
            st.markdown(f"<div class='pp-nav-sec'>{html.escape(sec)}</div>", unsafe_allow_html=True)
            for p in hits:
                st.page_link(p, label=p.title, icon=p.icon, width="stretch")
            shown += len(hits)
        if q and not shown:
            st.caption(f"No page matches “{q}”.")
        st.markdown("<div class='pp-nav-gap'></div>", unsafe_allow_html=True)
        if fetch_enabled():
            clicked = st.button(FETCH_LABEL, key="fetch_sidebar", type="primary", icon=":material/refresh:",
                                width="stretch", help=FETCH_HELP)
        if readonly():       # no button on a read-only deployment: say which data this is
            st.markdown(f"<div class='pp-side-foot'>{html.escape(snapshot_text())}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='pp-side-foot'>Data updated: {stamp(app_data.last_updated())}</div>",
                        unsafe_allow_html=True)
    return clicked


# ============================================================================ tables
def _num_fmt(col: str, s: pd.Series):
    dec = None
    lc = str(col)
    if lc.lower() in ("year", "fy", "#") or lc.lower().startswith("year"):      # identifiers: no separators
        return lambda x: "–" if pd.isna(x) else f"{int(x)}"
    if "(GW)" in lc or "°C" in lc:
        dec = 1
    elif "(MU)" in lc or "(TWh)" in lc or lc.lower() in ("days", "n", "weeks", "rows"):
        dec = 0
    elif "%" in lc or "MAPE" in lc:
        dec = 1 if "MAPE" not in lc else 2
    if dec is None:
        v = s.dropna().abs()
        dec = 0 if (len(v) and (v % 1 == 0).all()) else 2
    return lambda x: "–" if pd.isna(x) else f"{x:,.{dec}f}"


def table(df: pd.DataFrame, max_height: int | None = 420, index: bool | None = None, name: str = "table",
          download: bool = True, **_ignored) -> None:
    """HTML table in a card: no index (unless it is meaningful: a named or date index), numbers
    right-aligned with thousand separators and the unit's precision, dates as '09 Nov 2026'.
    Below it, right-aligned, a 'Download CSV' button with the unformatted values (`name` sets
    the file name: power-peak-<name>.csv)."""
    d = df.copy()
    show_index = index if index is not None else (d.index.name is not None or isinstance(d.index, pd.DatetimeIndex))
    if show_index:
        if isinstance(d.index, pd.DatetimeIndex):
            d.index = d.index.strftime(fmt.DATE)
        d = d.reset_index()
    if download:
        _download_after = d.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [" · ".join(map(str, c)) for c in d.columns]
        if download:
            _download_after.columns = d.columns
    cols = list(d.columns)
    numeric = {c: pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]) for c in cols}
    fmts = {c: _num_fmt(c, d[c]) for c in cols if numeric[c]}
    short = {c for c in cols if not numeric[c] and d[c].astype(str).str.len().max() <= 24}   # no wrapping
    head = "".join(f"<th class='{'num' if numeric[c] else ''}'>{html.escape(str(c))}</th>" for c in cols)
    rows = []
    for r in d.itertuples(index=False):
        cells = []
        for c, v in zip(cols, r):
            if numeric[c]:
                cells.append(f"<td class='num'>{fmts[c](v)}</td>")
            elif isinstance(v, (pd.Timestamp,)) or (hasattr(v, "year") and hasattr(v, "month") and not isinstance(v, str)):
                cells.append(f"<td class='date'>{fmt.date(v)}</td>")
            else:
                cls = " class='nw'" if c in short else ""
                cells.append(f"<td{cls}>{'–' if v is None or (isinstance(v, float) and pd.isna(v)) else html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    style = f" style='max-height:{max_height}px'" if max_height else ""
    st.markdown(f"<div class='pp-table-wrap'{style}><table class='pp-table'><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>", unsafe_allow_html=True)
    if download:
        download_csv(_download_after, name)


def download_csv(df: pd.DataFrame, name: str, label: str = "Download CSV", primary: bool = False) -> None:
    """A right-aligned CSV download (tertiary link-style button unless primary)."""
    slug = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        st.download_button(label, df.to_csv(index=False).encode(), file_name=f"power-peak-{slug}.csv",
                           mime="text/csv", icon=":material/download:", key=_seq("dl"),
                           type="secondary" if primary else "tertiary")


# ============================================================================ data update
FETCH_LABEL = "Update data"
FETCH_HELP = ("Downloads new Grid-India reports and weather, then rebuilds the features and the 7-day forecast. "
              "Takes about 5 minutes; the weather download is most of it.")
TRIGGER = {"button": "Run from the app", "timer": "Scheduled", "manual": "Command line", "cli": "Command line"}
_BADGE = {pl.PENDING: "gray", pl.RUNNING: "blue", pl.DONE: "green", pl.NO_DATA: "orange",
          pl.FAILED: "red", pl.SKIPPED: "gray"}
_CLS = {pl.PENDING: "s-pending", pl.RUNNING: "s-running", pl.DONE: "s-done", pl.NO_DATA: "s-nodata",
        pl.FAILED: "s-failed", pl.SKIPPED: "s-skipped"}


def fetch_enabled() -> bool:
    """ENABLE_FETCH_BUTTON (default true); always false when POWER_PEAK_READONLY=1."""
    if readonly():
        return False
    return os.environ.get("ENABLE_FETCH_BUTTON", "true").strip().lower() not in ("0", "false", "no", "off")


def _get(step, name):
    return step[name] if isinstance(step, dict) else getattr(step, name)


def flow_html(steps) -> str:
    """Horizontal step indicator; completed steps (done / no new data) are highlighted."""
    parts = []
    for i, s in enumerate(steps):
        status = _get(s, "status")
        if i:
            prev = _get(steps[i - 1], "status")
            parts.append(f"<div class='pp-bar{' on' if prev in pl.COMPLETED else ''}'></div>")
        parts.append(f"<div class='pp-node {_CLS.get(status, 's-pending')}'><div class='pp-dot'>{i + 1}</div>"
                     f"<div class='pp-lbl'>{_get(s, 'short')}</div><div class='pp-state'>{status}</div></div>")
    return "<div class='pp-flow'>" + "".join(parts) + "</div>"


def flow(steps) -> None:
    """Render the step indicator in the page."""
    st.markdown(flow_html(steps), unsafe_allow_html=True)


STATE = {pl.PENDING: "Waiting", pl.RUNNING: "Running", pl.DONE: "Done", pl.NO_DATA: "No new data",
         pl.FAILED: "Failed", pl.SKIPPED: "Skipped"}


def steps_table(steps) -> None:
    """The steps of a run as a regular table: number, step, state, duration, result."""
    table(pd.DataFrame([{"#": i, "Step": _get(s, "label"), "State": STATE.get(_get(s, "status"), _get(s, "status")),
                         "Duration": fmt.duration(_get(s, "duration_s")) if _get(s, "duration_s") is not None else "–",
                         "Result": _get(s, "message") or ""} for i, s in enumerate(steps, 1)]),
          max_height=None, download=False)


def steps_markdown(steps) -> str:
    """One line per step: number, label, state badge, duration and result."""
    rows = ["| | Step | State | Duration | Result |", "|---|---|---|---|---|"]
    for i, s in enumerate(steps, 1):
        status, dur = _get(s, "status"), _get(s, "duration_s")
        msg = (_get(s, "message") or "").replace("|", "/").replace("\n", " ")
        rows.append(f"| {i} | {_get(s, 'label')} | :{_BADGE.get(status, 'gray')}-badge[{status}] | "
                    f"{fmt.duration(dur) if dur is not None else ''} | {msg} |")
    return "\n".join(rows)


def _day(x) -> str:
    return "–" if not x else f"{pd.Timestamp(x):%a %d %b %Y}"


def _span(days: list[str]) -> str:
    return "" if not days else fmt.date_range(pd.Timestamp(days[0]), pd.Timestamp(days[-1]))


def changes_markdown(ch: dict | None) -> str:
    """The 'What changed' lines: latest data date before and after, new forecast dates, days added
    to the live scorecard."""
    if not ch:
        return ""
    b, a = ch.get("data_before"), ch.get("data_after")
    data = (f"{_day(b)} → **{_day(a)}**" if b != a else f"unchanged, still **{_day(a)}**")
    fc = ch.get("forecast_new_dates") or []
    fcs = (f"**{_span(fc)}** ({len(fc)} new day{'s' if len(fc) != 1 else ''})" if fc
           else "no new days" + (f" (forecast refreshed for {_span([str((pd.Timestamp(ch['forecast_after']) + pd.Timedelta(days=i)).date()) for i in (1, 7)])})"
                                 if ch.get("forecast_after") else ""))
    n = ch.get("scorecard_days_added") or 0
    sc = (f"**{n}** ({_span(ch['scorecard_new_dates'])})" if n else "**0**, none of the forecast days gained actuals")
    return (f"- **Latest data date:** {data}\n- **New forecast dates:** {fcs}\n"
            f"- **Days added to the live scorecard:** {sc}")


def changes_panel(ch: dict | None) -> None:
    md = changes_markdown(ch)
    if md:
        with st.container(border=True):
            st.markdown("**What changed**")
            st.markdown(md)


def _last_run_card():
    """Result card shown once, on the page the update ends on."""
    run = st.session_state.pop("pipeline_last_run", None)
    if not run:
        return
    status = run["status"]
    if status in ("success", "partial"):
        with st.container(border=True):
            st.markdown(f"**{'Update complete' if status == 'success' else 'Update complete, with warnings'}** · "
                        f"{fmt.datetime(run['finished'])} · {fmt.duration(run['duration_s'])}")
            st.markdown(run["summary"])
            changes_panel(run.get("changes"))
            with st.expander("Step details"):
                st.markdown(flow_html(run["steps"]), unsafe_allow_html=True)
                steps_table(run["steps"])
        return
    bad = next((s for s in run["steps"] if s["status"] == pl.FAILED), None)
    if bad is None:                                   # no failed step, e.g. the run ended some other way
        st.warning(run.get("summary") or "The update did not finish.")
        return
    i = [s["key"] for s in run["steps"]].index(bad["key"]) + 1
    st.error(f"Step {i}, {bad['label']}, failed: {bad['error']}.  \nResults of the earlier steps are saved. Try "
             "again in a few minutes; if it fails again, open Data pipeline › History for the details.")
    with st.expander("Traceback"):
        st.code(bad.get("traceback") or "", language="text")


# note shown under a step while the update runs (about how long it takes)
def step_notes() -> dict[str, str]:
    from src.data.weather import CITIES

    return {"weather": f"Fetching weather for {len(CITIES)} cities, about 4 minutes in total."}


def _elapsed(s: dict, now: pd.Timestamp) -> str:
    if s.get("duration_s") is not None:
        return fmt.duration(s["duration_s"])
    if s["status"] == pl.RUNNING and s.get("started"):
        return fmt.duration(max((now - pd.Timestamp(s["started"])).total_seconds(), 0))
    return ""


def live_steps_html(steps: list[dict], now: pd.Timestamp | None = None) -> str:
    """The seven steps as rows: number, name, state, elapsed time; a note and the result under it."""
    now = now or pd.Timestamp.now()
    notes = step_notes()
    rows = []
    for i, s in enumerate(steps, 1):
        sub = [notes[s["key"]]] if s["key"] in notes else []
        if s["status"] in (pl.DONE, pl.NO_DATA, pl.FAILED) and s.get("message"):
            sub.append(s["message"])
        sub_html = "".join(f"<div class='sub'>{html.escape(t)}</div>" for t in sub)
        rows.append(f"<div class='pp-live-row {_CLS.get(s['status'], 's-pending')}'><div class='n'>{i}</div>"
                    f"<div class='t'>{html.escape(s['label'])}</div>"
                    f"<div class='st'>{STATE.get(s['status'], s['status'])}</div>"
                    f"<div class='el'>{_elapsed(s, now)}</div>{sub_html}</div>")
    return "<div class='pp-live'>" + "".join(rows) + "</div>"


@st.fragment(run_every=1)
def _live_run():
    """The running update, redrawn every second on its own, so the rest of the page stays usable.
    It reads the lock and progress files, not a Python object: it works for a run started by this
    session, by another one (page refreshed during the run) or by the command line."""
    paths = pl.Paths()
    held = pl.read_lock(paths)
    if held is None:                                   # finished (or died): show the result, once
        settle_run()
        st.rerun()
        return
    prog = pl.read_progress(paths)
    if not prog or prog.get("status") != "running" or (held.get("run_id") and prog.get("run_id") != held["run_id"]):
        prog = {"steps": [dict(key=k, label=label, short=short, status=pl.PENDING)
                          for k, label, short in pl.STEPS], "started": held.get("started")}
    now = pd.Timestamp.now()
    done = sum(s["status"] in (*pl.COMPLETED, pl.FAILED, pl.SKIPPED) for s in prog["steps"])
    total = fmt.duration(max((now - pd.Timestamp(prog.get("started") or held["started"])).total_seconds(), 0))
    with st.container(border=True):
        st.subheader("Updating data")
        st.caption(f"{done} of {len(prog['steps'])} steps finished · {total} elapsed · started "
                   f"{pd.Timestamp(held['started']):%H:%M}. You can leave or refresh this page; the update keeps running.")
        st.markdown(flow_html(prog["steps"]), unsafe_allow_html=True)
        st.markdown(live_steps_html(prog["steps"], now), unsafe_allow_html=True)


def update_running() -> bool:
    return not readonly() and pl.read_lock(pl.Paths()) is not None


def settle_run() -> None:
    """When an update this session was watching has ended, queue its result card for the page and
    drop the cached data. Called by the live panel and at the top of every script run, so the result
    shows up whichever of the two notices the end first."""
    if st.session_state.get("_pp_watching") and not update_running():
        st.session_state["_pp_watching"] = False
        prog = pl.read_progress(pl.Paths())
        if prog and prog.get("status") != "running":
            st.session_state["pipeline_last_run"] = prog
        st.cache_data.clear()                          # whatever was saved is now the current data


def pipeline_panel(start: bool = True):
    """Show the update in progress, starting one first when `start` (the Update data button was
    pressed). The run itself happens in a background thread (pl.start_background): the page stays
    responsive, and a refresh or a second session just reattaches to the same run. Drawn at most once
    per script run. Never in a read-only deployment."""
    if readonly():
        return
    run_no = st.session_state.get("_pp_script_run", 0)
    if st.session_state.get("_pp_panel_run") == run_no:
        return
    st.session_state["_pp_panel_run"] = run_no
    if start and not update_running():
        try:
            pl.start_background(trigger="button")
        except pl.PipelineLocked:
            pass                                       # someone else got there first: just show theirs
        except Exception as e:                         # outside any step (e.g. the log directory is not writable)
            st.error(f"The update could not start ({type(e).__name__}: {e}).  \nCheck that the logs/ folder is "
                     "writable, then try again.")
            return
    st.session_state["_pp_watching"] = True
    _live_run()
