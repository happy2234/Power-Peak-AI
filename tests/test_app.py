"""Every page and every tab must render without exceptions — with data, and with nothing built yet.
The app runs through its router (app/Home.py, st.navigation); pages live in app/views/."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
MAIN = APP / "Home.py"
# page file -> (title, tab labels in page order, including tabs nested inside them)
PAGES = {
    "views/overview.py": ("Overview", []),
    "views/forecast.py": ("7-day forecast", ["Solar-hour peak", "Evening peak", "Energy"]),
    "views/what_if.py": ("What-if simulator", []),
    "views/vs_grid_india.py": ("Our forecast vs Grid-India", ["Live scorecard", "Backtest", "Cumulative error",
                                                               "Daily error", "Bias-corrected"]),
    "views/model_lab.py": ("Model lab", ["Leaderboard", "By horizon", "By weather", "Calibration",
                                          "Evaluation design", "Drift"]),
    "views/demand_explorer.py": ("Demand explorer", ["Trends", "Seasonality", "States"]),
    "views/weather.py": ("Weather impact", ["By condition", "By season", "Next 7 days"]),
    "views/evening_peak.py": ("Evening peak", ["Daytime vs evening", "Daily gap"]),
    "views/raw_to_processed.py": ("Raw to processed", ["Raw file", "Parse", "Clean", "Weather", "Model input", "Files"]),
    "views/data_pipeline.py": ("Data pipeline", ["Update", "History", "Data quality"]),
}


def run_page(page: str) -> AppTest:
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(str(MAIN), default_timeout=180)
    at.run()
    if page != "views/overview.py":
        at.switch_page(page)
        at.run()
    return at


def no_data(monkeypatch, tmp_path):
    import src.app_data as ad
    import src.models.baselines as bl
    import src.models.registry as reg

    import src.walkthrough as wt

    for mod, name in [(wt, "PROCESSED"), (wt, "MODELS_DIR"), (wt, "RAW_LATEST"), (ad, "PROCESSED"), (ad, "REPORTS"), (ad, "MODELS_DIR"), (reg, "MODELS_DIR"), (bl, "PROCESSED")]:
        monkeypatch.setattr(mod, name, tmp_path)


@pytest.mark.parametrize("page", list(PAGES), ids=lambda p: Path(p).stem)
def test_page_and_all_tabs_render_with_data(page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    title, tabs = PAGES[page]
    assert [t.value for t in at.title] == [title]
    labels = [t.label for t in at.tabs]
    it = iter(labels)                       # page tabs in order; nested tabs (secondary charts) may follow
    assert all(t in it for t in tabs), labels


@pytest.mark.parametrize("page", list(PAGES), ids=lambda p: Path(p).stem)
def test_page_renders_without_data(page, tmp_path, monkeypatch):
    no_data(monkeypatch, tmp_path)
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert at.info or at.markdown  # shows a "not available yet" message instead of crashing


@pytest.mark.parametrize("old", ["views/redirect_seasonality.py", "views/redirect_states.py"])
def test_old_urls_redirect_to_demand_explorer(old):
    at = run_page(old)
    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == ["Demand explorer"]


def test_navigation_sections_and_old_url_paths():
    src = MAIN.read_text()
    for section in ('"Forecast":', '"Performance":', '"Analysis":', '"Data":'):
        assert section in src
    for path in ("Forecast", "What_If", "Scorecard", "Model_Lab", "Demand_Explorer", "Weather", "Evening_Peak",
                 "Raw_to_processed", "Data_Pipeline", "Seasonality", "States"):
        assert f'url_path="{path}"' in src, path
    assert 'position="hidden"' in src          # ui.sidebar draws the grouped links


def test_sidebar_search_filters_links_and_jumps():
    at = run_page("views/overview.py")
    links = lambda: [e.proto.label for e in at.sidebar.get("page_link")]  # noqa: E731
    assert len(links()) == 10
    at.sidebar.text_input(key="nav_q").input("weather").run()       # page titles and tab names match
    assert links() == ["Model lab", "Weather impact"]
    at.sidebar.text_input(key="nav_q").input("zzz").run()
    assert links() == []
    at.sidebar.text_input(key="nav_q").input("calibration").run()   # one match: Enter goes there
    at.run()
    assert [t.value for t in at.title] == ["Model lab"]
    assert len(links()) == 10                                         # and the search is cleared


# ============================================================================ presentation mode
PRESENTATION_LINKS = ["Overview", "7-day forecast", "Raw to processed", "Data pipeline", "Our forecast vs Grid-India", "Model lab"]
FULL_ONLY = ["views/what_if.py", "views/demand_explorer.py", "views/weather.py", "views/evening_peak.py"]


def test_mode_defaults_to_presentation_and_only_full_means_full(monkeypatch):
    import ui

    monkeypatch.delenv("POWER_PEAK_MODE", raising=False)
    assert ui.app_mode() == "presentation" and not ui.full_mode()
    for v, want in [("full", "full"), (" FULL ", "full"), ("presentation", "presentation"), ("typo", "presentation")]:
        monkeypatch.setenv("POWER_PEAK_MODE", v)
        assert ui.app_mode() == want


def test_presentation_sidebar_lists_six_pages_and_keeps_update_button(monkeypatch):
    monkeypatch.delenv("POWER_PEAK_MODE", raising=False)
    at = run_page("views/overview.py")
    assert not at.exception, [e.value for e in at.exception]
    assert [e.proto.label for e in at.sidebar.get("page_link")] == PRESENTATION_LINKS
    assert "fetch_sidebar" in [b.key for b in at.sidebar.button]            # Update data stays
    at.sidebar.text_input(key="nav_q").input("simulator").run()               # hidden pages are not searchable
    assert [e.proto.label for e in at.sidebar.get("page_link")] == []
    at.sidebar.text_input(key="nav_q").input("calibration").run()             # nor are the hidden Model lab tabs
    assert [e.proto.label for e in at.sidebar.get("page_link")] == []


@pytest.mark.parametrize("page", FULL_ONLY)
def test_presentation_mode_does_not_route_the_other_pages(page, monkeypatch):
    monkeypatch.delenv("POWER_PEAK_MODE", raising=False)
    at = AppTest.from_file(str(MAIN), default_timeout=180)
    at.run()
    with pytest.raises(Exception):
        at.switch_page(page)
    monkeypatch.setenv("POWER_PEAK_MODE", "full")                              # and they are there in full mode
    assert not run_page(page).exception


@pytest.mark.parametrize("page", ["views/overview.py", "views/forecast.py", "views/vs_grid_india.py",
                                  "views/raw_to_processed.py", "views/data_pipeline.py", "views/model_lab.py"])
def test_presentation_pages_render(page, monkeypatch):
    monkeypatch.delenv("POWER_PEAK_MODE", raising=False)
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]


def test_presentation_model_lab_shows_leaderboard_and_evaluation_design_only(monkeypatch):
    monkeypatch.delenv("POWER_PEAK_MODE", raising=False)
    at = run_page("views/model_lab.py")
    assert not at.exception, [e.value for e in at.exception]
    labels = [t.label for t in at.tabs]
    assert labels[:2] == ["Leaderboard", "Evaluation design"]
    assert not {"By horizon", "By weather", "Calibration", "Drift"} & set(labels)


# ============================================================================ read-only deployment
def test_readonly_hides_update_and_shows_snapshot(monkeypatch):
    monkeypatch.setenv("POWER_PEAK_READONLY", "1")
    at = run_page("views/overview.py")
    assert not at.exception, [e.value for e in at.exception]
    assert not [b for b in at.button if "fetch" in (b.key or "")]           # sidebar and Overview buttons gone
    side = " ".join(m.value for m in at.sidebar.markdown)
    assert "Data snapshot:" in side and "Data updated" not in side
    assert any("Data snapshot:" in m.value for m in at.markdown)           # the badge beside the title


def test_readonly_off_keeps_the_laptop_unchanged(monkeypatch):
    monkeypatch.setenv("POWER_PEAK_READONLY", "0")
    at = run_page("views/overview.py")
    assert {"fetch_sidebar", "fetch_overview"} <= {b.key for b in at.button}
    assert "Data updated:" in " ".join(m.value for m in at.sidebar.markdown)


def test_readonly_never_starts_a_run(monkeypatch, tmp_path):
    import ui
    from src import pipeline as pl

    monkeypatch.setenv("POWER_PEAK_READONLY", "1")
    monkeypatch.setattr(pl, "start_background", lambda *a, **k: pytest.fail("must not run"))
    assert not ui.fetch_enabled() and not ui.update_running()
    ui.pipeline_panel(start=True)                                            # returns at once


def test_readonly_defaults_on_for_streamlit_cloud(monkeypatch):
    import src.config as cfg

    monkeypatch.delenv("POWER_PEAK_READONLY", raising=False)
    assert cfg.readonly() is False
    monkeypatch.setattr(cfg, "ROOT", Path("/mount/src/india-power-forecast"))
    assert cfg.readonly() is True
    monkeypatch.setenv("POWER_PEAK_READONLY", "0")                          # an explicit setting wins
    assert cfg.readonly() is False


@pytest.mark.parametrize("page", list(PAGES), ids=lambda p: Path(p).stem)
def test_every_page_footer_has_sources_notice_and_snapshot(page):
    at = run_page(page)
    foot = " ".join(m.value for m in at.markdown if "pp-footer" in m.value)
    assert "Grid-India (NLDC)" in foot and "Open-Meteo" in foot
    assert "Not an official forecast." in foot and "Data snapshot:" in foot
