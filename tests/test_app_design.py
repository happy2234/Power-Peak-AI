"""Design rules every page must follow: no emoji anywhere in the rendered output, the shared page
structure (title, one-line description, footer with sources and update time), and the visual
system living in one place (no per-page styling, colours or HTML)."""
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from google.protobuf.json_format import MessageToDict  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.test_app import APP, PAGES, no_data, run_page  # noqa: E402

EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF"     # pictographs, emoticons, transport, symbols & pictographs ext.
    "☀-➿"              # misc symbols + dingbats (warning sign, check marks, lightning)
    "⬀-⯿"              # arrows/stars block used by emoji presentations
    "⌚⌛⏩-⏺"  # watch, hourglass, media controls
    "️‍⃣]"        # emoji variation selector, ZWJ, keycap
)


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def rendered_text(at: AppTest) -> str:
    """All text the page sent to the browser: markdown, headings, metrics, alerts, widget labels and
    options, chart specs (titles, legends, hover templates) and table cells."""
    out = []

    def walk(node):
        proto = getattr(node, "proto", None)
        if proto is not None:
            out.extend(_strings(MessageToDict(proto)))
        if type(node).__name__ == "Dataframe":
            df = node.value
            out.append(df.to_string())
            out.extend(map(str, df.columns))
        ch = getattr(node, "children", None)
        if isinstance(ch, dict):
            for c in ch.values():
                walk(c)

    walk(at._tree)
    return "\n".join(out)


@pytest.mark.parametrize("page", list(PAGES), ids=lambda p: Path(p).stem)
def test_page_has_no_emoji_and_shared_structure(page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    text = rendered_text(at)
    found = sorted(set(EMOJI.findall(text)))
    assert not found, f"emoji in {page}: {found}"
    assert len(at.title) == 1
    md = " ".join(m.value for m in at.markdown)
    assert "class='pp-head'" in md, "one-line description under the title"
    assert "Grid-India (NLDC)" in md and "Open-Meteo" in md and "Data updated" in md, "footer missing"


@pytest.mark.parametrize("page", list(PAGES), ids=lambda p: Path(p).stem)
def test_page_without_data_has_no_emoji_and_keeps_footer(page, tmp_path, monkeypatch):
    no_data(monkeypatch, tmp_path)
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not EMOJI.findall(rendered_text(at))
    md = " ".join(m.value for m in at.markdown)
    assert "Data updated" in md


def test_source_files_have_no_emoji():
    files = [*APP.rglob("*.py"), APP.parent / ".streamlit" / "config.toml"]
    bad = {f.name: EMOJI.findall(f.read_text()) for f in files if EMOJI.search(f.read_text())}
    assert not bad, bad


def test_styling_lives_only_in_the_shared_helpers():
    """Pages use ui.* and src.viz only: no inline CSS or HTML, no colour literals, no direct
    st.plotly_chart / st.dataframe / st.metric, no per-page page config."""
    for f in (APP / "views").glob("*.py"):
        src = f.read_text()
        for bad in ("<style", "st.plotly_chart(", "st.dataframe(", "st.metric(", "set_page_config",
                    "st.warning(", "st.success(", "unsafe_allow_html", "style="):
            assert bad not in src, f"{f.name} uses {bad}"
        assert not re.search(r"#[0-9A-Fa-f]{6}\b|rgba?\(\d", src), f"{f.name} hard-codes a colour"


def test_sidebar_is_grouped_and_ends_with_update_and_timestamp():
    import ui

    src = Path(ui.__file__).read_text()
    assert ui.FETCH_LABEL == "Update data" and "Data updated:" in src and "Jump to" in src
    cfg = (APP.parent / ".streamlit" / "config.toml").read_text()
    for token in ('"#FAFAFA"', '"#09090B"', '"#E4E4E7"', '"#27272A"', '"#2563EB"'):
        assert token in cfg, token


def test_css_never_hides_the_sidebar_toggle():
    """The reopen control (stExpandSidebarButton) sits inside the header toolbar: hiding the toolbar
    or collapsing the header leaves a collapsed sidebar with no way back."""
    import ui

    css = ui._CSS
    hidden = re.findall(r"([^{}]*)\{[^}]*display:\s*none", css)
    sel = " ".join(hidden)
    for keep in ("stToolbar", "stHeader", "stExpandSidebarButton", "stSidebarCollapseButton", "stSidebarCollapsedControl"):
        assert f'"{keep}"' not in sel, f"CSS hides {keep}"
    assert not re.search(r"stHeader[^{]*\{[^}]*height:\s*0", css)
    for gone in ("stAppDeployButton", "stMainMenu", "footer"):
        assert gone in sel


def test_home_starts_with_expanded_sidebar():
    src = (APP / "Home.py").read_text()
    assert 'initial_sidebar_state="expanded"' in src
