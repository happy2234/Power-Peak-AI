#!/usr/bin/env python
"""Smoke test of the public app with ONLY requirements.txt installed (no pytest, no PyTorch):
renders the five presentation pages read-only with Streamlit's AppTest and checks the essentials.

    POWER_PEAK_READONLY=1 python scripts/smoke_app.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POWER_PEAK_READONLY", "1")
os.environ.setdefault("POWER_PEAK_MODE", "presentation")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app")]

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = {"views/overview.py": "Overview", "views/forecast.py": "7-day forecast",
         "views/raw_to_processed.py": "Raw to processed", "views/data_pipeline.py": "Data pipeline", "views/vs_grid_india.py": "Our forecast vs Grid-India",
         "views/model_lab.py": "Model lab"}
bad = []
for page, title in PAGES.items():
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180)
    at.run()
    if page != "views/overview.py":
        at.switch_page(page)
        at.run()
    md = " ".join(m.value for m in at.markdown)
    checks = {
        "no exception": not at.exception,
        "title": [t.value for t in at.title] == [title],
        "no Update data button": not [b for b in at.button if "fetch" in (b.key or "")],
        "snapshot in footer": "Not an official forecast." in md and "Data snapshot:" in md,
    }
    for name, ok in checks.items():
        if not ok:
            bad.append(f"{title}: {name}" + (f" {[e.value for e in at.exception]}" if at.exception else ""))
    print(f"{'ok  ' if all(checks.values()) else 'FAIL'} {title}")
loaded = [m for m in ("torch", "lightgbm", "shap", "sklearn") if m in sys.modules]
if loaded:
    bad.append(f"heavy libraries were imported by the app: {loaded}")
print("heavy libraries imported:", loaded or "none")
sys.exit(1 if bad else 0)
