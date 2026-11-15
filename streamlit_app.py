"""Entry point for hosts that look for streamlit_app.py in the repository root (Streamlit Community Cloud's
default). The app itself is app/Home.py: this file only puts app/ on the import path and runs it.

    streamlit run streamlit_app.py        # same as: streamlit run app/Home.py
"""
import runpy
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))          # the pages do `import common` / `import ui`
runpy.run_path(str(APP / "Home.py"), run_name="__main__")
