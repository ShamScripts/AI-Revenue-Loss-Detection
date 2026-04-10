import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st

from components.overview_content import render_overview
from components.sidebar import render_sidebar_stats
from components.styling import inject_css

inject_css()
render_sidebar_stats()
render_overview()
