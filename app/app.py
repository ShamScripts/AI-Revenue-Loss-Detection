"""
Fraud Detection Dashboard — entry point.

Run from project root:
  streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st

from components.overview_content import render_overview
from components.sidebar import render_sidebar_stats
from components.styling import inject_css

st.set_page_config(
    page_title="Fraud Analytics Lab",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
render_sidebar_stats()
st.sidebar.caption("Select a page below to explore each pipeline stage.")

render_overview()
