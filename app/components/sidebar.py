"""Shared sidebar (must run on every page — Streamlit does not re-run app.py on navigation)."""

from __future__ import annotations

import streamlit as st

from components import file_utils as fu
from components import stage_utils as su


def render_sidebar_stats() -> None:
    with st.sidebar:
        st.markdown(
            """
<div style="padding: 0.75rem 0 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 0.5rem;">
  <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">Fraud Analytics</div>
  <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">IEEE-CIS · Elliptic · Hybrid ML</div>
</div>
""",
            unsafe_allow_html=True,
        )
        ok, tot = su.count_artifacts_found()
        delta = f"{100 * ok // tot}% complete" if tot else None
        st.metric("Artifacts", f"{ok} / {tot}", delta=delta)
        if tot:
            st.progress(min(ok / tot, 1.0))
        with st.expander("Project root"):
            st.code(str(fu.get_project_root()), language="text")
        st.caption("Navigate via the page list above.")
        st.divider()
