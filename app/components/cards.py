"""KPI and status cards."""

from __future__ import annotations

import streamlit as st


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    """items: (label, value, hint)"""
    cols = st.columns(len(items))
    for col, (label, value, hint) in zip(cols, items):
        with col:
            st.markdown(
                f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-hint">{hint}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def status_badge(ok: bool, text_ok: str = "Available", text_bad: str = "Missing") -> str:
    cls = "badge-ok" if ok else "badge-bad"
    txt = text_ok if ok else text_bad
    return f'<span class="badge {cls}">{txt}</span>'


def info_panel(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="info-panel">
  <div class="info-panel-title">{title}</div>
  <div class="info-panel-body">{body}</div>
</div>
""",
        unsafe_allow_html=True,
    )
