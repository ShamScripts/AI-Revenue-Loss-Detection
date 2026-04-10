"""Plotly charts for dashboard metrics."""

from __future__ import annotations

import streamlit as st
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def bar_metrics(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if not HAS_PLOTLY or df is None or df.empty or x not in df.columns or y not in df.columns:
        st.info("Install plotly or add data to show this chart.")
        return
    fig = px.bar(df, x=x, y=y, title=title, color=color, template="plotly_white")
    fig.update_layout(font_family="Inter, system-ui", margin=dict(t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


def line_series(x, y1, y2, name1: str, name2: str, title: str) -> None:
    if not HAS_PLOTLY:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, name=name1, mode="lines"))
    fig.add_trace(go.Scatter(x=x, y=y2, name=name2, mode="lines"))
    fig.update_layout(title=title, template="plotly_white", font_family="Inter, system-ui")
    st.plotly_chart(fig, use_container_width=True)


def hist_series(s: pd.Series, title: str, bins: int = 40) -> None:
    if not HAS_PLOTLY or s is None or s.empty:
        return
    fig = px.histogram(s.dropna(), nbins=bins, title=title, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
