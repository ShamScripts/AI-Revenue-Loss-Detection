import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st

from components import file_utils as fu
from components.cards import info_panel, kpi_row
from components.styling import inject_css, section_title
from components.sidebar import render_sidebar_stats
from components.tables import preview_df

inject_css()
render_sidebar_stats()

section_title("Fusion & Final Results · Stage 4")
st.markdown(
    "Scores are **min–max normalized** per component, then fused with weights "
    "**0.45 GBDT + 0.40 DNN + 0.15 anomaly**. Threshold is tuned on **validation F1**."
)

proc = fu.processed_dir()
m_path = proc / "final_hybrid_comparison_metrics.csv"
s_path = proc / "final_hybrid_scores.csv"
t_path = proc / "final_hybrid_threshold.txt"

if t_path.is_file():
    thr = fu.safe_read_text(t_path, max_chars=200)
    st.metric("Selected threshold (validation F1)", thr.strip() if thr else "—")

metrics = fu.safe_read_csv(m_path, nrows=500)
if metrics is not None and not metrics.empty:
    section_title("Model comparison (test)")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    try:
        import plotly.express as px

        if "model" in metrics.columns and "roc_auc" in metrics.columns:
            fig = px.bar(metrics, x="model", y="roc_auc", title="ROC-AUC by model", color="model")
            fig.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        if "f1" in metrics.columns:
            fig2 = px.bar(metrics, x="model", y="f1", title="F1 by model", color="model")
            fig2.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        pass
else:
    st.warning("`final_hybrid_comparison_metrics.csv` not found — run Stage 4.")

scores = fu.safe_read_csv(s_path, nrows=5000)
if scores is not None:
    section_title("Final scores (preview)")
    preview_df(scores, max_rows=25)
    if "hybrid_weighted_score" in scores.columns and "target" in scores.columns:
        try:
            import plotly.express as px

            sample = scores.sample(min(3000, len(scores)), random_state=42)
            fig = px.scatter(
                sample,
                x="hybrid_weighted_score",
                y=sample["target"].astype(str),
                color="target",
                title="Hybrid score vs label (sample)",
                opacity=0.35,
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
else:
    st.info("No `final_hybrid_scores.csv` yet.")

section_title("Report tables (for thesis / paper)")
st.caption(
    "Generated when you run **Stage 4** — same **test split** as fusion for IEEE rows; "
    "Elliptic baselines use a separate 80/20 split on licit/illicit. Empty cells are placeholders "
    "for models not implemented in this repo (e.g. GNN, FraudGT) or values you paste from elsewhere."
)

t1 = proc / "report_table_1_ieee_cis.csv"
t2 = proc / "report_table_2_elliptic.csv"
t3 = proc / "report_table_3_ablation.csv"

for label, path in (
    ("TABLE I — Performance comparison (IEEE-CIS)", t1),
    ("TABLE II — Performance comparison (Elliptic)", t2),
    ("TABLE III — Ablation study results", t3),
):
    df_r = fu.safe_read_csv(path, nrows=50)
    if df_r is not None and not df_r.empty:
        st.markdown(f"**{label}**")
        st.dataframe(df_r, use_container_width=True, hide_index=True)
        st.caption(f"`{path.name}`")
    else:
        st.warning(f"`{path.name}` not found — run Stage 4 after Stages 2–3.")

info_panel(
    "Conclusion",
    "Use the **comparison table** and **report tables** for faculty review; download **CSV** artifacts from "
    "`processed_data/` or the Reports page.",
)
