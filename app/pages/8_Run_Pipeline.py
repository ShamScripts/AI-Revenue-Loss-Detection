import subprocess
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st

from components import file_utils as fu
from components.sidebar import render_sidebar_stats
from components.styling import inject_css, section_title

inject_css()
render_sidebar_stats()

section_title("Run pipeline (optional)")
root = fu.get_project_root()
main_py = fu.main_py_path()

st.markdown(
    f"Executes the project **`main.py`** (same as CLI) with working directory:  \n`{root}`"
)

if not main_py.is_file():
    st.error(f"Cannot find main.py at `{main_py}`")
    st.stop()

_labels = ["All (1–4)", "1 — Data", "2 — GBDT", "3 — Deep + Anomaly", "4 — Fusion"]
_choice = st.selectbox("Stage", _labels, index=0)
_stage_val: int | None = {"All (1–4)": None, "1 — Data": 1, "2 — GBDT": 2, "3 — Deep + Anomaly": 3, "4 — Fusion": 4}[_choice]

no_plots = st.checkbox("Skip saving figures (--no-plots)", value=False)

st.warning("**Long-running.** Training may take tens of minutes to hours. Keep this tab open.")

if st.button("▶ Run pipeline", type="primary"):
    cmd = [sys.executable, str(main_py)]
    if _stage_val is not None:
        cmd.extend(["--stage", str(_stage_val)])
    if no_plots:
        cmd.append("--no-plots")
    with st.spinner("Running…"):
        try:
            r = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=86400,
                shell=False,
            )
            st.code((r.stdout or "") + "\n" + (r.stderr or ""), language="text")
            if r.returncode == 0:
                st.success("Finished successfully.")
            else:
                st.error(f"Exit code {r.returncode}")
        except subprocess.TimeoutExpired:
            st.error("Timed out.")
        except Exception as e:
            st.exception(e)

with st.expander("Equivalent shell command"):
    line = f'cd "{root}"\npython main.py'
    if _stage_val is not None:
        line += f" --stage {_stage_val}"
    if no_plots:
        line += " --no-plots"
    st.code(line, language="text")
