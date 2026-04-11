import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st

from components import file_utils as fu
from components.figure_captions import caption_for_figure
from components.sidebar import render_sidebar_stats
from components.styling import inject_css, section_title

inject_css()
render_sidebar_stats()

section_title("Exploratory Data Analysis")
st.markdown(
    "Figures are generated when you run the pipeline **without** `--no-plots`. "
    "They live under **`figures/`** (project root). "
    "Each plot below includes a **short caption** for presentations and reports."
)

fig_root = fu.figures_dir()
if not fig_root.is_dir():
    fig_root.mkdir(parents=True, exist_ok=True)

images = fu.list_images_recursive(fig_root)

seen = set()
uniq: list[Path] = []
for p in images:
    rp = str(p.resolve())
    if rp not in seen:
        seen.add(rp)
        uniq.append(p)
images = uniq

if not images:
    st.warning(
        f"No figures found under `{fu.figures_dir()}`. "
        "Run `python main.py` (with plots enabled) to generate PNGs."
    )
else:
    st.success(f"**{len(images)}** figure(s) discovered. Showing up to **24** in reading order.")
    for img_path in images[:24]:
        title, explain = caption_for_figure(img_path)
        st.markdown("---")
        st.markdown(f"##### {title}")
        st.caption(explain)
        try:
            st.image(str(img_path), use_container_width=True)
        except Exception as e:
            st.error(f"{img_path.name}: {e}")
        st.caption(f"`{img_path.name}` · {img_path.parent.name}/")

    if len(images) > 24:
        st.info(
            f"**{len(images) - 24}** additional file(s) not shown here — open `figures/` on disk for the full set."
        )

with st.expander("Figure naming & scan paths"):
    st.markdown(
        """
- **Stage 1** figures use the prefix `stage01_` (IEEE + Elliptic EDA).
- **Stages 2–4** use `stage02_` … `stage04_` for model and fusion plots.
- Unknown filenames still appear with a **generic** explanation; you can extend `app/components/figure_captions.py` for custom names.
"""
    )
    st.code(str(fu.figures_dir()), language="text")
