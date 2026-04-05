"""
CRSE — Streamlit Interactive Dashboard
=======================================

Upload two videos, run the Cognitive Response Similarity Engine,
and explore the results through interactive charts and brain maps.

Launch with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CRSE — Cognitive Response Similarity Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background-color: #0d1117; }

    .stApp {
        background: linear-gradient(160deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    }

    h1, h2, h3 { color: #f0f6fc !important; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22, #0d1117);
        border-right: 1px solid #30363d;
    }

    .metric-card {
        background: linear-gradient(135deg, #161b22, #1c2333);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(88, 166, 255, 0.1);
    }

    .score-positive { color: #7ee787; font-weight: 700; font-size: 1.6em; }
    .score-neutral  { color: #ffa657; font-weight: 700; font-size: 1.6em; }
    .score-negative { color: #f85149; font-weight: 700; font-size: 1.6em; }

    .region-badge {
        display: inline-block;
        background: rgba(88, 166, 255, 0.15);
        border: 1px solid rgba(88, 166, 255, 0.3);
        border-radius: 20px;
        padding: 4px 14px;
        margin: 4px;
        color: #58a6ff;
        font-size: 0.85em;
        font-weight: 500;
    }

    .hero-title {
        background: linear-gradient(135deg, #58a6ff, #d2a8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
    }

    .hero-subtitle {
        color: #8b949e;
        text-align: center;
        font-size: 1.1em;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────

st.markdown('<p class="hero-title">🧠 Cognitive Response Similarity Engine</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Compare neural response patterns between videos using Meta TRIBE v2</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_id = st.text_input("Model ID", value="facebook/tribev2")
    device = st.selectbox("Device", ["auto", "cuda", "cpu"])
    cache_dir = st.text_input("Cache Directory", value="./cache")

    st.markdown("---")
    st.markdown("### 🧩 Brain Regions")

    from crse.brain_regions import REGION_LABEL_PATTERNS

    all_regions = list(REGION_LABEL_PATTERNS.keys())
    selected_regions = st.multiselect(
        "Regions to analyse",
        options=all_regions,
        default=all_regions,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="color: #8b949e; font-size: 0.8em; text-align: center;">
            CRSE v0.1.0 · MIT License<br>
            TRIBE v2 · CC-BY-NC-4.0<br>
            <a href="https://github.com/facebookresearch/tribev2" style="color: #58a6ff;">
                facebookresearch/tribev2
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Video upload ───────────────────────────────────────────────────────

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🎬 Video A")
    video_a = st.file_uploader(
        "Upload first video", type=["mp4", "avi", "mkv", "mov", "webm"], key="video_a"
    )
    if video_a:
        st.video(video_a)

with col_b:
    st.markdown("### 🎬 Video B")
    video_b = st.file_uploader(
        "Upload second video", type=["mp4", "avi", "mkv", "mov", "webm"], key="video_b"
    )
    if video_b:
        st.video(video_b)

# ── Run comparison ─────────────────────────────────────────────────────

st.markdown("---")

if video_a and video_b:
    if st.button("🚀 Run Comparison", type="primary", use_container_width=True):

        # Save uploaded files to temp paths
        with tempfile.TemporaryDirectory() as tmp:
            path_a = Path(tmp) / video_a.name
            path_b = Path(tmp) / video_b.name
            path_a.write_bytes(video_a.read())
            path_b.write_bytes(video_b.read())

            with st.spinner("Loading TRIBE v2 model and running inference..."):
                from crse.engine import CRSEngine

                engine = CRSEngine(
                    model_id=model_id,
                    cache_folder=cache_dir,
                    device=device,
                    regions=selected_regions if selected_regions else None,
                )
                result = engine.compare(str(path_a), str(path_b))

        st.success(f"✅ Comparison complete in {result.elapsed_seconds:.1f}s")
        st.markdown("---")

        # ── Whole-brain metrics ────────────────────────────────────────

        st.markdown("## 🌐 Whole-Brain Similarity")
        metric_cols = st.columns(len(result.whole_brain))
        for col, (metric, score) in zip(metric_cols, result.whole_brain.items()):
            with col:
                score_class = (
                    "score-positive" if score >= 0.3
                    else ("score-neutral" if score >= 0 else "score-negative")
                )
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="color: #8b949e; font-size: 0.85em; text-transform: uppercase;">
                            {metric.replace('_', ' ')}
                        </div>
                        <div class="{score_class}">{score:+.4f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Per-region radar chart ─────────────────────────────────────

        if result.regions:
            st.markdown("---")
            st.markdown("## 🧩 Per-Region Analysis")

            # Interactive Plotly radar
            try:
                import plotly.graph_objects as go

                radar_metric = st.selectbox(
                    "Select metric for radar chart",
                    options=list(result.regions[0].metrics.keys()),
                    index=1,  # default to pearson
                )

                categories = [r.name.replace("_", " ").title() for r in result.regions]
                values = [r.metrics.get(radar_metric, 0) for r in result.regions]
                values += values[:1]  # close polygon
                categories += categories[:1]

                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill="toself",
                    fillcolor="rgba(88, 166, 255, 0.15)",
                    line=dict(color="#58a6ff", width=2),
                    marker=dict(size=8, color="#58a6ff"),
                ))
                fig.update_layout(
                    polar=dict(
                        bgcolor="#0d1117",
                        radialaxis=dict(
                            visible=True, range=[-1, 1],
                            gridcolor="#30363d", color="#8b949e",
                        ),
                        angularaxis=dict(gridcolor="#30363d", color="#c9d1d9"),
                    ),
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9"),
                    title=dict(
                        text=f"Neural Response Similarity — {radar_metric.replace('_', ' ').title()}",
                        font=dict(color="#f0f6fc", size=16),
                    ),
                    margin=dict(t=60, b=40),
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

            except ImportError:
                st.warning("Install `plotly` for interactive charts: `pip install plotly`")

            # Region detail cards
            for region in result.regions:
                with st.expander(f"📍 {region.name.replace('_', ' ').title()} — {region.n_vertices:,} vertices"):
                    st.markdown(f"*{region.description}*")
                    r_cols = st.columns(len(region.metrics))
                    for rc, (m, s) in zip(r_cols, region.metrics.items()):
                        rc.metric(m.replace("_", " ").title(), f"{s:+.4f}")

        # ── Download results ───────────────────────────────────────────

        st.markdown("---")
        st.download_button(
            label="📥 Download Results (JSON)",
            data=result.to_json(),
            file_name="crse_results.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    st.info("👆 Upload two videos above to begin the comparison.")
