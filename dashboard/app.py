"""
app.py
------
Interactive Streamlit dashboard for the AI-Powered Space Debris Collision
Prediction System.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import load_tle_file, tle_to_dataframe
from src.preprocessor import enrich_dataframe, filter_by_altitude
from src.trajectory_simulator import propagate_all
from src.collision_detector import find_close_approaches, summarize_conjunctions
from src.ml_model import build_and_train_model, predict_collision_probability
from src.visualizer import (
    plot_orbital_paths_3d_plotly,
    plot_risk_bar_chart,
    plot_probability_histogram,
    plot_conjunction_scatter,
)

DEFAULT_TLE = Path(__file__).parent.parent / "data" / "sample_tle_data.txt"

st.set_page_config(
    page_title="Space Debris Collision Prediction",
    page_icon="🛰️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

tle_path = st.sidebar.text_input(
    "TLE Data File",
    value=str(DEFAULT_TLE),
    help="Path to a TLE text file.",
)

sim_duration = st.sidebar.slider(
    "Simulation Duration (minutes)", min_value=30, max_value=360, value=90, step=30
)
step_seconds = st.sidebar.selectbox(
    "Time Step (seconds)", options=[30, 60, 120], index=1
)
threshold_km = st.sidebar.slider(
    "Close-Approach Threshold (km)", min_value=1.0, max_value=50.0, value=5.0, step=0.5
)
filter_leo = st.sidebar.checkbox("Filter to LEO only (200–2000 km)", value=True)
min_alt = st.sidebar.number_input("Min Altitude (km)", value=200.0, step=50.0)
max_alt = st.sidebar.number_input("Max Altitude (km)", value=2000.0, step=100.0)

run_button = st.sidebar.button("🚀 Run Simulation", type="primary")

# ---------------------------------------------------------------------------
# Main Title
# ---------------------------------------------------------------------------
st.title("🛰️ AI-Powered Space Debris Collision Prediction System")
st.markdown(
    """
    This dashboard simulates satellite and space debris trajectories using the
    **SGP4** astrodynamics model, detects close-approach conjunction events,
    and estimates collision probability with a **Random Forest** machine-learning
    model trained on orbital parameters.
    """
)

# ---------------------------------------------------------------------------
# Cached pipeline training (only train once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training ML model…")
def get_trained_pipeline():
    pipeline, metrics = build_and_train_model()
    return pipeline, metrics


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None

# ---------------------------------------------------------------------------
# Run Simulation
# ---------------------------------------------------------------------------
if run_button:
    pipeline, metrics = get_trained_pipeline()

    with st.spinner("Loading TLE data…"):
        try:
            tle_records = load_tle_file(tle_path)
        except FileNotFoundError:
            st.error(f"TLE file not found: `{tle_path}`")
            st.stop()

    with st.spinner("Preprocessing orbital data…"):
        raw_df = tle_to_dataframe(tle_records)
        df = enrich_dataframe(raw_df)
        if filter_leo:
            df = filter_by_altitude(df, min_alt=min_alt, max_alt=max_alt)
            tle_records = [r for r in tle_records if r["name"] in df["name"].values]

    if df.empty:
        st.warning("No objects remain after altitude filtering. Adjust filter settings.")
        st.stop()

    st.info(f"Simulating **{len(tle_records)}** objects for **{sim_duration}** minutes…")

    with st.spinner("Propagating trajectories (SGP4)…"):
        start_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        trajectories = propagate_all(
            tle_records,
            start_time=start_time,
            duration_minutes=sim_duration,
            step_seconds=step_seconds,
        )

    with st.spinner("Detecting close approaches…"):
        events_df = find_close_approaches(trajectories, threshold_km=threshold_km)

    with st.spinner("Estimating collision probabilities…"):
        if not events_df.empty:
            events_df = predict_collision_probability(pipeline, events_df, df)

    st.session_state["results"] = {
        "df": df,
        "trajectories": trajectories,
        "events_df": events_df,
        "metrics": metrics,
    }
    st.success("Simulation complete!")

# ---------------------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------------------
if st.session_state["results"]:
    res = st.session_state["results"]
    df: pd.DataFrame = res["df"]
    trajectories: dict = res["trajectories"]
    events_df: pd.DataFrame = res["events_df"]
    metrics: dict = res["metrics"]

    # ── Metrics row ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Objects Tracked", len(trajectories))
    col2.metric("Conjunction Events", len(events_df))
    if not events_df.empty and "risk_level" in events_df.columns:
        critical = (events_df["risk_level"] == "CRITICAL").sum()
        high = (events_df["risk_level"] == "HIGH").sum()
        col3.metric("🔴 CRITICAL", int(critical))
        col4.metric("🟠 HIGH Risk", int(high))
    else:
        col3.metric("🔴 CRITICAL", 0)
        col4.metric("🟠 HIGH Risk", 0)

    # ── ML Model Info ────────────────────────────────────────────────────────
    with st.expander("🤖 ML Model Performance", expanded=False):
        st.metric("ROC-AUC Score", metrics["roc_auc"])
        st.code(metrics["classification_report"], language="text")

    # ── 3-D Trajectory Plot ───────────────────────────────────────────────────
    st.subheader("🌍 Orbital Trajectories (3-D)")
    highlight = []
    if not events_df.empty:
        highlight = list(
            set(events_df["object_a"].tolist() + events_df["object_b"].tolist())
        )
    fig_3d = plot_orbital_paths_3d_plotly(trajectories, highlight_names=highlight)
    st.plotly_chart(fig_3d, use_container_width=True)

    # ── Conjunction Events Table ─────────────────────────────────────────────
    st.subheader("⚠️ Conjunction Events")
    if events_df.empty:
        st.success(f"No close approaches detected within {threshold_km} km.")
    else:
        # Summary view
        summary = summarize_conjunctions(events_df)
        st.dataframe(summary, use_container_width=True)

        # Full event table (expandable)
        with st.expander("Full Event Log"):
            display_cols = [c for c in [
                "object_a", "object_b", "time", "distance_km",
                "relative_velocity_kms", "collision_probability", "risk_level", "tca",
            ] if c in events_df.columns]
            st.dataframe(events_df[display_cols], use_container_width=True)

    # ── Risk Charts ───────────────────────────────────────────────────────────
    if not events_df.empty and "risk_level" in events_df.columns:
        st.subheader("📊 Risk Analysis Charts")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_risk_bar_chart(events_df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_probability_histogram(events_df), use_container_width=True)

        st.plotly_chart(plot_conjunction_scatter(events_df), use_container_width=True)

    # ── Orbital Parameters Table ─────────────────────────────────────────────
    st.subheader("🗂️ Orbital Parameters")
    display_orb_cols = [c for c in [
        "name", "norad_id", "inclination", "eccentricity", "mean_motion",
        "semi_major_axis", "perigee_alt", "apogee_alt", "mean_velocity",
    ] if c in df.columns]
    st.dataframe(df[display_orb_cols], use_container_width=True)

else:
    st.info("👈 Configure simulation parameters in the sidebar and click **Run Simulation**.")

    # Show placeholder info
    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📡 Data Loading\nParses TLE (Two-Line Element) files — the standard format used by NORAD and space agencies to describe satellite orbits.")
    with col2:
        st.markdown("#### 🔭 Trajectory Simulation\nUses **SGP4** orbital mechanics to propagate each object's position and velocity over time.")
    with col3:
        st.markdown("#### 🤖 AI Risk Estimation\nA **Random Forest** classifier estimates collision probability from orbital parameters and miss distance.")
