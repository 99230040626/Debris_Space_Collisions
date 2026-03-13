"""
visualizer.py
-------------
Visualization module for orbital trajectories and collision risk data.

Provides both static Matplotlib figures and interactive Plotly charts.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

# Risk level color map
RISK_COLORS = {
    "CRITICAL": "#FF0000",
    "HIGH": "#FF6600",
    "MEDIUM": "#FFAA00",
    "LOW": "#00AA00",
}


# ---------------------------------------------------------------------------
# Static Matplotlib helpers
# ---------------------------------------------------------------------------

def plot_orbital_paths_3d_mpl(
    trajectories: dict[str, pd.DataFrame],
    highlight_names: Optional[list[str]] = None,
    title: str = "Orbital Trajectories (ECI Frame)",
) -> plt.Figure:
    """
    Plot 3-D orbital trajectories in the Earth-Centered Inertial (ECI) frame
    using Matplotlib.

    Parameters
    ----------
    trajectories : dict[str, pd.DataFrame]
        Mapping from object name to trajectory DataFrame.
    highlight_names : list[str], optional
        Names of objects to highlight in red.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    highlight_names = highlight_names or []
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw Earth as a sphere
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    R = 6371  # km
    ax.plot_surface(
        R * np.cos(u) * np.sin(v),
        R * np.sin(u) * np.sin(v),
        R * np.cos(v),
        color="deepskyblue",
        alpha=0.3,
        linewidth=0,
    )

    for name, traj in trajectories.items():
        colour = "red" if name in highlight_names else "steelblue"
        lw = 1.5 if name in highlight_names else 0.8
        ax.plot(traj["x"], traj["y"], traj["z"], color=colour, lw=lw, label=name)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title(title)

    if len(trajectories) <= 10:
        ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    return fig


def plot_distance_over_time(
    traj_a: pd.DataFrame,
    traj_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    threshold_km: float = 5.0,
) -> plt.Figure:
    """
    Plot the inter-object distance over time for a specific object pair.

    Parameters
    ----------
    traj_a, traj_b : pd.DataFrame
        Trajectory DataFrames.
    name_a, name_b : str
        Object names (for labelling).
    threshold_km : float
        Danger threshold line.

    Returns
    -------
    matplotlib.figure.Figure
    """
    merged = pd.merge(
        traj_a.rename(columns=lambda c: c + "_a" if c != "time" else c),
        traj_b.rename(columns=lambda c: c + "_b" if c != "time" else c),
        on="time",
    )
    if merged.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No overlapping time steps", ha="center", transform=ax.transAxes)
        return fig

    pos_a = merged[["x_a", "y_a", "z_a"]].values
    pos_b = merged[["x_b", "y_b", "z_b"]].values
    distances = np.linalg.norm(pos_a - pos_b, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(merged["time"], distances, color="steelblue", label="Distance (km)")
    ax.axhline(threshold_km, color="red", linestyle="--", label=f"Threshold ({threshold_km} km)")
    ax.fill_between(merged["time"], 0, distances, where=distances < threshold_km,
                    color="red", alpha=0.2, label="Danger zone")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Distance (km)")
    ax.set_title(f"Distance Over Time: {name_a} ↔ {name_b}")
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interactive Plotly helpers
# ---------------------------------------------------------------------------

def plot_orbital_paths_3d_plotly(
    trajectories: dict[str, pd.DataFrame],
    highlight_names: Optional[list[str]] = None,
    title: str = "Orbital Trajectories (ECI Frame)",
) -> go.Figure:
    """
    Create an interactive 3-D orbital path plot using Plotly.

    Parameters
    ----------
    trajectories : dict[str, pd.DataFrame]
        Mapping from object name to trajectory DataFrame.
    highlight_names : list[str], optional
        Objects to render with a thicker, distinctive trace.
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    highlight_names = highlight_names or []
    fig = go.Figure()

    # Earth sphere
    R = 6371
    theta = np.linspace(0, 2 * np.pi, 60)
    phi = np.linspace(0, np.pi, 30)
    x_s = R * np.outer(np.cos(theta), np.sin(phi))
    y_s = R * np.outer(np.sin(theta), np.sin(phi))
    z_s = R * np.outer(np.ones(60), np.cos(phi))
    fig.add_trace(
        go.Surface(
            x=x_s, y=y_s, z=z_s,
            colorscale="Blues",
            opacity=0.4,
            showscale=False,
            name="Earth",
        )
    )

    for name, traj in trajectories.items():
        is_highlight = name in highlight_names
        fig.add_trace(
            go.Scatter3d(
                x=traj["x"],
                y=traj["y"],
                z=traj["z"],
                mode="lines",
                name=name,
                line=dict(
                    width=4 if is_highlight else 2,
                    color="red" if is_highlight else None,
                ),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "x: %{x:.1f} km<br>"
                    "y: %{y:.1f} km<br>"
                    "z: %{z:.1f} km<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0, y=1),
    )
    return fig


def plot_risk_bar_chart(events_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of collision events grouped by risk level.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame with a 'risk_level' column.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if "risk_level" not in events_df.columns or events_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No risk data available")
        return fig

    counts = events_df["risk_level"].value_counts().reindex(
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"], fill_value=0
    )
    colours = [RISK_COLORS.get(lvl, "grey") for lvl in counts.index]

    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=colours,
            text=counts.values,
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Conjunction Events by Risk Level",
        xaxis_title="Risk Level",
        yaxis_title="Number of Events",
        showlegend=False,
    )
    return fig


def plot_probability_histogram(events_df: pd.DataFrame) -> go.Figure:
    """
    Histogram of collision probabilities across all conjunction events.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame with a 'collision_probability' column.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if "collision_probability" not in events_df.columns or events_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No probability data available")
        return fig

    fig = px.histogram(
        events_df,
        x="collision_probability",
        nbins=20,
        color_discrete_sequence=["steelblue"],
        title="Distribution of Collision Probabilities",
        labels={"collision_probability": "Collision Probability"},
    )
    fig.update_layout(yaxis_title="Count")
    return fig


def plot_conjunction_scatter(events_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of miss distance vs. relative velocity coloured by risk level.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame with 'distance_km', 'relative_velocity_kms', and
        'risk_level' columns.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    required = {"distance_km", "relative_velocity_kms", "risk_level"}
    if not required.issubset(events_df.columns) or events_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No conjunction data available")
        return fig

    colour_map = RISK_COLORS
    fig = px.scatter(
        events_df,
        x="distance_km",
        y="relative_velocity_kms",
        color="risk_level",
        color_discrete_map=colour_map,
        hover_data=["object_a", "object_b", "collision_probability"],
        title="Conjunction Events: Miss Distance vs. Relative Velocity",
        labels={
            "distance_km": "Miss Distance (km)",
            "relative_velocity_kms": "Relative Velocity (km/s)",
        },
    )
    return fig
