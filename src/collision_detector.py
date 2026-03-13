"""
collision_detector.py
---------------------
Module for detecting close-approach events between satellites and space debris.

A close approach (conjunction) is flagged when two objects come within a
user-defined miss distance threshold.
"""

import logging
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default miss-distance threshold (km) — corresponds to ~1 km safety bubble
DEFAULT_THRESHOLD_KM = 5.0


def compute_relative_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Compute Euclidean distance (km) between two ECI position vectors.

    Parameters
    ----------
    pos1, pos2 : np.ndarray
        3-element position vectors in km.

    Returns
    -------
    float
        Distance in km.
    """
    return float(np.linalg.norm(pos1 - pos2))


def compute_relative_velocity(vel1: np.ndarray, vel2: np.ndarray) -> float:
    """
    Compute the magnitude of relative velocity (km/s) between two objects.

    Parameters
    ----------
    vel1, vel2 : np.ndarray
        3-element velocity vectors in km/s.

    Returns
    -------
    float
        Relative speed in km/s.
    """
    return float(np.linalg.norm(vel1 - vel2))


def find_close_approaches(
    trajectories: dict[str, pd.DataFrame],
    threshold_km: float = DEFAULT_THRESHOLD_KM,
) -> pd.DataFrame:
    """
    Scan all pairwise combinations of trajectories for close-approach events.

    Two objects are considered to have a conjunction when their separation
    drops below *threshold_km* at any shared time step.

    Parameters
    ----------
    trajectories : dict[str, pd.DataFrame]
        Mapping from object name to trajectory DataFrame (output of
        :func:`~src.trajectory_simulator.propagate_all`).
    threshold_km : float
        Miss-distance threshold in km.

    Returns
    -------
    pd.DataFrame
        DataFrame of conjunction events with columns:
        object_a, object_b, time, distance_km, relative_velocity_kms,
        tca (time of closest approach flag).
    """
    names = list(trajectories.keys())
    events: list[dict] = []

    for name_a, name_b in combinations(names, 2):
        traj_a = trajectories[name_a]
        traj_b = trajectories[name_b]

        # Align on common timestamps
        merged = pd.merge(
            traj_a.rename(columns=lambda c: c + "_a" if c != "time" else c),
            traj_b.rename(columns=lambda c: c + "_b" if c != "time" else c),
            on="time",
        )
        if merged.empty:
            continue

        pos_a = merged[["x_a", "y_a", "z_a"]].values
        pos_b = merged[["x_b", "y_b", "z_b"]].values
        vel_a = merged[["vx_a", "vy_a", "vz_a"]].values
        vel_b = merged[["vx_b", "vy_b", "vz_b"]].values

        distances = np.linalg.norm(pos_a - pos_b, axis=1)
        rel_speeds = np.linalg.norm(vel_a - vel_b, axis=1)

        close_mask = distances < threshold_km
        if not close_mask.any():
            continue

        # Find the time of closest approach (TCA) within this pair
        min_idx = int(np.argmin(distances))

        for idx in np.where(close_mask)[0]:
            events.append(
                {
                    "object_a": name_a,
                    "object_b": name_b,
                    "time": merged["time"].iloc[idx],
                    "distance_km": round(float(distances[idx]), 4),
                    "relative_velocity_kms": round(float(rel_speeds[idx]), 4),
                    "tca": bool(idx == min_idx),
                }
            )

    result = pd.DataFrame(events)
    if result.empty:
        logger.info("No close approaches found within %.1f km threshold.", threshold_km)
    else:
        logger.info(
            "Found %d close-approach events between %d object pairs.",
            len(result),
            result.groupby(["object_a", "object_b"]).ngroups,
        )
    return result


def summarize_conjunctions(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize close-approach events by object pair.

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of :func:`find_close_approaches`.

    Returns
    -------
    pd.DataFrame
        One row per object pair with: object_a, object_b, n_events,
        min_distance_km, max_rel_velocity_kms, tca_time.
    """
    if events_df.empty:
        return pd.DataFrame(
            columns=["object_a", "object_b", "n_events", "min_distance_km",
                     "max_rel_velocity_kms", "tca_time"]
        )

    tca_df = events_df[events_df["tca"]].rename(columns={"time": "tca_time"})

    summary = (
        events_df.groupby(["object_a", "object_b"])
        .agg(
            n_events=("distance_km", "count"),
            min_distance_km=("distance_km", "min"),
            max_rel_velocity_kms=("relative_velocity_kms", "max"),
        )
        .reset_index()
    )
    summary = summary.merge(
        tca_df[["object_a", "object_b", "tca_time"]],
        on=["object_a", "object_b"],
        how="left",
    )
    return summary.sort_values("min_distance_km").reset_index(drop=True)
