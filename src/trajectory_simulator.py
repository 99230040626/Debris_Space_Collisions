"""
trajectory_simulator.py
-----------------------
Module for propagating satellite/debris orbital trajectories using the SGP4
astrodynamics model.

SGP4 (Simplified General Perturbations 4) is the standard model used by
NORAD and space agencies to propagate TLE-based orbits.
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

logger = logging.getLogger(__name__)


def _build_satrec(line1: str, line2: str) -> Satrec:
    """Construct an SGP4 satellite record from TLE lines."""
    sat = Satrec.twoline2rv(line1, line2)
    return sat


def propagate_satellite(
    line1: str,
    line2: str,
    start_time: datetime,
    duration_minutes: float = 90.0,
    step_seconds: float = 60.0,
) -> pd.DataFrame:
    """
    Propagate a single satellite's trajectory using SGP4.

    Parameters
    ----------
    line1, line2 : str
        TLE lines for the satellite.
    start_time : datetime
        UTC start time for propagation.
    duration_minutes : float
        Total propagation time in minutes.
    step_seconds : float
        Time step between position samples in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, x, y, z (km), vx, vy, vz (km/s).
        Returns an empty DataFrame if propagation fails.
    """
    sat = _build_satrec(line1, line2)

    times: list[datetime] = []
    positions: list[tuple[float, float, float]] = []
    velocities: list[tuple[float, float, float]] = []

    current = start_time.replace(tzinfo=timezone.utc) if start_time.tzinfo is None else start_time
    n_steps = int(duration_minutes * 60 / step_seconds)

    for i in range(n_steps + 1):
        t = current + timedelta(seconds=i * step_seconds)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            logger.debug("SGP4 error code %d at step %d; skipping.", e, i)
            continue
        times.append(t)
        positions.append(r)
        velocities.append(v)

    if not times:
        logger.warning("No valid propagation steps for TLE.")
        return pd.DataFrame(columns=["time", "x", "y", "z", "vx", "vy", "vz"])

    pos_arr = np.array(positions)
    vel_arr = np.array(velocities)
    return pd.DataFrame(
        {
            "time": times,
            "x": pos_arr[:, 0],
            "y": pos_arr[:, 1],
            "z": pos_arr[:, 2],
            "vx": vel_arr[:, 0],
            "vy": vel_arr[:, 1],
            "vz": vel_arr[:, 2],
        }
    )


def propagate_all(
    tle_records: list[dict],
    start_time: datetime,
    duration_minutes: float = 90.0,
    step_seconds: float = 60.0,
) -> dict[str, pd.DataFrame]:
    """
    Propagate trajectories for all objects in *tle_records*.

    Parameters
    ----------
    tle_records : list[dict]
        List of dicts with keys 'name', 'line1', 'line2'.
    start_time : datetime
        UTC start time for propagation.
    duration_minutes : float
        Total simulation duration in minutes.
    step_seconds : float
        Time step in seconds.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from satellite name to its trajectory DataFrame.
    """
    trajectories: dict[str, pd.DataFrame] = {}
    for record in tle_records:
        name = record["name"]
        traj = propagate_satellite(
            record["line1"],
            record["line2"],
            start_time,
            duration_minutes=duration_minutes,
            step_seconds=step_seconds,
        )
        if not traj.empty:
            trajectories[name] = traj
        else:
            logger.warning("Empty trajectory for '%s'; skipping.", name)
    logger.info("Propagated %d/%d trajectories.", len(trajectories), len(tle_records))
    return trajectories


def get_position_at_time(traj: pd.DataFrame, t: datetime) -> np.ndarray | None:
    """
    Return the interpolated ECI position (km) at a given time.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory DataFrame from :func:`propagate_satellite`.
    t : datetime
        Target UTC time.

    Returns
    -------
    np.ndarray or None
        3-element position vector [x, y, z] in km, or None if out of range.
    """
    if traj.empty:
        return None
    t = t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t
    times = traj["time"].dt.tz_localize("UTC") if traj["time"].dt.tz is None else traj["time"]
    ts = times.apply(lambda dt: dt.timestamp())
    t_ts = t.timestamp()
    if t_ts < ts.iloc[0] or t_ts > ts.iloc[-1]:
        return None
    x = np.interp(t_ts, ts, traj["x"])
    y = np.interp(t_ts, ts, traj["y"])
    z = np.interp(t_ts, ts, traj["z"])
    return np.array([x, y, z])
