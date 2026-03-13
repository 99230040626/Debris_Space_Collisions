"""
preprocessor.py
---------------
Module for preprocessing and feature engineering of satellite orbital data.

Converts raw orbital elements into a normalised feature matrix suitable for
machine-learning and simulation pipelines.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Earth's gravitational parameter (km³/s²)
MU_EARTH = 398600.4418
# Earth's mean radius (km)
R_EARTH = 6371.0


def compute_orbital_period(mean_motion: float) -> float:
    """
    Compute orbital period in minutes from mean motion (rev/day).

    Parameters
    ----------
    mean_motion : float
        Mean motion in revolutions per day.

    Returns
    -------
    float
        Orbital period in minutes.
    """
    if mean_motion <= 0:
        return float("inf")
    return 1440.0 / mean_motion


def compute_semi_major_axis(mean_motion: float) -> float:
    """
    Derive semi-major axis (km) from mean motion (rev/day) using Kepler's
    third law.

    Parameters
    ----------
    mean_motion : float
        Mean motion in revolutions per day.

    Returns
    -------
    float
        Semi-major axis in km.
    """
    if mean_motion <= 0:
        return float("nan")
    # Convert rev/day → rad/s
    n = mean_motion * 2.0 * np.pi / 86400.0
    return (MU_EARTH / n**2) ** (1.0 / 3.0)


def compute_altitude(semi_major_axis: float, eccentricity: float = 0.0) -> dict:
    """
    Compute perigee and apogee altitudes (km) above Earth's surface.

    Parameters
    ----------
    semi_major_axis : float
        Semi-major axis in km.
    eccentricity : float
        Orbital eccentricity (0 ≤ e < 1).

    Returns
    -------
    dict
        Dictionary with keys 'perigee_alt' and 'apogee_alt' in km.
    """
    perigee = semi_major_axis * (1.0 - eccentricity) - R_EARTH
    apogee = semi_major_axis * (1.0 + eccentricity) - R_EARTH
    return {"perigee_alt": perigee, "apogee_alt": apogee}


def compute_orbital_velocity(semi_major_axis: float) -> float:
    """
    Estimate mean orbital velocity (km/s) from semi-major axis.

    Parameters
    ----------
    semi_major_axis : float
        Semi-major axis in km.

    Returns
    -------
    float
        Approximate mean orbital velocity in km/s.
    """
    if semi_major_axis <= 0:
        return float("nan")
    return np.sqrt(MU_EARTH / semi_major_axis)


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived orbital parameters to a TLE DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by :func:`~src.data_loader.tle_to_dataframe`.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with additional columns:
        semi_major_axis, orbital_period, perigee_alt, apogee_alt,
        mean_velocity.
    """
    df = df.copy()

    df["semi_major_axis"] = df["mean_motion"].apply(compute_semi_major_axis)
    df["orbital_period"] = df["mean_motion"].apply(compute_orbital_period)

    altitudes = df.apply(
        lambda row: compute_altitude(row["semi_major_axis"], row["eccentricity"]),
        axis=1,
        result_type="expand",
    )
    df["perigee_alt"] = altitudes["perigee_alt"]
    df["apogee_alt"] = altitudes["apogee_alt"]

    df["mean_velocity"] = df["semi_major_axis"].apply(compute_orbital_velocity)

    logger.info("Enriched DataFrame with %d derived features for %d objects.", 5, len(df))
    return df


def build_feature_matrix(df: pd.DataFrame, scale: bool = True) -> tuple[np.ndarray, list[str], StandardScaler | None]:
    """
    Build a normalised numeric feature matrix from the enriched DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched orbital DataFrame.
    scale : bool
        Whether to apply :class:`~sklearn.preprocessing.StandardScaler`.

    Returns
    -------
    tuple
        (feature_matrix, feature_names, scaler_or_None)
    """
    feature_cols = [
        "inclination",
        "raan",
        "eccentricity",
        "arg_perigee",
        "mean_anomaly",
        "mean_motion",
        "bstar",
        "semi_major_axis",
        "perigee_alt",
        "apogee_alt",
        "mean_velocity",
    ]
    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].fillna(0).values.astype(np.float64)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, feature_cols, scaler


def filter_by_altitude(df: pd.DataFrame, min_alt: float = 200.0, max_alt: float = 2000.0) -> pd.DataFrame:
    """
    Filter satellites to a low-Earth orbit (LEO) altitude band.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched DataFrame (must contain 'perigee_alt' and 'apogee_alt').
    min_alt : float
        Minimum altitude threshold in km.
    max_alt : float
        Maximum altitude threshold in km.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    mask = (df["perigee_alt"] >= min_alt) & (df["apogee_alt"] <= max_alt)
    filtered = df[mask].reset_index(drop=True)
    logger.info(
        "Altitude filter [%g, %g] km: kept %d/%d objects.",
        min_alt, max_alt, len(filtered), len(df),
    )
    return filtered
