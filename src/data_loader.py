"""
data_loader.py
--------------
Module for loading and parsing TLE (Two-Line Element) orbital data.

TLE format reference: https://celestrak.org/columns/v04n03/
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TLE_PATH = Path(__file__).parent.parent / "data" / "sample_tle_data.txt"


def load_tle_file(filepath: str | Path = DEFAULT_TLE_PATH) -> list[dict]:
    """
    Parse a TLE file and return a list of satellite dictionaries.

    Each entry contains:
        - name: satellite/object name
        - line1: TLE line 1
        - line2: TLE line 2

    Parameters
    ----------
    filepath : str or Path
        Path to the TLE text file.

    Returns
    -------
    list[dict]
        Parsed TLE records.

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    ValueError
        If the TLE file contains malformed entries.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TLE file not found: {filepath}")

    entries: list[dict] = []
    with filepath.open("r") as fh:
        lines = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

    i = 0
    while i < len(lines):
        # Expect groups of three: name, line1, line2
        if i + 2 >= len(lines):
            logger.warning("Incomplete TLE entry at end of file; skipping.")
            break

        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        if not _is_tle_line(line1, expected_num="1") or not _is_tle_line(line2, expected_num="2"):
            # Skip malformed block and try next line
            logger.warning("Skipping malformed TLE block near line %d: '%s'", i, name)
            i += 1
            continue

        entries.append({"name": name, "line1": line1, "line2": line2})
        i += 3

    logger.info("Loaded %d TLE records from %s", len(entries), filepath)
    return entries


def tle_to_dataframe(tle_records: list[dict]) -> pd.DataFrame:
    """
    Convert a list of TLE records into a pandas DataFrame with parsed orbital
    parameters extracted from the TLE lines.

    Parameters
    ----------
    tle_records : list[dict]
        Output from :func:`load_tle_file`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: name, norad_id, inclination, raan,
        eccentricity, arg_perigee, mean_anomaly, mean_motion,
        bstar, epoch_year, epoch_day, line1, line2.
    """
    rows = []
    for record in tle_records:
        parsed = _parse_tle_lines(record["line1"], record["line2"])
        parsed["name"] = record["name"]
        parsed["line1"] = record["line1"]
        parsed["line2"] = record["line2"]
        rows.append(parsed)

    df = pd.DataFrame(rows)
    # Reorder columns for readability
    cols_order = [
        "name", "norad_id", "epoch_year", "epoch_day",
        "inclination", "raan", "eccentricity", "arg_perigee",
        "mean_anomaly", "mean_motion", "bstar", "line1", "line2",
    ]
    cols_order = [c for c in cols_order if c in df.columns]
    return df[cols_order]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_tle_line(line: str, expected_num: str) -> bool:
    """Return True if *line* looks like a TLE line with the given line number."""
    return len(line) >= 69 and line[0] == expected_num


def _parse_tle_lines(line1: str, line2: str) -> dict:
    """
    Extract orbital elements from raw TLE lines.

    TLE field positions follow the standard format documented at
    https://celestrak.org/columns/v04n03/.
    """
    try:
        norad_id = int(line1[2:7].strip())
        epoch_year = int(line1[18:20].strip())
        epoch_day = float(line1[20:32].strip())
        bstar_str = line1[53:61].strip()
        bstar = _decode_tle_decimal(bstar_str)

        inclination = float(line2[8:16].strip())
        raan = float(line2[17:25].strip())
        eccentricity = float("0." + line2[26:33].strip())
        arg_perigee = float(line2[34:42].strip())
        mean_anomaly = float(line2[43:51].strip())
        mean_motion = float(line2[52:63].strip())
    except (ValueError, IndexError) as exc:
        logger.error("Failed to parse TLE lines: %s", exc)
        return {}

    return {
        "norad_id": norad_id,
        "epoch_year": epoch_year,
        "epoch_day": epoch_day,
        "bstar": bstar,
        "inclination": inclination,
        "raan": raan,
        "eccentricity": eccentricity,
        "arg_perigee": arg_perigee,
        "mean_anomaly": mean_anomaly,
        "mean_motion": mean_motion,
    }


def _decode_tle_decimal(s: str) -> float:
    """
    Decode TLE assumed-decimal notation (e.g. '41420-4' → 0.41420e-4).
    """
    s = s.strip()
    if not s or s in ("00000-0", "00000+0"):
        return 0.0
    # Find the exponent part (last occurrence of + or -)
    for i in range(len(s) - 1, 0, -1):
        if s[i] in ("+", "-"):
            mantissa = float("0." + s[:i])
            exp = int(s[i:])
            return mantissa * (10 ** exp)
    return float(s)
