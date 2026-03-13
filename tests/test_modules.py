"""
test_modules.py
---------------
Unit tests for the AI-Powered Space Debris Collision Prediction System.

Tests cover:
 - data_loader: TLE parsing
 - preprocessor: orbital parameter derivation
 - trajectory_simulator: SGP4 propagation
 - collision_detector: close-approach detection
 - ml_model: training and prediction pipeline
"""

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_tle_file,
    tle_to_dataframe,
    _decode_tle_decimal,
    _is_tle_line,
    _parse_tle_lines,
)
from src.preprocessor import (
    compute_orbital_period,
    compute_semi_major_axis,
    compute_altitude,
    compute_orbital_velocity,
    enrich_dataframe,
    filter_by_altitude,
    build_feature_matrix,
)
from src.trajectory_simulator import propagate_satellite, propagate_all
from src.collision_detector import (
    compute_relative_distance,
    compute_relative_velocity,
    find_close_approaches,
    summarize_conjunctions,
)
from src.ml_model import (
    build_and_train_model,
    predict_collision_probability,
    _classify_risk,
    _generate_synthetic_training_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_TLE_PATH = Path(__file__).parent.parent / "data" / "sample_tle_data.txt"

# A minimal set of valid TLE lines for ISS-like orbit
ISS_LINE1 = "1 25544U 98067A   23001.50000000  .00001764  00000-0  41420-4 0  9990"
ISS_LINE2 = "2 25544  51.6435 123.4567 0002345  78.9012  281.1234 15.49815160380472"

TLE_RECORDS_SAMPLE = [
    {"name": "ISS (ZARYA)", "line1": ISS_LINE1, "line2": ISS_LINE2},
]


@pytest.fixture
def sample_tle_records():
    return load_tle_file(DEFAULT_TLE_PATH)


@pytest.fixture
def sample_df(sample_tle_records):
    raw = tle_to_dataframe(sample_tle_records)
    return enrich_dataframe(raw)


@pytest.fixture
def sample_trajectories(sample_tle_records):
    start = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return propagate_all(
        sample_tle_records,
        start_time=start,
        duration_minutes=10,
        step_seconds=60,
    )


# ---------------------------------------------------------------------------
# data_loader tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_load_tle_file_returns_list(self, sample_tle_records):
        assert isinstance(sample_tle_records, list)
        assert len(sample_tle_records) > 0

    def test_tle_record_has_required_keys(self, sample_tle_records):
        for rec in sample_tle_records:
            assert "name" in rec
            assert "line1" in rec
            assert "line2" in rec

    def test_tle_lines_correct_length(self, sample_tle_records):
        for rec in sample_tle_records:
            assert len(rec["line1"]) >= 69
            assert len(rec["line2"]) >= 69

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_tle_file("/nonexistent/path/file.txt")

    def test_tle_to_dataframe_columns(self, sample_tle_records):
        df = tle_to_dataframe(sample_tle_records)
        for col in ("name", "inclination", "eccentricity", "mean_motion"):
            assert col in df.columns

    def test_tle_to_dataframe_row_count(self, sample_tle_records):
        df = tle_to_dataframe(sample_tle_records)
        assert len(df) == len(sample_tle_records)

    def test_decode_tle_decimal_zero(self):
        assert _decode_tle_decimal("00000-0") == 0.0
        assert _decode_tle_decimal("00000+0") == 0.0

    def test_decode_tle_decimal_positive(self):
        val = _decode_tle_decimal("41420-4")
        assert abs(val - 0.41420e-4) < 1e-10

    def test_is_tle_line_valid(self):
        assert _is_tle_line(ISS_LINE1, "1")
        assert _is_tle_line(ISS_LINE2, "2")

    def test_is_tle_line_wrong_number(self):
        assert not _is_tle_line(ISS_LINE1, "2")

    def test_parse_tle_inclination_range(self, sample_tle_records):
        df = tle_to_dataframe(sample_tle_records)
        assert df["inclination"].between(0.0, 180.0).all()

    def test_parse_tle_eccentricity_range(self, sample_tle_records):
        df = tle_to_dataframe(sample_tle_records)
        assert (df["eccentricity"] >= 0.0).all()
        assert (df["eccentricity"] < 1.0).all()


# ---------------------------------------------------------------------------
# preprocessor tests
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_compute_orbital_period_iss(self):
        # ISS mean motion ≈ 15.5 rev/day → period ≈ 92.9 min
        period = compute_orbital_period(15.49815160)
        assert 90 < period < 96

    def test_compute_orbital_period_zero(self):
        assert compute_orbital_period(0) == float("inf")

    def test_compute_semi_major_axis_iss(self):
        # ISS SMA ≈ 6780 km
        sma = compute_semi_major_axis(15.49815160)
        assert 6500 < sma < 7000

    def test_compute_altitude_iss(self):
        sma = compute_semi_major_axis(15.49815160)
        alts = compute_altitude(sma, eccentricity=0.0002345)
        assert alts["perigee_alt"] > 0
        assert alts["apogee_alt"] > alts["perigee_alt"]

    def test_compute_orbital_velocity_iss(self):
        sma = compute_semi_major_axis(15.49815160)
        v = compute_orbital_velocity(sma)
        # ISS orbital velocity ≈ 7.66 km/s
        assert 7.0 < v < 8.5

    def test_enrich_dataframe_adds_columns(self, sample_df):
        for col in ("semi_major_axis", "orbital_period", "perigee_alt", "apogee_alt", "mean_velocity"):
            assert col in sample_df.columns

    def test_enrich_dataframe_positive_sma(self, sample_df):
        assert (sample_df["semi_major_axis"] > 0).all()

    def test_filter_by_altitude(self, sample_df):
        filtered = filter_by_altitude(sample_df, min_alt=300, max_alt=600)
        if not filtered.empty:
            assert (filtered["perigee_alt"] >= 300).all()
            assert (filtered["apogee_alt"] <= 600).all()

    def test_build_feature_matrix_shape(self, sample_df):
        X, cols, scaler = build_feature_matrix(sample_df)
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] == len(cols)

    def test_build_feature_matrix_scaled(self, sample_df):
        X, _, scaler = build_feature_matrix(sample_df, scale=True)
        assert scaler is not None
        # Scaled matrix should have mean ≈ 0
        assert abs(X.mean()) < 2.0

    def test_build_feature_matrix_unscaled(self, sample_df):
        X, _, scaler = build_feature_matrix(sample_df, scale=False)
        assert scaler is None


# ---------------------------------------------------------------------------
# trajectory_simulator tests
# ---------------------------------------------------------------------------

class TestTrajectorySimulator:
    def test_propagate_satellite_returns_dataframe(self):
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        traj = propagate_satellite(ISS_LINE1, ISS_LINE2, start, duration_minutes=5, step_seconds=60)
        assert isinstance(traj, pd.DataFrame)
        assert not traj.empty

    def test_propagate_satellite_columns(self):
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        traj = propagate_satellite(ISS_LINE1, ISS_LINE2, start, duration_minutes=5, step_seconds=60)
        for col in ("time", "x", "y", "z", "vx", "vy", "vz"):
            assert col in traj.columns

    def test_propagate_satellite_position_magnitude(self):
        # ISS position vector magnitude should be ~6780 km
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        traj = propagate_satellite(ISS_LINE1, ISS_LINE2, start, duration_minutes=5, step_seconds=60)
        r = np.sqrt(traj["x"]**2 + traj["y"]**2 + traj["z"]**2)
        assert r.between(6500, 7000).all()

    def test_propagate_satellite_time_steps(self):
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        traj = propagate_satellite(ISS_LINE1, ISS_LINE2, start, duration_minutes=5, step_seconds=60)
        # 5 min / 60 s = 5 steps + 1
        assert len(traj) == 6

    def test_propagate_all_returns_dict(self, sample_tle_records, sample_trajectories):
        assert isinstance(sample_trajectories, dict)
        assert len(sample_trajectories) > 0

    def test_propagate_all_nonempty_trajectories(self, sample_trajectories):
        for name, traj in sample_trajectories.items():
            assert not traj.empty, f"Empty trajectory for {name}"


# ---------------------------------------------------------------------------
# collision_detector tests
# ---------------------------------------------------------------------------

class TestCollisionDetector:
    def test_compute_relative_distance(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 0.0])
        assert abs(compute_relative_distance(p1, p2) - 5.0) < 1e-9

    def test_compute_relative_velocity(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([4.0, 0.0, 0.0])
        assert abs(compute_relative_velocity(v1, v2) - 3.0) < 1e-9

    def test_find_close_approaches_no_events(self, sample_trajectories):
        # With a very tight threshold there should be few/no events for our sample data
        events = find_close_approaches(sample_trajectories, threshold_km=0.001)
        assert isinstance(events, pd.DataFrame)

    def test_find_close_approaches_large_threshold(self, sample_trajectories):
        events = find_close_approaches(sample_trajectories, threshold_km=10000.0)
        assert isinstance(events, pd.DataFrame)
        if not events.empty:
            assert "distance_km" in events.columns

    def test_find_close_approaches_columns(self, sample_trajectories):
        events = find_close_approaches(sample_trajectories, threshold_km=10000.0)
        if not events.empty:
            for col in ("object_a", "object_b", "time", "distance_km", "relative_velocity_kms"):
                assert col in events.columns

    def test_synthesised_close_approach(self):
        """Force a close approach using identical trajectories."""
        times = pd.date_range("2023-01-01", periods=5, freq="1min")
        traj = pd.DataFrame({
            "time": times,
            "x": [7000.0] * 5,
            "y": [0.0] * 5,
            "z": [0.0] * 5,
            "vx": [0.0] * 5,
            "vy": [7.0] * 5,
            "vz": [0.0] * 5,
        })
        # Two objects at the same position
        trajs = {"SAT_A": traj.copy(), "SAT_B": traj.copy()}
        events = find_close_approaches(trajs, threshold_km=1.0)
        assert not events.empty
        assert events["distance_km"].max() < 1.0

    def test_summarize_conjunctions_empty(self):
        empty = pd.DataFrame(columns=["object_a", "object_b", "time", "distance_km",
                                      "relative_velocity_kms", "tca"])
        summary = summarize_conjunctions(empty)
        assert summary.empty

    def test_summarize_conjunctions_nonempty(self):
        times = pd.date_range("2023-01-01", periods=3, freq="1min")
        traj = pd.DataFrame({
            "time": times,
            "x": [7000.0] * 3, "y": [0.0] * 3, "z": [0.0] * 3,
            "vx": [0.0] * 3, "vy": [7.0] * 3, "vz": [0.0] * 3,
        })
        trajs = {"A": traj.copy(), "B": traj.copy()}
        events = find_close_approaches(trajs, threshold_km=1.0)
        summary = summarize_conjunctions(events)
        assert not summary.empty
        assert "min_distance_km" in summary.columns


# ---------------------------------------------------------------------------
# ml_model tests
# ---------------------------------------------------------------------------

class TestMLModel:
    def test_generate_synthetic_data_shape(self):
        df = _generate_synthetic_training_data(n_samples=100)
        assert len(df) == 100
        assert "label" in df.columns

    def test_generate_synthetic_data_binary_labels(self):
        df = _generate_synthetic_training_data(n_samples=200)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_build_and_train_model_returns_pipeline(self):
        pipeline, metrics = build_and_train_model(n_samples=500)
        assert pipeline is not None
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] > 0.5  # Should be better than random

    def test_classify_risk_levels(self):
        assert _classify_risk(0.9) == "CRITICAL"
        assert _classify_risk(0.6) == "HIGH"
        assert _classify_risk(0.3) == "MEDIUM"
        assert _classify_risk(0.1) == "LOW"

    def test_predict_collision_probability_empty(self):
        pipeline, _ = build_and_train_model(n_samples=500)
        empty_events = pd.DataFrame(columns=["object_a", "object_b", "distance_km",
                                              "relative_velocity_kms"])
        empty_orb = pd.DataFrame(columns=["name", "inclination", "eccentricity",
                                           "mean_motion", "bstar"])
        result = predict_collision_probability(pipeline, empty_events, empty_orb)
        assert result.empty

    def test_predict_collision_probability_with_data(self, sample_trajectories, sample_df):
        pipeline, _ = build_and_train_model(n_samples=500)
        events = find_close_approaches(sample_trajectories, threshold_km=10000.0)
        if events.empty:
            pytest.skip("No conjunction events to test prediction on.")
        result = predict_collision_probability(pipeline, events, sample_df)
        assert "collision_probability" in result.columns
        assert "risk_level" in result.columns
        assert result["collision_probability"].between(0.0, 1.0).all()
        assert result["risk_level"].isin(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).all()
