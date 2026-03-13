"""
main.py
-------
Entry-point script for the AI-Powered Space Debris Collision Prediction System.

Runs the full pipeline end-to-end:
  1. Load TLE data
  2. Preprocess and enrich orbital parameters
  3. Simulate trajectories (SGP4)
  4. Detect close-approach (conjunction) events
  5. Estimate collision probability via ML
  6. Display a summary report

Usage:
    python main.py [--tle PATH] [--duration MINUTES] [--step SECONDS]
                   [--threshold KM] [--plot]
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on the path when executed directly
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.data_loader import load_tle_file, tle_to_dataframe
from src.preprocessor import enrich_dataframe, filter_by_altitude
from src.trajectory_simulator import propagate_all
from src.collision_detector import find_close_approaches, summarize_conjunctions
from src.ml_model import build_and_train_model, predict_collision_probability
from src.visualizer import (
    plot_orbital_paths_3d_mpl,
    plot_conjunction_scatter,
    plot_risk_bar_chart,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_TLE = Path(__file__).parent / "data" / "sample_tle_data.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Powered Space Debris Collision Prediction System"
    )
    parser.add_argument("--tle", default=str(DEFAULT_TLE), help="Path to TLE data file")
    parser.add_argument("--duration", type=float, default=90.0, help="Simulation duration (minutes)")
    parser.add_argument("--step", type=float, default=60.0, help="Time step (seconds)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Close-approach threshold (km)")
    parser.add_argument("--min-alt", type=float, default=200.0, help="Min altitude filter (km)")
    parser.add_argument("--max-alt", type=float, default=2000.0, help="Max altitude filter (km)")
    parser.add_argument("--no-filter", action="store_true", help="Disable altitude filter")
    parser.add_argument("--plot", action="store_true", help="Display matplotlib plots")
    return parser.parse_args()


def run_pipeline(
    tle_path: str | Path,
    duration_minutes: float,
    step_seconds: float,
    threshold_km: float,
    min_alt: float,
    max_alt: float,
    apply_filter: bool,
    show_plot: bool,
) -> dict:
    """Execute the full prediction pipeline and return result dictionary."""

    # 1. Load data ────────────────────────────────────────────────────────────
    logger.info("Loading TLE data from %s", tle_path)
    tle_records = load_tle_file(tle_path)
    raw_df = tle_to_dataframe(tle_records)

    # 2. Preprocess ───────────────────────────────────────────────────────────
    logger.info("Enriching orbital parameters…")
    df = enrich_dataframe(raw_df)

    if apply_filter:
        logger.info("Applying altitude filter [%g, %g] km…", min_alt, max_alt)
        df = filter_by_altitude(df, min_alt=min_alt, max_alt=max_alt)
        tle_records = [r for r in tle_records if r["name"] in df["name"].values]

    if df.empty:
        logger.warning("No objects remain after filtering.")
        return {}

    # 3. Simulate trajectories ────────────────────────────────────────────────
    logger.info("Propagating %d trajectories (duration=%g min, step=%g s)…",
                len(tle_records), duration_minutes, step_seconds)
    start_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    trajectories = propagate_all(
        tle_records,
        start_time=start_time,
        duration_minutes=duration_minutes,
        step_seconds=step_seconds,
    )

    # 4. Detect conjunctions ──────────────────────────────────────────────────
    logger.info("Detecting close approaches (threshold=%.1f km)…", threshold_km)
    events_df = find_close_approaches(trajectories, threshold_km=threshold_km)

    # 5. ML collision probability ─────────────────────────────────────────────
    logger.info("Training ML model and estimating collision probabilities…")
    pipeline, metrics = build_and_train_model()
    logger.info("Model ROC-AUC: %.4f", metrics["roc_auc"])

    if not events_df.empty:
        events_df = predict_collision_probability(pipeline, events_df, df)

    # 6. Report ───────────────────────────────────────────────────────────────
    _print_summary(df, trajectories, events_df, metrics)

    # 7. Optional plots ───────────────────────────────────────────────────────
    if show_plot:
        import matplotlib.pyplot as plt
        highlight = []
        if not events_df.empty:
            highlight = list(
                set(events_df["object_a"].tolist() + events_df["object_b"].tolist())
            )
        fig = plot_orbital_paths_3d_mpl(trajectories, highlight_names=highlight)
        plt.show()

    return {
        "orbital_df": df,
        "trajectories": trajectories,
        "events_df": events_df,
        "metrics": metrics,
    }


def _print_summary(
    df: pd.DataFrame,
    trajectories: dict,
    events_df: pd.DataFrame,
    metrics: dict,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  AI-Powered Space Debris Collision Prediction System")
    print(sep)
    print(f"  Objects simulated  : {len(trajectories)}")
    print(f"  Conjunction events : {len(events_df)}")
    print(f"  ML ROC-AUC score   : {metrics['roc_auc']:.4f}")
    print(sep)

    if events_df.empty:
        print("  ✅ No close approaches detected.")
    else:
        summary = summarize_conjunctions(events_df)
        print(f"\n  Top {min(5, len(summary))} closest approaches:")
        top = summary.head(5)[["object_a", "object_b", "min_distance_km", "max_rel_velocity_kms"]]
        print(top.to_string(index=False))

        if "risk_level" in events_df.columns:
            print("\n  Risk level breakdown:")
            print(events_df["risk_level"].value_counts().to_string())
    print()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        tle_path=args.tle,
        duration_minutes=args.duration,
        step_seconds=args.step,
        threshold_km=args.threshold,
        min_alt=args.min_alt,
        max_alt=args.max_alt,
        apply_filter=not args.no_filter,
        show_plot=args.plot,
    )
