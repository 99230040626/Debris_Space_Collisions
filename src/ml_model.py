"""
ml_model.py
-----------
Machine-learning model for estimating collision probability from conjunction
event features.

A Random Forest classifier is trained on synthetically generated conjunction
data (to act as a working demonstration).  In a production system you would
replace synthetic labels with ground-truth conjunction screening data (e.g.
from Space-Track or LeoLabs).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Feature columns used by the model
FEATURE_COLS = [
    "distance_km",
    "relative_velocity_kms",
    "inclination_a",
    "inclination_b",
    "eccentricity_a",
    "eccentricity_b",
    "mean_motion_a",
    "mean_motion_b",
    "bstar_a",
    "bstar_b",
]

# Risk probability thresholds
RISK_THRESHOLDS = {
    "CRITICAL": 0.75,
    "HIGH": 0.50,
    "MEDIUM": 0.25,
    "LOW": 0.0,
}


def _generate_synthetic_training_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate labelled synthetic conjunction data for training.

    Label = 1 (collision risk) when distance < 2 km and velocity > 5 km/s,
    with probabilistic noise to make the problem non-trivial.
    """
    rng = np.random.default_rng(random_state)

    distance = rng.exponential(scale=3.0, size=n_samples).clip(0.01, 50.0)
    rel_velocity = rng.uniform(0.1, 15.0, size=n_samples)
    incl_a = rng.uniform(0.0, 98.0, size=n_samples)
    incl_b = rng.uniform(0.0, 98.0, size=n_samples)
    ecc_a = rng.exponential(0.001, size=n_samples).clip(0, 0.3)
    ecc_b = rng.exponential(0.001, size=n_samples).clip(0, 0.3)
    mm_a = rng.uniform(12.0, 16.0, size=n_samples)
    mm_b = rng.uniform(12.0, 16.0, size=n_samples)
    bstar_a = rng.uniform(0, 1e-3, size=n_samples)
    bstar_b = rng.uniform(0, 1e-3, size=n_samples)

    # Deterministic risk score based on physical intuition
    risk_score = (
        0.6 * np.exp(-distance / 2.0)
        + 0.3 * (rel_velocity / 15.0)
        + 0.1 * rng.random(n_samples)
    )
    label = (risk_score > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "distance_km": distance,
            "relative_velocity_kms": rel_velocity,
            "inclination_a": incl_a,
            "inclination_b": incl_b,
            "eccentricity_a": ecc_a,
            "eccentricity_b": ecc_b,
            "mean_motion_a": mm_a,
            "mean_motion_b": mm_b,
            "bstar_a": bstar_a,
            "bstar_b": bstar_b,
            "label": label,
        }
    )
    return df


def build_and_train_model(
    n_samples: int = 5000,
    random_state: int = 42,
    n_estimators: int = 100,
) -> tuple[Pipeline, dict]:
    """
    Build and train a Random Forest collision risk classifier.

    Parameters
    ----------
    n_samples : int
        Number of synthetic training samples.
    random_state : int
        Random seed for reproducibility.
    n_estimators : int
        Number of trees in the Random Forest.

    Returns
    -------
    tuple[Pipeline, dict]
        (trained_pipeline, training_metrics)
    """
    data = _generate_synthetic_training_data(n_samples, random_state)
    X = data[FEATURE_COLS].values
    y = data["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=8,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "classification_report": classification_report(y_test, y_pred),
    }
    logger.info("Model trained. ROC-AUC: %.4f", metrics["roc_auc"])
    return pipeline, metrics


def predict_collision_probability(
    pipeline: Pipeline,
    events_df: pd.DataFrame,
    orbital_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate collision probability for each conjunction event.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline from :func:`build_and_train_model`.
    events_df : pd.DataFrame
        Conjunction events from
        :func:`~src.collision_detector.find_close_approaches`.
    orbital_df : pd.DataFrame
        Enriched orbital DataFrame from
        :func:`~src.preprocessor.enrich_dataframe`.

    Returns
    -------
    pd.DataFrame
        events_df with additional columns: collision_probability, risk_level.
    """
    if events_df.empty:
        return events_df.assign(collision_probability=[], risk_level=[])

    # Build a lookup by satellite name
    orb = orbital_df.set_index("name")

    rows = []
    for _, row in events_df.iterrows():
        feat = _extract_features(row, orb)
        rows.append(feat)

    feat_df = pd.DataFrame(rows, columns=FEATURE_COLS).fillna(0.0)
    probas = pipeline.predict_proba(feat_df.values)[:, 1]

    result = events_df.copy()
    result["collision_probability"] = np.round(probas, 4)
    result["risk_level"] = result["collision_probability"].apply(_classify_risk)
    return result


def _extract_features(event_row: pd.Series, orb: pd.DataFrame) -> list:
    """Extract model features for a single conjunction event row."""
    name_a = event_row["object_a"]
    name_b = event_row["object_b"]

    def safe_get(name: str, col: str) -> float:
        try:
            return float(orb.at[name, col])
        except (KeyError, ValueError):
            return 0.0

    return [
        float(event_row["distance_km"]),
        float(event_row["relative_velocity_kms"]),
        safe_get(name_a, "inclination"),
        safe_get(name_b, "inclination"),
        safe_get(name_a, "eccentricity"),
        safe_get(name_b, "eccentricity"),
        safe_get(name_a, "mean_motion"),
        safe_get(name_b, "mean_motion"),
        safe_get(name_a, "bstar"),
        safe_get(name_b, "bstar"),
    ]


def _classify_risk(probability: float) -> str:
    """Map a collision probability to a categorical risk level."""
    if probability >= RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if probability >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    if probability >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"
