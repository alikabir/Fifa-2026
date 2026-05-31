from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_DIR, MODEL_FEATURES, PROCESSED_DIR
from .features import feature_vector_from_states
from .team_names import normalize_team


def best_model_name(metrics_path: Path = MODEL_DIR / "classification_metrics.json") -> str:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return sorted(metrics, key=lambda row: (row["log_loss"], row["calibration_ece"]))[0]["model"]


class MatchPredictor:
    def __init__(self, model_name: str | None = None):
        model_name = model_name or best_model_name()
        self.model_name = model_name
        self.classifier = joblib.load(MODEL_DIR / f"{model_name}.joblib")
        self.label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
        self.xgoals_model = joblib.load(MODEL_DIR / "expected_goals.joblib")
        self.states = pd.read_parquet(PROCESSED_DIR / "team_states.parquet")

    def predict_match(self, home_team: str, away_team: str, neutral: bool = True) -> dict:
        home_team = normalize_team(home_team)
        away_team = normalize_team(away_team)
        X = feature_vector_from_states(home_team, away_team, self.states, neutral=neutral)
        probabilities = self.classifier.predict_proba(X)[0]
        labels = self.label_encoder.inverse_transform(range(len(probabilities)))
        prob_map = {label: float(prob) for label, prob in zip(labels, probabilities)}
        goals = self.xgoals_model.predict(X[MODEL_FEATURES])[0].clip(0, None)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "model": self.model_name,
            "home_win": prob_map.get("H", 0.0),
            "draw": prob_map.get("D", 0.0),
            "away_win": prob_map.get("A", 0.0),
            "home_xg": float(goals[0]),
            "away_xg": float(goals[1]),
        }
