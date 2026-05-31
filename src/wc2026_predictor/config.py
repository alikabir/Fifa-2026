from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TOURNAMENT_DIR = DATA_DIR / "tournament"
MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
)

RANDOM_STATE = 42
MODEL_FEATURES = [
    "elo_diff",
    "fifa_rank_diff",
    "home_goals_for_last5",
    "home_goals_against_last5",
    "away_goals_for_last5",
    "away_goals_against_last5",
    "home_win_rate_last10",
    "away_win_rate_last10",
    "home_form_points_last5",
    "away_form_points_last5",
    "is_neutral",
    "is_home_advantage",
]

CLASS_LABELS = ["A", "D", "H"]
