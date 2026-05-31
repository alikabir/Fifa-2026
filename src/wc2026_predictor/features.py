from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MODEL_FEATURES, PROCESSED_DIR


@dataclass
class TeamState:
    elo: float = 1500.0
    goals_for: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    goals_against: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    wins: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    points: deque[int] = field(default_factory=lambda: deque(maxlen=10))


def _mean_recent(values: deque[int], n: int) -> float:
    if not values:
        return 0.0
    return float(np.mean(list(values)[-n:]))


def _sum_recent(values: deque[int], n: int) -> float:
    if not values:
        return 0.0
    return float(np.sum(list(values)[-n:]))


def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo_a - elo_b) / 400.0))


def _elo_k(row: pd.Series) -> float:
    if bool(row.get("is_world_cup", False)):
        return 45.0
    if bool(row.get("is_qualifier", False)) or bool(row.get("is_continental", False)):
        return 35.0
    return 20.0


def _goal_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    return np.log(goal_diff + 1.0)


def _update_team(state: TeamState, gf: int, ga: int) -> None:
    state.goals_for.append(gf)
    state.goals_against.append(ga)
    state.wins.append(int(gf > ga))
    state.points.append(3 if gf > ga else 1 if gf == ga else 0)


def build_feature_table(matches: pd.DataFrame, output_path=PROCESSED_DIR / "features.parquet") -> pd.DataFrame:
    """Create pre-match features by walking matches chronologically."""
    states: dict[str, TeamState] = defaultdict(TeamState)
    rows: list[dict] = []
    matches = matches.sort_values("date").reset_index(drop=True)

    for _, match in matches.iterrows():
        home = states[match.home_team]
        away = states[match.away_team]
        fifa_rank_diff = np.nan
        if pd.notna(match.home_fifa_rank) and pd.notna(match.away_fifa_rank):
            fifa_rank_diff = float(match.home_fifa_rank) - float(match.away_fifa_rank)

        row = match.to_dict()
        row.update(
            {
                "elo_home": home.elo,
                "elo_away": away.elo,
                "elo_diff": home.elo - away.elo,
                "fifa_rank_diff": fifa_rank_diff,
                "home_goals_for_last5": _sum_recent(home.goals_for, 5),
                "home_goals_against_last5": _sum_recent(home.goals_against, 5),
                "away_goals_for_last5": _sum_recent(away.goals_for, 5),
                "away_goals_against_last5": _sum_recent(away.goals_against, 5),
                "home_win_rate_last10": _mean_recent(home.wins, 10),
                "away_win_rate_last10": _mean_recent(away.wins, 10),
                "home_form_points_last5": _sum_recent(home.points, 5),
                "away_form_points_last5": _sum_recent(away.points, 5),
                "is_neutral": int(bool(match.neutral)),
                "is_home_advantage": int(not bool(match.neutral)),
            }
        )
        rows.append(row)

        home_score = int(match.home_score)
        away_score = int(match.away_score)
        actual_home = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
        home_for_elo = home.elo + (45.0 if not bool(match.neutral) else 0.0)
        expected_home = _expected_score(home_for_elo, away.elo)
        change = _elo_k(match) * _goal_multiplier(abs(home_score - away_score)) * (actual_home - expected_home)
        home.elo += change
        away.elo -= change
        _update_team(home, home_score, away_score)
        _update_team(away, away_score, home_score)

    features = pd.DataFrame(rows)
    features["fifa_rank_diff"] = features["fifa_rank_diff"].fillna(0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features


def latest_team_states(features: pd.DataFrame) -> pd.DataFrame:
    """Return the latest known feature state for each team from the training table."""
    records = []
    for prefix in ("home", "away"):
        cols = {
            f"{prefix}_team": "team",
            f"elo_{prefix}": "elo",
            f"{prefix}_goals_for_last5": "goals_for_last5",
            f"{prefix}_goals_against_last5": "goals_against_last5",
            f"{prefix}_win_rate_last10": "win_rate_last10",
            f"{prefix}_form_points_last5": "form_points_last5",
        }
        available = [c for c in cols if c in features.columns]
        part = features[["date", *available]].rename(columns=cols)
        records.append(part)
    states = pd.concat(records, ignore_index=True).sort_values("date")
    return states.groupby("team", as_index=False).tail(1).reset_index(drop=True)


def final_team_states(matches: pd.DataFrame) -> pd.DataFrame:
    """Return each team's post-match state after the full historical table."""
    states: dict[str, TeamState] = defaultdict(TeamState)
    matches = matches.sort_values("date").reset_index(drop=True)
    latest_date: dict[str, pd.Timestamp] = {}

    for _, match in matches.iterrows():
        home = states[match.home_team]
        away = states[match.away_team]
        home_score = int(match.home_score)
        away_score = int(match.away_score)
        actual_home = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
        home_for_elo = home.elo + (45.0 if not bool(match.neutral) else 0.0)
        expected_home = _expected_score(home_for_elo, away.elo)
        change = _elo_k(match) * _goal_multiplier(abs(home_score - away_score)) * (actual_home - expected_home)
        home.elo += change
        away.elo -= change
        _update_team(home, home_score, away_score)
        _update_team(away, away_score, home_score)
        latest_date[match.home_team] = match.date
        latest_date[match.away_team] = match.date

    rows = []
    for team, state in states.items():
        rows.append(
            {
                "team": team,
                "date": latest_date.get(team),
                "elo": state.elo,
                "goals_for_last5": _sum_recent(state.goals_for, 5),
                "goals_against_last5": _sum_recent(state.goals_against, 5),
                "win_rate_last10": _mean_recent(state.wins, 10),
                "form_points_last5": _sum_recent(state.points, 5),
            }
        )
    return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)


def feature_vector_from_states(home_team: str, away_team: str, states: pd.DataFrame, neutral: bool = True) -> pd.DataFrame:
    """Build one model-ready row for a future match."""
    defaults = {
        "elo": 1500.0,
        "goals_for_last5": 0.0,
        "goals_against_last5": 0.0,
        "win_rate_last10": 0.0,
        "form_points_last5": 0.0,
    }
    by_team = states.set_index("team").to_dict("index") if not states.empty else {}
    home = {**defaults, **by_team.get(home_team, {})}
    away = {**defaults, **by_team.get(away_team, {})}
    row = {
        "elo_diff": home["elo"] - away["elo"],
        "fifa_rank_diff": 0.0,
        "home_goals_for_last5": home["goals_for_last5"],
        "home_goals_against_last5": home["goals_against_last5"],
        "away_goals_for_last5": away["goals_for_last5"],
        "away_goals_against_last5": away["goals_against_last5"],
        "home_win_rate_last10": home["win_rate_last10"],
        "away_win_rate_last10": away["win_rate_last10"],
        "home_form_points_last5": home["form_points_last5"],
        "away_form_points_last5": away["form_points_last5"],
        "is_neutral": int(neutral),
        "is_home_advantage": int(not neutral),
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)
