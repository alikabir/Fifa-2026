from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from .team_names import normalize_team

BASE_URL = "https://worldcup26.ir"


@dataclass(frozen=True)
class WorldCupApiClient:
    base_url: str = BASE_URL
    timeout: int = 30

    def _get(self, path: str) -> dict:
        response = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def games(self) -> pd.DataFrame:
        payload = self._get("/get/games")
        games = pd.DataFrame(payload.get("games", []))
        if games.empty:
            return games
        games["home_team_name_en"] = games["home_team_name_en"].map(normalize_team)
        games["away_team_name_en"] = games["away_team_name_en"].map(normalize_team)
        games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce").fillna(0).astype(int)
        games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce").fillna(0).astype(int)
        games["finished"] = games["finished"].astype(str).str.upper().eq("TRUE")
        games["local_date"] = pd.to_datetime(games["local_date"], format="%m/%d/%Y %H:%M", errors="coerce")
        return games.sort_values(["local_date", "id"], na_position="last").reset_index(drop=True)

    def teams(self) -> pd.DataFrame:
        payload = self._get("/get/teams")
        teams = pd.DataFrame(payload.get("teams", []))
        if teams.empty:
            return teams
        teams["name_en"] = teams["name_en"].map(normalize_team)
        return teams.rename(columns={"id": "team_id", "name_en": "team"})

    def groups(self) -> pd.DataFrame:
        payload = self._get("/get/groups")
        groups = []
        for group in payload.get("groups", []):
            group_name = group.get("name")
            for team in group.get("teams", []):
                row = {"group": group_name, **team}
                groups.append(row)
        table = pd.DataFrame(groups)
        if table.empty:
            return table
        numeric_cols = ["mp", "w", "l", "d", "pts", "gf", "ga", "gd"]
        for col in numeric_cols:
            table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0).astype(int)
        teams = self.teams()
        if not teams.empty:
            table = table.merge(teams[["team_id", "team", "flag", "fifa_code"]], on="team_id", how="left")
        return table.sort_values(["group", "pts", "gd", "gf"], ascending=[True, False, False, False]).reset_index(drop=True)


def completed_games_as_matches(games: pd.DataFrame) -> pd.DataFrame:
    """Convert completed API games into the historical match schema."""
    if games.empty:
        return pd.DataFrame()
    completed = games[games["finished"]].copy()
    if completed.empty:
        return pd.DataFrame()
    matches = pd.DataFrame(
        {
            "date": completed["local_date"],
            "home_team": completed["home_team_name_en"],
            "away_team": completed["away_team_name_en"],
            "home_score": completed["home_score"],
            "away_score": completed["away_score"],
            "tournament": "FIFA World Cup 2026",
            "city": "",
            "country": "",
            "neutral": True,
            "home_fifa_rank": pd.NA,
            "away_fifa_rank": pd.NA,
            "is_world_cup": True,
            "is_qualifier": False,
            "is_continental": False,
            "is_friendly": False,
        }
    )
    matches["outcome"] = "D"
    matches.loc[matches["home_score"] > matches["away_score"], "outcome"] = "H"
    matches.loc[matches["home_score"] < matches["away_score"], "outcome"] = "A"
    return matches.dropna(subset=["date", "home_team", "away_team"]).reset_index(drop=True)
