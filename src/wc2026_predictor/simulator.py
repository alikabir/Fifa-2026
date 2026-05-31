from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

import numpy as np
import pandas as pd

from .config import TOURNAMENT_DIR
from .team_names import normalize_team


class MatchPredictionProtocol(Protocol):
    def predict_match(self, home_team: str, away_team: str, neutral: bool = True) -> dict:
        ...


@dataclass
class SimulatedMatch:
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int

    @property
    def winner(self) -> str | None:
        if self.goals_a > self.goals_b:
            return self.team_a
        if self.goals_b > self.goals_a:
            return self.team_b
        return None


class WorldCupSimulator:
    """Monte Carlo simulator for the 48-team 2026 World Cup format."""

    def __init__(
        self,
        predictor: MatchPredictionProtocol,
        groups_path=TOURNAMENT_DIR / "worldcup_2026_groups.csv",
        bracket_path=TOURNAMENT_DIR / "worldcup_2026_bracket_slots.csv",
        seed: int = 42,
    ):
        self.predictor = predictor
        self.groups = pd.read_csv(groups_path)
        self.groups["team"] = self.groups["team"].map(normalize_team)
        self.bracket = pd.read_csv(bracket_path).sort_values("match")
        self.rng = np.random.default_rng(seed)

    def _simulate_regulation_match(self, team_a: str, team_b: str) -> SimulatedMatch:
        prediction = self.predictor.predict_match(team_a, team_b, neutral=True)
        goals_a = int(self.rng.poisson(max(prediction["home_xg"], 0.05)))
        goals_b = int(self.rng.poisson(max(prediction["away_xg"], 0.05)))
        return SimulatedMatch(team_a, team_b, goals_a, goals_b)

    def _simulate_knockout_winner(self, team_a: str, team_b: str) -> str:
        match = self._simulate_regulation_match(team_a, team_b)
        if match.winner:
            return match.winner
        prediction = self.predictor.predict_match(team_a, team_b, neutral=True)
        decisive_home = prediction["home_win"] / max(prediction["home_win"] + prediction["away_win"], 1e-9)
        return team_a if self.rng.random() < decisive_home else team_b

    @staticmethod
    def _head_to_head_stats(teams: list[str], matches: list[SimulatedMatch]) -> dict[str, tuple[int, int, int]]:
        stats = {team: [0, 0, 0] for team in teams}
        team_set = set(teams)
        for match in matches:
            if match.team_a not in team_set or match.team_b not in team_set:
                continue
            if match.goals_a > match.goals_b:
                stats[match.team_a][0] += 3
            elif match.goals_b > match.goals_a:
                stats[match.team_b][0] += 3
            else:
                stats[match.team_a][0] += 1
                stats[match.team_b][0] += 1
            stats[match.team_a][1] += match.goals_a - match.goals_b
            stats[match.team_b][1] += match.goals_b - match.goals_a
            stats[match.team_a][2] += match.goals_a
            stats[match.team_b][2] += match.goals_b
        return {team: tuple(values) for team, values in stats.items()}

    def _rank_group(self, teams: list[str], matches: list[SimulatedMatch]) -> pd.DataFrame:
        table = {team: {"team": team, "points": 0, "gf": 0, "ga": 0, "wins": 0} for team in teams}
        for match in matches:
            a = table[match.team_a]
            b = table[match.team_b]
            a["gf"] += match.goals_a
            a["ga"] += match.goals_b
            b["gf"] += match.goals_b
            b["ga"] += match.goals_a
            if match.goals_a > match.goals_b:
                a["points"] += 3
                a["wins"] += 1
            elif match.goals_b > match.goals_a:
                b["points"] += 3
                b["wins"] += 1
            else:
                a["points"] += 1
                b["points"] += 1

        rows = pd.DataFrame(table.values())
        rows["gd"] = rows["gf"] - rows["ga"]
        rows["draw_lots"] = self.rng.random(len(rows))

        h2h_points = []
        h2h_gd = []
        h2h_gf = []
        for _, row in rows.iterrows():
            tied = rows[
                (rows["points"] == row.points)
                & (rows["gd"] == row.gd)
                & (rows["gf"] == row.gf)
            ]["team"].tolist()
            stats = self._head_to_head_stats(tied, matches)
            hp, hg, hf = stats.get(row.team, (0, 0, 0))
            h2h_points.append(hp)
            h2h_gd.append(hg)
            h2h_gf.append(hf)
        rows["h2h_points"] = h2h_points
        rows["h2h_gd"] = h2h_gd
        rows["h2h_gf"] = h2h_gf

        return rows.sort_values(
            ["points", "gd", "gf", "h2h_points", "h2h_gd", "h2h_gf", "draw_lots"],
            ascending=[False, False, False, False, False, False, False],
        ).reset_index(drop=True)

    def _simulate_group_stage(self) -> tuple[dict[str, dict[int, str]], list[dict]]:
        group_positions: dict[str, dict[int, str]] = {}
        third_place_rows = []
        for group, group_df in self.groups.groupby("group", sort=True):
            teams = group_df.sort_values("slot")["team"].tolist()
            matches = [self._simulate_regulation_match(a, b) for a, b in combinations(teams, 2)]
            ranked = self._rank_group(teams, matches)
            group_positions[group] = {index + 1: row.team for index, row in ranked.iterrows()}
            third = ranked.iloc[2].to_dict()
            third["group"] = group
            third_place_rows.append(third)
        return group_positions, third_place_rows

    def _best_thirds(self, third_place_rows: list[dict]) -> list[dict]:
        thirds = pd.DataFrame(third_place_rows)
        thirds["draw_lots"] = self.rng.random(len(thirds))
        thirds = thirds.sort_values(
            ["points", "gd", "gf", "draw_lots"],
            ascending=[False, False, False, False],
        )
        return thirds.head(8).to_dict("records")

    def _resolve_slot(self, slot: str, group_positions: dict[str, dict[int, str]], thirds: list[dict], winners: dict[int, str]) -> str:
        if slot.startswith("W"):
            return winners[int(slot[1:])]
        if slot.startswith("3:"):
            eligible = set(slot[2:].split("/"))
            for index, third in enumerate(thirds):
                if third["group"] in eligible:
                    return thirds.pop(index)["team"]
            return thirds.pop(0)["team"]
        group = slot[0]
        position = int(slot[1:])
        return group_positions[group][position]

    def simulate_once(self) -> dict[str, set[str] | str]:
        group_positions, third_rows = self._simulate_group_stage()
        thirds = self._best_thirds(third_rows)
        advanced_r32 = {team for group in group_positions.values() for team in (group[1], group[2])}
        advanced_r32.update(third["team"] for third in thirds)

        winners: dict[int, str] = {}
        reached = {
            "round_of_32": set(advanced_r32),
            "round_of_16": set(),
            "quarterfinal": set(),
            "semifinal": set(),
            "final": set(),
        }
        for row in self.bracket.itertuples(index=False):
            team_a = self._resolve_slot(row.slot_a, group_positions, thirds, winners)
            team_b = self._resolve_slot(row.slot_b, group_positions, thirds, winners)
            winner = self._simulate_knockout_winner(team_a, team_b)
            winners[int(row.match)] = winner
            if row.round == "R32":
                reached["round_of_16"].add(winner)
            elif row.round == "R16":
                reached["quarterfinal"].add(winner)
            elif row.round == "QF":
                reached["semifinal"].add(winner)
            elif row.round == "SF":
                reached["final"].add(winner)
            elif row.round == "FINAL":
                reached["champion"] = winner
        return reached

    def run(self, simulations: int = 100_000) -> pd.DataFrame:
        teams = sorted(self.groups["team"].unique())
        counters = {
            team: {
                "round_of_32": 0,
                "round_of_16": 0,
                "quarterfinal": 0,
                "semifinal": 0,
                "final": 0,
                "champion": 0,
            }
            for team in teams
        }
        for _ in range(simulations):
            result = self.simulate_once()
            for stage in ("round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"):
                for team in result[stage]:
                    counters[team][stage] += 1
            counters[result["champion"]]["champion"] += 1

        rows = []
        for team, values in counters.items():
            row = {"team": team}
            row.update({stage: count / simulations for stage, count in values.items()})
            rows.append(row)
        return pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)
