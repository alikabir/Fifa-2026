from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from .config import PROCESSED_DIR, RAW_DIR, RESULTS_URL, SHOOTOUTS_URL
from .team_names import normalize_team


@dataclass(frozen=True)
class DataPaths:
    results: Path = RAW_DIR / "results.csv"
    shootouts: Path = RAW_DIR / "shootouts.csv"
    rankings: Path = RAW_DIR / "fifa_rankings.csv"
    cleaned_matches: Path = PROCESSED_DIR / "matches.parquet"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def download_historical_data(paths: DataPaths | None = None) -> None:
    """Download public international match results and shootout data."""
    paths = paths or DataPaths()
    _download_file(RESULTS_URL, paths.results)
    _download_file(SHOOTOUTS_URL, paths.shootouts)


def load_raw_results(paths: DataPaths | None = None) -> pd.DataFrame:
    paths = paths or DataPaths()
    if not paths.results.exists():
        download_historical_data(paths)
    return pd.read_csv(paths.results)


def load_fifa_rankings(paths: DataPaths | None = None) -> pd.DataFrame | None:
    """Load optional rankings with columns: date, team, rank."""
    paths = paths or DataPaths()
    if not paths.rankings.exists():
        return None
    rankings = pd.read_csv(paths.rankings)
    required = {"date", "team", "rank"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"fifa_rankings.csv is missing columns: {sorted(missing)}")
    rankings = rankings[list(required)].copy()
    rankings["date"] = pd.to_datetime(rankings["date"])
    rankings["team"] = rankings["team"].map(normalize_team)
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="coerce")
    return rankings.dropna(subset=["date", "team", "rank"])


def _attach_rankings(matches: pd.DataFrame, rankings: pd.DataFrame | None) -> pd.DataFrame:
    matches = matches.sort_values("date").copy()
    if rankings is None or rankings.empty:
        matches["home_fifa_rank"] = pd.NA
        matches["away_fifa_rank"] = pd.NA
        return matches

    rankings = rankings.sort_values("date")
    home_lookup = rankings.rename(columns={"team": "home_team", "rank": "home_fifa_rank"})
    away_lookup = rankings.rename(columns={"team": "away_team", "rank": "away_fifa_rank"})

    matches = pd.merge_asof(
        matches,
        home_lookup,
        on="date",
        by="home_team",
        direction="backward",
    )
    matches = pd.merge_asof(
        matches,
        away_lookup,
        on="date",
        by="away_team",
        direction="backward",
    )
    return matches


def clean_matches(raw: pd.DataFrame, rankings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clean historical results into a structured pandas table."""
    matches = raw.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches["home_team"] = matches["home_team"].map(normalize_team)
    matches["away_team"] = matches["away_team"].map(normalize_team)
    matches["country"] = matches["country"].map(normalize_team)
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)
    matches["neutral"] = matches["neutral"].astype(bool)

    matches["outcome"] = "D"
    matches.loc[matches["home_score"] > matches["away_score"], "outcome"] = "H"
    matches.loc[matches["home_score"] < matches["away_score"], "outcome"] = "A"
    matches["is_world_cup"] = matches["tournament"].str.contains("FIFA World Cup", case=False, na=False)
    matches["is_qualifier"] = matches["tournament"].str.contains("qualification|qualifier", case=False, na=False)
    matches["is_friendly"] = matches["tournament"].str.contains("friendly", case=False, na=False)
    matches["is_continental"] = matches["tournament"].str.contains(
        "Euro|Copa|Gold Cup|Asian Cup|African Cup|Africa Cup|Nations Cup|Oceania",
        case=False,
        na=False,
    )
    matches = _attach_rankings(matches, rankings)
    columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
        "outcome",
        "home_fifa_rank",
        "away_fifa_rank",
        "is_world_cup",
        "is_qualifier",
        "is_continental",
        "is_friendly",
    ]
    return matches[columns].sort_values("date").reset_index(drop=True)


def build_clean_dataset(paths: DataPaths | None = None) -> pd.DataFrame:
    paths = paths or DataPaths()
    raw = load_raw_results(paths)
    rankings = load_fifa_rankings(paths)
    matches = clean_matches(raw, rankings)
    paths.cleaned_matches.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(paths.cleaned_matches, index=False)
    return matches


def load_clean_dataset(paths: DataPaths | None = None) -> pd.DataFrame:
    paths = paths or DataPaths()
    if not paths.cleaned_matches.exists():
        return build_clean_dataset(paths)
    return pd.read_parquet(paths.cleaned_matches)
