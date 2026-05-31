from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wc2026_predictor.config import MODEL_DIR, PROCESSED_DIR, TOURNAMENT_DIR
from wc2026_predictor.predict import MatchPredictor
from wc2026_predictor.simulator import WorldCupSimulator


st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
st.title("World Cup 2026 Prediction System")


def models_ready() -> bool:
    classifier_exists = any(
        (MODEL_DIR / name).exists()
        for name in ("logistic_regression.joblib", "random_forest.joblib", "xgboost.joblib")
    )
    return (
        (MODEL_DIR / "label_encoder.joblib").exists()
        and (MODEL_DIR / "expected_goals.joblib").exists()
        and classifier_exists
    )


@st.cache_resource
def load_predictor() -> MatchPredictor:
    return MatchPredictor()


@st.cache_data(show_spinner="Running tournament simulation...")
def run_tournament_simulation(simulations: int, seed: int) -> pd.DataFrame:
    simulator = WorldCupSimulator(load_predictor(), seed=seed)
    return simulator.run(simulations)


@st.cache_data
def load_groups() -> pd.DataFrame:
    return pd.read_csv(TOURNAMENT_DIR / "worldcup_2026_groups.csv")


groups = load_groups()
teams = sorted(groups["team"].unique())

if not models_ready():
    st.warning("Models are not trained yet. Run `python -m wc2026_predictor.cli train` first.")
    st.stop()

predictor = load_predictor()

tab_match, tab_rankings, tab_simulation, tab_bracket = st.tabs(
    ["Match Predictions", "Team Rankings", "Tournament Simulations", "Bracket Projections"]
)

with tab_match:
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        home_team = st.selectbox(
            "Team A",
            teams,
            index=teams.index("Argentina") if "Argentina" in teams else 0,
            key="match_home_team",
        )
    with col_b:
        away_team = st.selectbox(
            "Team B",
            teams,
            index=teams.index("France") if "France" in teams else 1,
            key="match_away_team",
        )
    with col_c:
        neutral = st.toggle("Neutral venue", value=True, key="match_neutral")

    if home_team == away_team:
        st.warning("Choose two different teams.")
    else:
        prediction = predictor.predict_match(home_team, away_team, neutral=neutral)
        metrics = st.columns(5)
        metrics[0].metric(f"{home_team} win", f"{prediction['home_win']:.1%}")
        metrics[1].metric("Draw", f"{prediction['draw']:.1%}")
        metrics[2].metric(f"{away_team} win", f"{prediction['away_win']:.1%}")
        metrics[3].metric(f"{home_team} xG", f"{prediction['home_xg']:.2f}")
        metrics[4].metric(f"{away_team} xG", f"{prediction['away_xg']:.2f}")

        prob_df = pd.DataFrame(
            {
                "Outcome": [f"{home_team} win", "Draw", f"{away_team} win"],
                "Probability": [prediction["home_win"], prediction["draw"], prediction["away_win"]],
            }
        )
        st.plotly_chart(px.bar(prob_df, x="Outcome", y="Probability", range_y=[0, 1]), use_container_width=True)

with tab_rankings:
    states_path = PROCESSED_DIR / "team_states.parquet"
    if states_path.exists():
        states = pd.read_parquet(states_path)
        world_cup_states = groups[["team", "group"]].merge(states, on="team", how="left")
        world_cup_states = world_cup_states.sort_values("elo", ascending=False)
        st.dataframe(world_cup_states, use_container_width=True, hide_index=True)
        chart = px.bar(world_cup_states.head(20), x="team", y="elo", color="group")
        st.plotly_chart(chart, use_container_width=True)

with tab_simulation:
    col_sims, col_seed = st.columns([2, 1])
    with col_sims:
        simulations = st.slider("Simulations", min_value=100, max_value=20_000, value=2_000, step=100)
    with col_seed:
        seed = st.number_input("Seed", min_value=1, max_value=1_000_000, value=42, step=1)

    probabilities = run_tournament_simulation(simulations, int(seed))
    st.session_state["simulation_probabilities"] = probabilities
    st.dataframe(probabilities, use_container_width=True, hide_index=True)
    chart = px.bar(probabilities.head(20), x="team", y="champion", color="round_of_16")
    st.plotly_chart(chart, use_container_width=True)

with tab_bracket:
    bracket = pd.read_csv(TOURNAMENT_DIR / "worldcup_2026_bracket_slots.csv")
    st.dataframe(bracket, use_container_width=True, hide_index=True)
    if "simulation_probabilities" in st.session_state:
        top = st.session_state["simulation_probabilities"].head(8)
        st.plotly_chart(px.bar(top, x="team", y=["round_of_16", "quarterfinal", "semifinal", "final", "champion"]), use_container_width=True)
