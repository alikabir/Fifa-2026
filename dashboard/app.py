from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wc2026_predictor.config import MODEL_DIR, PROCESSED_DIR, TOURNAMENT_DIR
from wc2026_predictor.config import RESULTS_URL
from wc2026_predictor.data_pipeline import clean_matches
from wc2026_predictor.features import final_team_states
from wc2026_predictor.live_worldcup_api import WorldCupApiClient, completed_games_as_matches
from wc2026_predictor.predict import MatchPredictor
from wc2026_predictor.simulator import WorldCupSimulator


st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #101114;
        --panel: #181b20;
        --panel-soft: #20242b;
        --line: rgba(255, 255, 255, 0.09);
        --text-soft: rgba(255, 255, 255, 0.68);
        --gold: #f4c95d;
        --green: #48c78e;
        --cyan: #5cc8ff;
        --red: #ff6b6b;
    }
    .stApp {
        background:
            linear-gradient(135deg, rgba(72, 199, 142, 0.12), transparent 34%),
            linear-gradient(315deg, rgba(244, 201, 93, 0.10), transparent 32%),
            var(--bg);
        color: #f7f7f4;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    div[data-testid="stTabs"] button {
        font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    div[data-testid="stMetric"] label {
        color: var(--text-soft);
    }
    .hero {
        border: 1px solid var(--line);
        background:
            linear-gradient(100deg, rgba(24, 27, 32, 0.98), rgba(24, 27, 32, 0.78)),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.06) 0 1px, transparent 1px 76px);
        border-radius: 8px;
        padding: 26px 28px;
        margin-bottom: 18px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.24);
    }
    .hero-kicker {
        color: var(--gold);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: clamp(2.1rem, 5vw, 4.4rem);
        font-weight: 900;
        line-height: 1;
        margin: 0 0 12px;
    }
    .hero-copy {
        color: var(--text-soft);
        max-width: 880px;
        font-size: 1.02rem;
        line-height: 1.55;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 850;
        margin: 8px 0 4px;
    }
    .muted {
        color: var(--text-soft);
    }
    .fixture {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px 20px;
        background: var(--panel);
    }
    .fixture-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        font-size: clamp(1.1rem, 3vw, 2rem);
        font-weight: 900;
    }
    .versus {
        color: var(--gold);
        font-size: 0.9rem;
        letter-spacing: 0.12em;
    }
    .small-note {
        color: var(--text-soft);
        font-size: 0.88rem;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Live tournament intelligence</div>
        <div class="hero-title">World Cup 2026 Predictor</div>
        <div class="hero-copy">
            Match probabilities, expected goals, team strength, and Monte Carlo tournament paths
            from a trained international football model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


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


@st.cache_data(ttl=3600, show_spinner="Refreshing latest public match data...")
def load_live_team_states(include_world_cup_api: bool = True) -> tuple[pd.DataFrame, pd.Timestamp, int]:
    raw = pd.read_csv(RESULTS_URL)
    matches = clean_matches(raw)
    api_match_count = 0
    if include_world_cup_api:
        try:
            api_games = WorldCupApiClient().games()
            api_matches = completed_games_as_matches(api_games)
            api_match_count = len(api_matches)
            if api_match_count:
                matches = pd.concat([matches, api_matches], ignore_index=True).sort_values("date")
        except Exception:
            api_match_count = 0
    states = final_team_states(matches)
    latest_match_date = matches["date"].max()
    return states, latest_match_date, api_match_count


@st.cache_data(ttl=900, show_spinner="Loading World Cup API...")
def load_worldcup_api_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    client = WorldCupApiClient()
    return client.games(), client.groups(), client.teams()


groups = load_groups()
teams = sorted(groups["team"].unique())

if not models_ready():
    st.warning("Models are not trained yet. Run `python -m wc2026_predictor.cli train` first.")
    st.stop()

predictor = load_predictor()

with st.sidebar:
    st.markdown("### Data Mode")
    live_mode = st.toggle("Use latest public match data", value=True)
    use_worldcup_api = st.toggle("Include World Cup API", value=True)
    if live_mode:
        live_states, latest_match_date, api_match_count = load_live_team_states(use_worldcup_api)
        predictor.states = live_states
        st.success(f"Updated through {latest_match_date.date()}")
        if use_worldcup_api:
            st.caption(f"World Cup API completed matches included: {api_match_count}")
        if st.button("Refresh now"):
            load_live_team_states.clear()
            load_worldcup_api_data.clear()
            st.rerun()
    else:
        st.info("Using committed model artifacts.")
    st.caption(
        "Live mode refreshes rolling team state from the public results dataset. "
        "The trained model itself is not retrained in the web app."
    )


def percent(value: float) -> str:
    return f"{value:.1%}"


def style_probability_table(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    probability_cols = [
        "round_of_32",
        "round_of_16",
        "quarterfinal",
        "semifinal",
        "final",
        "champion",
    ]
    return (
        frame.style.format({col: "{:.1%}" for col in probability_cols if col in frame.columns})
        .background_gradient(subset=[col for col in probability_cols if col in frame.columns], cmap="Greens")
    )


def probability_table_config() -> dict:
    return {
        "team": st.column_config.TextColumn("Team", width="medium"),
        "round_of_32": st.column_config.ProgressColumn("R32", format="%.1%", min_value=0, max_value=1),
        "round_of_16": st.column_config.ProgressColumn("R16", format="%.1%", min_value=0, max_value=1),
        "quarterfinal": st.column_config.ProgressColumn("QF", format="%.1%", min_value=0, max_value=1),
        "semifinal": st.column_config.ProgressColumn("SF", format="%.1%", min_value=0, max_value=1),
        "final": st.column_config.ProgressColumn("Final", format="%.1%", min_value=0, max_value=1),
        "champion": st.column_config.ProgressColumn("Win", format="%.1%", min_value=0, max_value=1),
    }


def outcome_chart(prob_df: pd.DataFrame):
    colors = ["#48c78e", "#f4c95d", "#ff6b6b"]
    fig = px.bar(
        prob_df,
        x="Probability",
        y="Outcome",
        orientation="h",
        text=prob_df["Probability"].map(percent),
        color="Outcome",
        color_discrete_sequence=colors,
        range_x=[0, 1],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        showlegend=False,
        height=310,
        margin=dict(l=10, r=42, t=16, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickformat=".0%", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=None),
    )
    return fig


def champion_chart(probabilities: pd.DataFrame):
    top = probabilities.head(16).sort_values("champion", ascending=True)
    fig = px.bar(
        top,
        x="champion",
        y="team",
        orientation="h",
        color="round_of_16",
        color_continuous_scale=["#ff6b6b", "#f4c95d", "#48c78e"],
        text=top["champion"].map(percent),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=64, t=18, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickformat=".0%", title="Title probability", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=None),
        coloraxis_colorbar=dict(title="R32"),
    )
    return fig


def stage_funnel(team_row: pd.Series):
    stages = ["round_of_16", "quarterfinal", "semifinal", "final", "champion"]
    labels = ["R16", "QF", "SF", "Final", "Win"]
    values = [float(team_row[stage]) for stage in stages]
    fig = go.Figure(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            line=dict(color="#5cc8ff", width=4),
            marker=dict(size=11, color="#f4c95d"),
            fill="tozeroy",
            fillcolor="rgba(92, 200, 255, 0.16)",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=20, t=18, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(title=None),
    )
    return fig

tab_match, tab_live, tab_rankings, tab_simulation, tab_bracket = st.tabs(
    ["Match Center", "Live Feed", "Power Rankings", "Tournament Odds", "Bracket"]
)

with tab_match:
    st.markdown('<div class="section-title">Match Center</div>', unsafe_allow_html=True)
    st.caption("Compare any two qualified teams using model probabilities and expected goals.")
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
        st.markdown(
            f"""
            <div class="fixture">
                <div class="fixture-row">
                    <span>{home_team}</span>
                    <span class="versus">VS</span>
                    <span>{away_team}</span>
                </div>
                <div class="small-note">Model: {prediction["model"]} · Venue: {"Neutral" if neutral else "Home advantage"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        metrics = st.columns(5)
        metrics[0].metric(f"{home_team} win", percent(prediction["home_win"]))
        metrics[1].metric("Draw", percent(prediction["draw"]))
        metrics[2].metric(f"{away_team} win", percent(prediction["away_win"]))
        metrics[3].metric(f"{home_team} xG", f"{prediction['home_xg']:.2f}")
        metrics[4].metric(f"{away_team} xG", f"{prediction['away_xg']:.2f}")

        prob_df = pd.DataFrame(
            {
                "Outcome": [f"{home_team} win", "Draw", f"{away_team} win"],
                "Probability": [prediction["home_win"], prediction["draw"], prediction["away_win"]],
            }
        )
        chart_col, table_col = st.columns([2, 1])
        with chart_col:
            st.plotly_chart(outcome_chart(prob_df), use_container_width=True)
        with table_col:
            st.markdown('<div class="section-title">Probability Split</div>', unsafe_allow_html=True)
            display = prob_df.copy()
            display["Probability"] = display["Probability"].map(percent)
            st.dataframe(display, use_container_width=True, hide_index=True)

with tab_live:
    st.markdown('<div class="section-title">World Cup API Feed</div>', unsafe_allow_html=True)
    st.caption("Live schedule, scores, teams, and group tables from the public World Cup 2026 API.")
    try:
        api_games, api_groups, api_teams = load_worldcup_api_data()
        live_metrics = st.columns(4)
        live_metrics[0].metric("Matches", f"{len(api_games)}")
        live_metrics[1].metric("Finished", f"{int(api_games['finished'].sum()) if not api_games.empty else 0}")
        live_metrics[2].metric("Teams", f"{len(api_teams)}")
        live_metrics[3].metric("Groups", f"{api_groups['group'].nunique() if not api_groups.empty else 0}")

        match_filter = st.segmented_control(
            "Matches",
            ["All", "Upcoming", "Finished"],
            default="Upcoming",
        )
        games_view = api_games.copy()
        if match_filter == "Upcoming":
            games_view = games_view[~games_view["finished"]]
        elif match_filter == "Finished":
            games_view = games_view[games_view["finished"]]
        games_view = games_view[
            [
                "id",
                "type",
                "group",
                "matchday",
                "local_date",
                "home_team_name_en",
                "home_score",
                "away_score",
                "away_team_name_en",
                "finished",
                "time_elapsed",
            ]
        ].rename(
            columns={
                "id": "Match",
                "type": "Type",
                "group": "Group",
                "matchday": "Matchday",
                "local_date": "Kickoff",
                "home_team_name_en": "Home",
                "home_score": "Home score",
                "away_score": "Away score",
                "away_team_name_en": "Away",
                "finished": "Finished",
                "time_elapsed": "Status",
            }
        )
        st.dataframe(games_view, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Group Standings</div>', unsafe_allow_html=True)
        group_choice = st.selectbox("Group", sorted(api_groups["group"].unique()) if not api_groups.empty else [])
        group_table = api_groups[api_groups["group"] == group_choice][
            ["team", "mp", "w", "d", "l", "pts", "gf", "ga", "gd"]
        ].rename(
            columns={
                "team": "Team",
                "mp": "MP",
                "w": "W",
                "d": "D",
                "l": "L",
                "pts": "Pts",
                "gf": "GF",
                "ga": "GA",
                "gd": "GD",
            }
        )
        st.dataframe(group_table, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error("Could not reach the World Cup API right now.")
        st.caption(str(exc))

with tab_rankings:
    st.markdown('<div class="section-title">Power Rankings</div>', unsafe_allow_html=True)
    st.caption("Latest model state for the 48-team field, sorted by Elo strength.")
    states_path = PROCESSED_DIR / "team_states.parquet"
    if states_path.exists():
        states = pd.read_parquet(states_path)
        world_cup_states = groups[["team", "group"]].merge(states, on="team", how="left")
        world_cup_states = world_cup_states.sort_values("elo", ascending=False)
        group_filter = st.multiselect(
            "Groups",
            sorted(world_cup_states["group"].unique()),
            default=sorted(world_cup_states["group"].unique()),
        )
        filtered_states = world_cup_states[world_cup_states["group"].isin(group_filter)]
        metrics = st.columns(4)
        metrics[0].metric("Teams", f"{len(filtered_states)}")
        metrics[1].metric("Top Elo", f"{filtered_states['elo'].max():.0f}")
        metrics[2].metric("Avg Elo", f"{filtered_states['elo'].mean():.0f}")
        metrics[3].metric("Groups", f"{len(group_filter)}")
        chart = px.bar(
            filtered_states.head(24),
            x="team",
            y="elo",
            color="group",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        chart.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title=None),
            yaxis=dict(title="Elo", gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(chart, use_container_width=True)
        table = filtered_states[
            ["group", "team", "elo", "goals_for_last5", "goals_against_last5", "win_rate_last10", "form_points_last5"]
        ].copy()
        table["elo"] = table["elo"].round(0).astype("Int64")
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "group": "Group",
                "team": "Team",
                "elo": "Elo",
                "goals_for_last5": st.column_config.NumberColumn("GF last 5", format="%.0f"),
                "goals_against_last5": st.column_config.NumberColumn("GA last 5", format="%.0f"),
                "win_rate_last10": st.column_config.ProgressColumn("Win rate last 10", format="%.0f%%", min_value=0, max_value=1),
                "form_points_last5": st.column_config.NumberColumn("Form pts last 5", format="%.0f"),
            },
        )

with tab_simulation:
    st.markdown('<div class="section-title">Tournament Odds</div>', unsafe_allow_html=True)
    st.caption("Monte Carlo paths through groups, Round of 32, knockouts, and the final.")
    col_sims, col_seed = st.columns([2, 1])
    with col_sims:
        simulations = st.slider("Simulations", min_value=100, max_value=20_000, value=2_000, step=100)
    with col_seed:
        seed = st.number_input("Seed", min_value=1, max_value=1_000_000, value=42, step=1)

    probabilities = run_tournament_simulation(simulations, int(seed))
    st.session_state["simulation_probabilities"] = probabilities
    leader = probabilities.iloc[0]
    metrics = st.columns(5)
    metrics[0].metric("Favorite", leader["team"])
    metrics[1].metric("Title odds", percent(float(leader["champion"])))
    metrics[2].metric("Final odds", percent(float(leader["final"])))
    metrics[3].metric("Semifinal odds", percent(float(leader["semifinal"])))
    metrics[4].metric("Runs", f"{simulations:,}")

    probability_cols = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final", "champion"]
    table_view = probabilities[["team", *probability_cols]].copy()
    st.markdown('<div class="section-title">Full Probability Table</div>', unsafe_allow_html=True)
    st.dataframe(
        table_view,
        use_container_width=True,
        hide_index=True,
        height=650,
        column_config=probability_table_config(),
    )

    selected_team = st.selectbox(
        "Inspect team path",
        probabilities["team"].tolist(),
        index=0,
        key="simulation_team_path",
    )
    row = probabilities.loc[probabilities["team"] == selected_team].iloc[0]
    detail_cols = st.columns(6)
    detail_cols[0].metric("R32", percent(float(row["round_of_32"])))
    detail_cols[1].metric("R16", percent(float(row["round_of_16"])))
    detail_cols[2].metric("QF", percent(float(row["quarterfinal"])))
    detail_cols[3].metric("SF", percent(float(row["semifinal"])))
    detail_cols[4].metric("Final", percent(float(row["final"])))
    detail_cols[5].metric("Win", percent(float(row["champion"])))

with tab_bracket:
    st.markdown('<div class="section-title">Bracket Projection</div>', unsafe_allow_html=True)
    st.caption("Official slot configuration used by the simulator. Update this file if FIFA changes mappings.")
    bracket = pd.read_csv(TOURNAMENT_DIR / "worldcup_2026_bracket_slots.csv")
    st.dataframe(
        bracket,
        use_container_width=True,
        hide_index=True,
        column_config={
            "match": "Match",
            "round": "Round",
            "slot_a": "Slot A",
            "slot_b": "Slot B",
        },
    )
    if "simulation_probabilities" in st.session_state:
        top = st.session_state["simulation_probabilities"].head(8)
        path = top.melt(
            id_vars="team",
            value_vars=["round_of_16", "quarterfinal", "semifinal", "final", "champion"],
            var_name="Stage",
            value_name="Probability",
        )
        fig = px.line(
            path,
            x="Stage",
            y="Probability",
            color="team",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickformat=".0%", gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(title=None),
        )
        st.plotly_chart(fig, use_container_width=True)
