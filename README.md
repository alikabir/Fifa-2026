# World Cup 2026 Prediction System

Python system for predicting World Cup 2026 matches and simulating the tournament.
It downloads historical international football results, builds chronological team
features, trains baseline models, predicts expected goals, and runs Monte Carlo
tournament simulations.

## Project Structure

```text
data/
  raw/                 # downloaded match data and optional FIFA rankings
  processed/           # cleaned matches, feature tables, simulation outputs
  tournament/          # 2026 groups and bracket slot configuration
models/                # trained model artifacts and metrics
notebooks/             # exploratory analysis
src/wc2026_predictor/  # production package
dashboard/             # Streamlit app
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data

The default downloader uses the public `martj42/international_results` dataset,
which includes World Cups, qualifiers, continental tournaments, and friendlies.

Optional FIFA rankings:

Place a CSV at `data/raw/fifa_rankings.csv` with:

```csv
date,team,rank
2026-05-01,Argentina,1
2026-05-01,France,2
```

If rankings are absent, the pipeline still creates `fifa_rank_diff` and fills it
with `0.0`, so experiments remain runnable. For production forecasts, provide a
rankings file or replace the loader with a licensed ranking feed.

## Commands

Download data:

```bash
python -m wc2026_predictor.cli download
```

Clean data and build features:

```bash
python -m wc2026_predictor.cli prepare
```

Train Logistic Regression, Random Forest, XGBoost, and expected-goals models:

```bash
python -m wc2026_predictor.cli train
```

Predict one match:

```bash
python -m wc2026_predictor.cli predict Argentina France --neutral
```

Run the full tournament simulator 100,000 times:

```bash
python -m wc2026_predictor.cli simulate --simulations 100000
```

Start the dashboard:

```bash
streamlit run dashboard/app.py
```

## Features

The feature pipeline creates pre-match values only from prior matches:

- Elo rating difference
- FIFA ranking difference
- goals scored in last 5 matches
- goals conceded in last 5 matches
- win rate in last 10 matches
- neutral/home venue indicators
- recent form points

## Models

Classification models predict `home win`, `draw`, and `away win`:

- Logistic Regression
- Random Forest
- XGBoost

Evaluation metrics:

- accuracy
- log loss
- multiclass Brier score
- expected calibration error

Expected goals are modeled separately with a multi-output Random Forest
regressor that predicts goals for both teams.

## Tournament Simulation

The simulator follows the 2026 format: 12 groups of four, top two from every
group plus the eight best third-place teams, then a Round of 32 before the Round
of 16. Group tables use points, goal difference, goals scored, head-to-head
points, head-to-head goal difference, head-to-head goals scored, then random
draw-lots as the final unknown tie-break.

Outputs include probabilities for:

- Round of 32
- Round of 16
- Quarterfinal
- Semifinal
- Final
- Champion

The bracket lives in `data/tournament/worldcup_2026_bracket_slots.csv`, so it can
be adjusted if FIFA updates slot mappings.

## Live Updates During The Tournament

Append newly played 2026 matches to the raw results source or maintain a live CSV
with the same schema, rerun:

```bash
python -m wc2026_predictor.cli prepare
python -m wc2026_predictor.cli train
python -m wc2026_predictor.cli simulate --simulations 100000
```

For a lower-latency setup, retrain nightly and run simulations immediately after
each match using the latest feature table.
