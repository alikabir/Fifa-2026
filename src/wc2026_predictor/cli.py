from __future__ import annotations

import argparse

from .data_pipeline import build_clean_dataset, download_historical_data
from .features import build_feature_table
from .predict import MatchPredictor
from .simulator import WorldCupSimulator
from .train import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="World Cup 2026 prediction system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download historical match data")
    subparsers.add_parser("prepare", help="Clean data and build features")
    subparsers.add_parser("train", help="Train classifiers and expected-goals model")

    predict_parser = subparsers.add_parser("predict", help="Predict one match")
    predict_parser.add_argument("home_team")
    predict_parser.add_argument("away_team")
    predict_parser.add_argument("--neutral", action="store_true", default=False)

    simulate_parser = subparsers.add_parser("simulate", help="Run tournament Monte Carlo simulation")
    simulate_parser.add_argument("--simulations", type=int, default=100_000)
    simulate_parser.add_argument("--output", default="data/processed/tournament_probabilities.csv")

    args = parser.parse_args()
    if args.command == "download":
        download_historical_data()
    elif args.command == "prepare":
        matches = build_clean_dataset()
        build_feature_table(matches)
    elif args.command == "train":
        train_all()
    elif args.command == "predict":
        predictor = MatchPredictor()
        print(predictor.predict_match(args.home_team, args.away_team, neutral=args.neutral))
    elif args.command == "simulate":
        simulator = WorldCupSimulator(MatchPredictor())
        probabilities = simulator.run(args.simulations)
        probabilities.to_csv(args.output, index=False)
        print(probabilities.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
