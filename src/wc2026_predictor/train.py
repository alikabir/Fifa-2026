from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import MODEL_DIR, MODEL_FEATURES, PROCESSED_DIR, RANDOM_STATE
from .data_pipeline import load_clean_dataset
from .evaluate import evaluate_classifier
from .features import build_feature_table, final_team_states


def _xgboost_classifier():
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=350,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
    )


def _chronological_split(features: pd.DataFrame, test_fraction: float = 0.2):
    features = features.sort_values("date").reset_index(drop=True)
    split_at = int(len(features) * (1.0 - test_fraction))
    train_df = features.iloc[:split_at].copy()
    test_df = features.iloc[split_at:].copy()
    return train_df, test_df


def train_match_models(features: pd.DataFrame) -> pd.DataFrame:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = _chronological_split(features)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["outcome"])
    y_test = label_encoder.transform(test_df["outcome"])
    X_train = train_df[MODEL_FEATURES]
    X_test = test_df[MODEL_FEATURES]

    models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": _xgboost_classifier(),
    }

    metrics = []
    labels = list(range(len(label_encoder.classes_)))
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        metrics.append(evaluate_classifier(name, model, X_test, y_test, labels))

    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")
    metrics_df = pd.DataFrame(metrics).sort_values(["log_loss", "calibration_ece"])
    metrics_df.to_json(MODEL_DIR / "classification_metrics.json", orient="records", indent=2)
    return metrics_df


def train_expected_goals_model(features: pd.DataFrame):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = _chronological_split(features)
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    model.fit(train_df[MODEL_FEATURES], train_df[["home_score", "away_score"]])
    predictions = model.predict(test_df[MODEL_FEATURES]).clip(0, None)
    mae_home = abs(predictions[:, 0] - test_df["home_score"].to_numpy()).mean()
    mae_away = abs(predictions[:, 1] - test_df["away_score"].to_numpy()).mean()
    joblib.dump(model, MODEL_DIR / "expected_goals.joblib")
    metrics = {"home_goals_mae": float(mae_home), "away_goals_mae": float(mae_away)}
    (MODEL_DIR / "expected_goals_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_all() -> tuple[pd.DataFrame, dict]:
    matches = load_clean_dataset()
    features = build_feature_table(matches)
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    states = final_team_states(matches)
    states.to_parquet(PROCESSED_DIR / "team_states.parquet", index=False)
    classification_metrics = train_match_models(features)
    xg_metrics = train_expected_goals_model(features)
    return classification_metrics, xg_metrics


if __name__ == "__main__":
    cls_metrics, goal_metrics = train_all()
    print(cls_metrics.to_string(index=False))
    print(json.dumps(goal_metrics, indent=2))
