from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import label_binarize


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray, labels: list[int]) -> float:
    y_binary = label_binarize(y_true, classes=labels)
    return float(np.mean(np.sum((probabilities - y_binary) ** 2, axis=1)))


def evaluate_classifier(name: str, model, X_test: pd.DataFrame, y_test: np.ndarray, labels: list[int]) -> dict:
    probabilities = model.predict_proba(X_test)
    predictions = probabilities.argmax(axis=1)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities, labels=labels)),
        "brier_score": multiclass_brier_score(y_test, probabilities, labels),
        "calibration_ece": expected_calibration_error(y_test, probabilities),
    }
