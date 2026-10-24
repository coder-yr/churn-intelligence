"""Unit tests for Model Training, Inference, and Threshold Evaluation."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.train import compute_metrics, optimize_business_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def test_model_artifact_exists_and_loads():
    """Verify that the serialized LightGBM model artifact loads cleanly."""
    model_path = MODELS_DIR / "churn_lightgbm.joblib"
    assert model_path.exists(), "Trained model artifact must exist."

    artifact = joblib.load(model_path)
    assert "model" in artifact
    assert "optimal_threshold" in artifact
    assert "default_threshold" in artifact

    # Threshold sanity bounds
    assert 0.10 <= artifact["optimal_threshold"] <= 0.90
    assert artifact["default_threshold"] == 0.50


def test_model_inference_probabilities():
    """Verify that model generates valid prediction probabilities bounded in [0, 1]."""
    model_path = MODELS_DIR / "churn_lightgbm.joblib"
    artifact = joblib.load(model_path)
    model = artifact["model"]

    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    probas = model.predict_proba(X_test)[:, 1]

    assert len(probas) == len(X_test)
    assert (probas >= 0.0).all()
    assert (probas <= 1.0).all()


def test_compute_metrics_calculation():
    """Verify computation of evaluation metrics on synthetic ground truth."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.7, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 1.0  # Caught both 1s
    assert metrics["roc_auc"] == 1.0


def test_optimize_business_threshold_selection():
    """Verify that threshold optimizer returns a valid decision threshold."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.8])

    threshold, stats, df_curve = optimize_business_threshold(
        y_true, y_proba, cost_offer=20.0, offer_save_rate=0.5, customer_margin=120.0
    )

    assert 0.10 <= threshold <= 0.90
    assert "net_value" in stats
    assert len(df_curve) > 0


def test_model_metrics_json_integrity():
    """Verify that model_metrics.json exists and contains all required benchmark models."""
    metrics_file = MODELS_DIR / "model_metrics.json"
    assert metrics_file.exists(), "model_metrics.json must exist."

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    assert "dummy_baseline" in metrics
    assert "logistic_regression" in metrics
    assert "lightgbm_default_threshold" in metrics
    assert "lightgbm_optimal_threshold" in metrics

    # Verify LightGBM outperforms random chance
    assert metrics["lightgbm_default_threshold"]["roc_auc"] > 0.80
    assert metrics["lightgbm_default_threshold"]["pr_auc"] > 0.60
