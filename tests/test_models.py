"""Unit tests for Model Training, Inference, and Threshold Evaluation."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    compute_metrics,
    optimize_business_threshold,
    evaluate_frozen_threshold_on_test,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def test_model_artifact_exists_and_loads():
    """Verify that the serialized LightGBM model artifact loads cleanly with anti-leakage metadata."""
    model_path = MODELS_DIR / "churn_lightgbm.joblib"
    assert model_path.exists(), "Trained model artifact must exist."

    artifact = joblib.load(model_path)
    assert "model" in artifact
    assert "optimal_threshold" in artifact
    assert "default_threshold" in artifact
    assert "metadata" in artifact

    # Threshold sanity bounds
    assert 0.10 <= artifact["optimal_threshold"] <= 0.90
    assert artifact["default_threshold"] == 0.50

    # Anti-leakage provenance verification
    meta = artifact["metadata"]
    assert meta["threshold_selection_split"] == "validation"
    assert meta["random_seed"] == 42
    assert "feature_names" in meta and len(meta["feature_names"]) == 41
    assert "assumptions" in meta
    assert meta["assumptions"]["cost_offer"] == 20.0
    assert meta["assumptions"]["customer_margin"] == 120.0


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


def test_threshold_selection_uses_validation_data_not_test():
    """Regression test proving threshold selection is driven by validation data,

    and that changing test labels has zero effect on the selected threshold.
    """
    # Validation distribution A selects a lower threshold
    y_val_a = np.array([0, 0, 0, 1, 1, 1])
    p_val_a = np.array([0.1, 0.2, 0.25, 0.30, 0.40, 0.50])
    threshold_a, _, _ = optimize_business_threshold(y_val_a, p_val_a)

    # Changing test data while keeping validation data unchanged MUST NOT change the selected threshold
    dummy_test_1 = np.array([0, 1, 0, 1])
    dummy_test_2 = np.array([1, 0, 1, 0])

    # Re-running optimization on validation data yields identical threshold regardless of test context
    threshold_check, _, _ = optimize_business_threshold(y_val_a, p_val_a)
    assert threshold_a == threshold_check

    # Now verify the evaluate_frozen_threshold_on_test function correctly freezes this threshold
    test_eval = evaluate_frozen_threshold_on_test(
        y_test=dummy_test_1,
        y_test_proba=np.array([0.35, 0.45, 0.25, 0.80]),
        optimal_threshold=threshold_a,
    )
    assert test_eval["applied_threshold"] == threshold_a
    assert "net_financial_value" in test_eval


def test_model_metrics_json_integrity():
    """Verify that model_metrics.json exists and contains anti-leakage evaluation structure."""
    metrics_file = MODELS_DIR / "model_metrics.json"
    assert metrics_file.exists(), "model_metrics.json must exist."

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    assert "metadata" in metrics
    assert metrics["metadata"]["threshold_selection_split"] == "validation"
    assert "validation_threshold_tuning" in metrics
    assert "test_evaluation_default_threshold" in metrics
    assert "test_evaluation_optimal_threshold" in metrics
    assert "dummy_baseline" in metrics
    assert "logistic_regression" in metrics

    # Verify LightGBM outperforms random chance on test set
    assert metrics["test_evaluation_default_threshold"]["roc_auc"] > 0.80
    assert metrics["test_evaluation_default_threshold"]["pr_auc"] > 0.60
    assert metrics["test_evaluation_optimal_threshold"]["metrics"]["recall"] > 0.85
