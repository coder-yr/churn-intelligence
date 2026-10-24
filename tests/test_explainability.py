"""Unit tests for SHAP Explainability and Individual Customer Interpretability."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import shap

from src.models.explainability import (
    compute_shap_explanations,
    explain_single_customer,
    get_global_feature_rankings,
    verify_shap_additivity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture
def loaded_model_and_sample():
    """Load trained LightGBM model and a 10-row slice of test features."""
    model_artifact = joblib.load(MODELS_DIR / "churn_lightgbm.joblib")
    model = model_artifact["model"]

    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv").head(10)
    test_unprocessed = pd.read_csv(PROCESSED_DATA_DIR / "test_unprocessed.csv").head(10)
    return model, X_test, test_unprocessed


def test_shap_computation_and_additivity(loaded_model_and_sample):
    """Verify that TreeExplainer computes valid SHAP values adhering to additivity."""
    model, X_sample, _ = loaded_model_and_sample
    explainer, shap_explanation = compute_shap_explanations(model, X_sample)

    assert shap_explanation.values.shape == X_sample.shape
    assert verify_shap_additivity(model, explainer, X_sample) is True


def test_explain_single_customer(loaded_model_and_sample):
    """Verify that single customer explanation returns expected structure and valid bounds."""
    model, X_sample, test_unprocessed = loaded_model_and_sample
    explainer = shap.TreeExplainer(model)

    explanation = explain_single_customer(
        customer_idx=0,
        explainer=explainer,
        X_df=X_sample,
        test_unprocessed=test_unprocessed,
        top_k=3,
    )

    assert "customerID" in explanation
    assert "predicted_churn_probability" in explanation
    assert 0.0 <= explanation["predicted_churn_probability"] <= 1.0
    assert "risk_drivers" in explanation
    assert "protective_drivers" in explanation
    assert len(explanation["risk_drivers"]) <= 3
    assert len(explanation["protective_drivers"]) <= 3

    # Risk drivers must have positive SHAP values
    for d in explanation["risk_drivers"]:
        assert d["shap_value"] > 0

    # Protective drivers must have negative SHAP values
    for d in explanation["protective_drivers"]:
        assert d["shap_value"] < 0


def test_shap_artifacts_exist_and_valid():
    """Verify that serialized SHAP JSON files exist with expected schemas."""
    rankings_file = MODELS_DIR / "shap_feature_importance.json"
    explanations_file = MODELS_DIR / "sample_customer_explanations.json"

    assert rankings_file.exists(), "shap_feature_importance.json must exist."
    assert explanations_file.exists(), "sample_customer_explanations.json must exist."

    with open(rankings_file, "r") as f:
        rankings = json.load(f)
    assert len(rankings) == 15
    assert "feature" in rankings[0]
    assert "mean_abs_shap" in rankings[0]

    with open(explanations_file, "r") as f:
        explanations = json.load(f)
    assert "high_risk_customer" in explanations
    assert "low_risk_customer" in explanations
    assert explanations["high_risk_customer"]["predicted_churn_probability"] > explanations["low_risk_customer"]["predicted_churn_probability"]
