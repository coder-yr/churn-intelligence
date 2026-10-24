"""Unit tests for Survival Analysis using Lifelines (Kaplan-Meier and Cox PH)."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.survival_analysis import (
    fit_cox_proportional_hazards,
    fit_kaplan_meier_by_group,
    fit_kaplan_meier_overall,
    load_cleaned_customer_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


@pytest.fixture
def sample_survival_data():
    """Load cleaned customer data slice for fast survival tests."""
    df = load_cleaned_customer_data()
    return df


def test_fit_kaplan_meier_overall(sample_survival_data):
    """Verify that overall Kaplan-Meier estimator produces monotonically non-increasing probabilities."""
    kmf, metrics = fit_kaplan_meier_overall(sample_survival_data)

    s12 = metrics["survival_probability_12m"]
    s24 = metrics["survival_probability_24m"]
    s36 = metrics["survival_probability_36m"]

    # Bounds check
    assert 0.0 < s36 <= s24 <= s12 <= 1.0
    assert kmf.timeline is not None
    assert len(kmf.survival_function_) > 0


def test_fit_kaplan_meier_by_contract_significance(sample_survival_data):
    """Verify that contract stratification produces statistically significant survival separation."""
    fitters, stats = fit_kaplan_meier_by_group(sample_survival_data, "Contract")

    assert "Month-to-month" in fitters
    assert "One year" in fitters
    assert "Two year" in fitters

    # Month-to-month survival at 24m must be lower than Two-year
    s_m2m_24 = stats["Month-to-month"]["survival_24m"]
    s_2yr_24 = stats["Two year"]["survival_24m"]
    assert s_m2m_24 < s_2yr_24

    # Log-rank test must confirm p < 0.001
    assert "_logrank_test" in stats
    assert stats["_logrank_test"]["p_value"] < 0.001


def test_fit_cox_proportional_hazards(sample_survival_data):
    """Verify that multivariable Cox model converges with high concordance."""
    cph, hr_df = fit_cox_proportional_hazards(sample_survival_data.sample(2000, random_state=42))

    assert cph.concordance_index_ > 0.85
    assert len(hr_df) > 5

    # Hazard ratios must be strictly positive (exponentiated coefficients)
    assert (hr_df["hazard_ratio"] > 0).all()
    assert (hr_df["lower_95"] > 0).all()
    assert (hr_df["upper_95"] > 0).all()


def test_survival_serialized_artifacts():
    """Verify that survival models and metrics artifacts are saved and loadable."""
    artifact_path = MODELS_DIR / "survival_models.joblib"
    metrics_path = MODELS_DIR / "survival_metrics.json"

    assert artifact_path.exists(), "survival_models.joblib must exist."
    assert metrics_path.exists(), "survival_metrics.json must exist."

    artifacts = joblib.load(artifact_path)
    assert "km_overall" in artifacts
    assert "km_contract" in artifacts
    assert "cox_model" in artifacts
    assert "c_index" in artifacts

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    assert "overall_survival" in metrics
    assert "contract_survival" in metrics
    assert "top_hazard_ratios" in metrics
