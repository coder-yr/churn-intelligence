"""Unit tests for Customer Lifetime Value (CLV) and Customer Segmentation."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.clv_segmentation import (
    build_survival_lookup_cache,
    calculate_discounted_clv_fast,
    load_data_and_models,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture
def survival_cache_fixture():
    """Build fast survival cache using serialized survival models."""
    survival_artifacts = joblib.load(MODELS_DIR / "survival_models.joblib")
    km_contract = survival_artifacts["km_contract"]
    return build_survival_lookup_cache(km_contract)


def test_clv_calculation_properties(survival_cache_fixture):
    """Verify core financial properties of the discounted cash flow CLV formula."""
    cache = survival_cache_fixture

    # 1. Standard calculation
    clv_base = calculate_discounted_clv_fast(
        monthly_charge=100.0,
        current_tenure=12,
        contract="Month-to-month",
        survival_cache=cache,
        margin_rate=0.70,
        annual_discount_rate=0.10,
        horizon_months=24,
    )
    assert clv_base > 0.0

    # 2. Zero monthly charge yields zero CLV
    clv_zero = calculate_discounted_clv_fast(
        monthly_charge=0.0,
        current_tenure=12,
        contract="Month-to-month",
        survival_cache=cache,
        margin_rate=0.70,
        annual_discount_rate=0.10,
        horizon_months=24,
    )
    assert clv_zero == 0.0

    # 3. Higher margin increases CLV
    clv_high_margin = calculate_discounted_clv_fast(
        monthly_charge=100.0,
        current_tenure=12,
        contract="Month-to-month",
        survival_cache=cache,
        margin_rate=0.85,
        annual_discount_rate=0.10,
        horizon_months=24,
    )
    assert clv_high_margin > clv_base

    # 4. Higher discount rate decreases CLV (Time value of money)
    clv_high_discount = calculate_discounted_clv_fast(
        monthly_charge=100.0,
        current_tenure=12,
        contract="Month-to-month",
        survival_cache=cache,
        margin_rate=0.70,
        annual_discount_rate=0.20,
        horizon_months=24,
    )
    assert clv_high_discount < clv_base

    # 5. Longer horizon increases CLV
    clv_36m = calculate_discounted_clv_fast(
        monthly_charge=100.0,
        current_tenure=12,
        contract="Month-to-month",
        survival_cache=cache,
        margin_rate=0.70,
        annual_discount_rate=0.10,
        horizon_months=36,
    )
    assert clv_36m > clv_base


def test_clv_segmented_dataset_integrity():
    """Verify that enriched dataset contains 7,043 rows and all economic fields."""
    csv_path = PROCESSED_DATA_DIR / "telco_customer_clv_segmented.csv"
    assert csv_path.exists(), "telco_customer_clv_segmented.csv must exist."

    df = pd.read_csv(csv_path)
    assert len(df) == 7043
    assert "clv_24m" in df.columns
    assert "churn_probability" in df.columns
    assert "historical_realized_profit" in df.columns
    assert "segment_name" in df.columns

    # Verify no null values in generated columns
    assert df["clv_24m"].isna().sum() == 0
    assert df["churn_probability"].isna().sum() == 0
    assert df["segment_name"].isna().sum() == 0

    # Verify exactly 4 distinct segments
    assert df["segment_name"].nunique() == 4


def test_clv_summary_json_integrity():
    """Verify that clv_summary.json exists and contains portfolio metrics."""
    json_path = MODELS_DIR / "clv_summary.json"
    assert json_path.exists(), "clv_summary.json must exist."

    with open(json_path, "r") as f:
        data = json.load(f)

    assert "overall_clv" in data
    assert "clustering_evaluation" in data
    assert "sensitivity_sample" in data

    assert data["overall_clv"]["mean_clv_24m"] > 500.0
    assert len(data["clustering_evaluation"]["profile"]) == 4
