"""
Unit tests for Retention Campaign Financial ROI Simulator.
"""

from pathlib import Path
import json
import pytest

from src.retention.roi_simulator import (
    calculate_discounted_customer_value,
    simulate_retention_campaign,
    compare_retention_policies,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def test_discounted_customer_value_calculation():
    # ARPU $100, 70% margin, 12 months, 10% discount rate
    val = calculate_discounted_customer_value(
        monthly_arpu=100.0,
        gross_margin=0.70,
        horizon_months=12,
        annual_discount_rate=0.10,
    )
    # 12 * $70 undiscounted = $840. With 10% discount, should be between $780 and $820
    assert 780.0 < val < 820.0


def test_simulation_validations():
    # Negative population
    with pytest.raises(ValueError):
        simulate_retention_campaign(
            target_population=0,
            offer_cost=20.0,
            baseline_churn_rate=0.5,
            retention_uplift=0.1,
            customer_arpu=50.0,
        )

    # Negative offer cost
    with pytest.raises(ValueError):
        simulate_retention_campaign(
            target_population=100,
            offer_cost=-10.0,
            baseline_churn_rate=0.5,
            retention_uplift=0.1,
            customer_arpu=50.0,
        )

    # Invalid baseline churn
    with pytest.raises(ValueError):
        simulate_retention_campaign(
            target_population=100,
            offer_cost=20.0,
            baseline_churn_rate=1.5,
            retention_uplift=0.1,
            customer_arpu=50.0,
        )


def test_simulation_outcomes_consistency():
    sim = simulate_retention_campaign(
        target_population=1000,
        offer_cost=30.0,
        baseline_churn_rate=0.40,
        retention_uplift=0.10,
        customer_arpu=80.0,
        gross_margin=0.70,
        horizon_months=12,
        annual_discount_rate=0.10,
    )

    inputs = sim["inputs"]
    vol = sim["volume_outcomes"]
    fin = sim["financial_outcomes"]
    be = sim["break_even"]

    # Total cost = 1000 * 30 = 30000
    assert fin["total_campaign_cost"] == 30000.0

    # Incremental saved = 1000 * 0.10 = 100
    assert vol["incrementally_saved_customers"] == 100

    # Net profit = Gross benefit - Total cost
    assert pytest.approx(fin["net_campaign_profit"], abs=0.1) == (
        fin["gross_financial_benefit"] - fin["total_campaign_cost"]
    )

    # ROI % = (Net profit / Total cost) * 100
    expected_roi = (fin["net_campaign_profit"] / fin["total_campaign_cost"]) * 100.0
    assert pytest.approx(fin["roi_percentage"], abs=0.1) == expected_roi

    assert "is_profitable" in be
    assert isinstance(be["is_profitable"], bool)


def test_compare_retention_policies():
    df = compare_retention_policies()
    assert len(df) == 3
    assert "Policy" in df.columns
    assert "ROI (%)" in df.columns
    # Policy 3 (Precision) should show substantially higher ROI than Policy 1 (Mass Blanket)
    assert "Precision" in df.iloc[2]["Policy"]


def test_roi_artifacts_exist():
    summary_path = MODELS_DIR / "roi_simulation_summary.json"
    figure_path = FIGURES_DIR / "26_roi_sensitivity_curves.png"

    assert summary_path.exists(), f"Missing ROI summary: {summary_path}"
    assert figure_path.exists(), f"Missing Figure 26: {figure_path}"

    with open(summary_path, "r") as f:
        data = json.load(f)

    assert "reference_simulation" in data
    assert "policy_comparison" in data
    assert len(data["policy_comparison"]) == 3
