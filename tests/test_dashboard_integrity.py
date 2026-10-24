"""
Integration test for Streamlit Dashboard data integrity and artifact loading.
"""

from pathlib import Path
import json
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def test_dashboard_data_artifact_exists():
    csv_path = DATA_DIR / "telco_customer_clv_segmented.csv"
    assert csv_path.exists(), f"Missing segmented dataset: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 7043
    required_cols = [
        "customerID", "Churn", "churn_probability", "clv_24m",
        "segment_name", "MonthlyCharges", "tenure", "Contract"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col} in dashboard dataset"


def test_all_json_metric_summaries_exist_and_load():
    expected_files = [
        "model_metrics.json",
        "shap_feature_importance.json",
        "sample_customer_explanations.json",
        "survival_metrics.json",
        "clv_summary.json",
        "uplift_metrics.json",
        "roi_simulation_summary.json",
    ]

    for fname in expected_files:
        fpath = MODELS_DIR / fname
        assert fpath.exists(), f"Missing JSON summary: {fpath}"
        with open(fpath, "r") as f:
            data = json.load(f)
            assert isinstance(data, (dict, list)), f"Corrupt JSON structure in {fname}"


def test_key_figures_exist_for_dashboard_tabs():
    expected_figures = [
        "02_churn_by_contract.png",
        "05_churn_by_payment_method.png",
        "09_confusion_matrices.png",
        "10_cost_threshold_curve.png",
        "12_shap_beeswarm.png",
        "16_km_by_contract.png",
        "18_cox_hazard_ratios.png",
        "21_customer_segments_scatter.png",
        "22_segment_profiles_comparison.png",
        "23_uplift_qini_curve.png",
        "25_uplift_quadrants.png",
        "26_roi_sensitivity_curves.png",
    ]

    for fig in expected_figures:
        fig_path = FIGURES_DIR / fig
        assert fig_path.exists(), f"Missing required dashboard figure: {fig_path}"
