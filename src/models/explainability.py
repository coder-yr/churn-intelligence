"""Model Explainability Module using SHAP (SHapley Additive exPlanations).

Provides:
1. Global feature importance and beeswarm distribution across the customer base.
2. Feature dependence plots capturing non-linear interactions.
3. Individual customer-level waterfall explanations for front-line retention reps.
4. Verification of the SHAP additivity property against the trained LightGBM model.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def load_model_and_test_data() -> Tuple[Any, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load the trained LightGBM model, preprocessed test set, and raw test profiles."""
    model_path = MODELS_DIR / "churn_lightgbm.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run src.models.train first."
        )

    artifact = joblib.load(model_path)
    model = artifact["model"]

    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    test_unprocessed = pd.read_csv(PROCESSED_DATA_DIR / "test_unprocessed.csv")
    y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")

    return model, X_test, test_unprocessed, y_test


def compute_shap_explanations(
    model: Any, X_test: pd.DataFrame
) -> Tuple[shap.TreeExplainer, shap.Explanation]:
    """Fit a TreeExplainer and calculate SHAP values for the test cohort."""
    logger.info("Computing TreeExplainer SHAP values for %d test records...", len(X_test))
    # TreeExplainer calculates exact Shapley values in polynomial time for tree ensembles
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X_test)
    logger.info("SHAP computation completed successfully.")
    return explainer, shap_explanation


def generate_global_shap_plots(
    shap_explanation: shap.Explanation,
    output_dir: Path = FIGURES_DIR,
) -> Dict[str, Path]:
    """Generate and save global bar, beeswarm, and dependence plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    # 1. Global Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(9, 8))
    shap.plots.bar(shap_explanation, max_display=15, show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontweight="bold", pad=12)
    bar_path = output_dir / "11_shap_summary_bar.png"
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_paths["bar"] = bar_path

    # 2. Beeswarm Summary Plot (Shows magnitude AND directional effect)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
    plt.title("SHAP Beeswarm Plot (Directional Feature Impact on Churn)", fontweight="bold", pad=12)
    beeswarm_path = output_dir / "12_shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_paths["beeswarm"] = beeswarm_path

    # 3. Dependence Plot: Tenure
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.scatter(shap_explanation[:, "tenure"], color=shap_explanation[:, "Contract_Two year"], show=False)
    plt.title("SHAP Dependence: Tenure vs. Two-Year Contract", fontweight="bold", pad=12)
    tenure_path = output_dir / "13_shap_dependence_tenure.png"
    plt.savefig(tenure_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_paths["dependence_tenure"] = tenure_path

    # 4. Dependence Plot: Monthly Charges
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.scatter(shap_explanation[:, "MonthlyCharges"], color=shap_explanation[:, "has_fiber_no_support"], show=False)
    plt.title("SHAP Dependence: Monthly Charges vs. Fiber Without Support", fontweight="bold", pad=12)
    charges_path = output_dir / "14_shap_dependence_monthly_charges.png"
    plt.savefig(charges_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_paths["dependence_charges"] = charges_path

    logger.info("Saved 4 global SHAP diagnostic plots to %s", output_dir)
    return saved_paths


def get_global_feature_rankings(
    shap_explanation: shap.Explanation, top_n: int = 15
) -> List[Dict[str, Any]]:
    """Compute mean absolute SHAP values and return ranked list of features."""
    mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)
    feature_names = shap_explanation.feature_names

    ranking = []
    for feat, val in sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)[:top_n]:
        ranking.append({
            "feature": feat,
            "mean_abs_shap": round(float(val), 4),
        })
    return ranking


def explain_single_customer(
    customer_idx: int,
    explainer: shap.TreeExplainer,
    X_df: pd.DataFrame,
    test_unprocessed: pd.DataFrame,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Generate transparent, customer-level churn risk factor explanations.

    Returns:
    - Base rate (expected average log-odds)
    - Customer churn probability
    - Top risk factors (features pushing toward churn)
    - Top protective factors (features keeping customer retained)
    """
    row_features = X_df.iloc[[customer_idx]]
    customer_profile = test_unprocessed.iloc[customer_idx].to_dict()
    customer_id = customer_profile.get("customerID", f"Index_{customer_idx}")

    # Compute SHAP values for this single customer
    shap_vals = explainer(row_features)
    values = shap_vals.values[0]
    feature_names = row_features.columns.tolist()
    feature_raw_vals = row_features.iloc[0].tolist()

    # Base value and model margin output
    base_val = float(shap_vals.base_values[0])
    model_output_margin = float(base_val + values.sum())

    # Sigmoid function converts margin to predicted probability
    predicted_proba = float(1.0 / (1.0 + np.exp(-model_output_margin)))

    # Pair features with their SHAP impact
    feature_impacts = []
    for name, s_val, r_val in zip(feature_names, values, feature_raw_vals):
        feature_impacts.append({
            "feature": name,
            "shap_value": round(float(s_val), 4),
            "feature_value": round(float(r_val), 4),
        })

    # Sort into risk drivers (positive SHAP) and retention drivers (negative SHAP)
    risk_drivers = sorted(
        [f for f in feature_impacts if f["shap_value"] > 0],
        key=lambda x: x["shap_value"],
        reverse=True,
    )[:top_k]

    protective_drivers = sorted(
        [f for f in feature_impacts if f["shap_value"] < 0],
        key=lambda x: x["shap_value"],
    )[:top_k]

    return {
        "customerID": customer_id,
        "predicted_churn_probability": round(predicted_proba, 4),
        "base_value_margin": round(base_val, 4),
        "total_shap_sum": round(float(values.sum()), 4),
        "risk_drivers": risk_drivers,
        "protective_drivers": protective_drivers,
        "customer_profile": customer_profile,
    }


def verify_shap_additivity(
    model: Any, explainer: shap.TreeExplainer, X_sample: pd.DataFrame
) -> bool:
    """Verify the SHAP efficiency / additivity property against the raw model predictions.

    Property: base_value + sum(SHAP_values) == raw_prediction_margin
    """
    raw_preds = model.predict(X_sample, raw_score=True)
    shap_vals = explainer(X_sample)

    for i in range(len(X_sample)):
        reconstructed = shap_vals.base_values[i] + shap_vals.values[i].sum()
        actual = raw_preds[i]
        if not np.isclose(reconstructed, actual, atol=1e-4):
            logger.error(
                "Additivity mismatch at index %d: Reconstructed=%.5f, Actual=%.5f",
                i,
                reconstructed,
                actual,
            )
            return False

    logger.info("SHAP additivity verification PASSED for all %d checked records.", len(X_sample))
    return True


def run_explainability_pipeline():
    """Execute full SHAP computation, plot generation, and individual explanations."""
    model, X_test, test_unprocessed, y_test = load_model_and_test_data()

    # 1. Compute SHAP explanations
    explainer, shap_explanation = compute_shap_explanations(model, X_test)

    # 2. Verify Additivity Property on sample
    verify_shap_additivity(model, explainer, X_test.head(50))

    # 3. Generate and save global diagnostic plots
    generate_global_shap_plots(shap_explanation, FIGURES_DIR)

    # 4. Extract top global rankings
    global_ranking = get_global_feature_rankings(shap_explanation, top_n=15)
    with open(MODELS_DIR / "shap_feature_importance.json", "w") as f:
        json.dump(global_ranking, f, indent=2)

    # 5. Explain sample high-risk, moderate-risk, and low-risk customers
    probas = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(probas))
    low_risk_idx = int(np.argmin(probas))
    moderate_risk_idx = int(np.argmin(np.abs(probas - 0.50)))

    high_risk_explanation = explain_single_customer(
        high_risk_idx, explainer, X_test, test_unprocessed
    )
    moderate_risk_explanation = explain_single_customer(
        moderate_risk_idx, explainer, X_test, test_unprocessed
    )
    low_risk_explanation = explain_single_customer(
        low_risk_idx, explainer, X_test, test_unprocessed
    )

    sample_explanations = {
        "high_risk_customer": high_risk_explanation,
        "moderate_risk_customer": moderate_risk_explanation,
        "low_risk_customer": low_risk_explanation,
    }
    with open(MODELS_DIR / "sample_customer_explanations.json", "w") as f:
        json.dump(sample_explanations, f, indent=2)

    logger.info("Saved sample customer explanations to %s", MODELS_DIR / "sample_customer_explanations.json")
    return global_ranking, sample_explanations


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING SHAP MODEL EXPLAINABILITY PIPELINE")
    print("=" * 60)
    rankings, samples = run_explainability_pipeline()

    print("\nTOP 10 GLOBAL CHURN DRIVERS (MEAN |SHAP|):")
    print("-" * 60)
    for i, r in enumerate(rankings[:10], 1):
        print(f" {i:2d}. {r['feature']:30s} : {r['mean_abs_shap']:.4f}")

    print("\nSAMPLE HIGH-RISK CUSTOMER EXPLANATION:")
    print("-" * 60)
    high_c = samples["high_risk_customer"]
    print(f"Customer ID: {high_c['customerID']} (Predicted Churn: {high_c['predicted_churn_probability']*100:.1f}%)")
    print("Top Risk Factors (Pulling toward Churn):")
    for d in high_c["risk_drivers"]:
        print(f"  [+] {d['feature']:25s} (SHAP: +{d['shap_value']:.3f})")
    print("Top Protective Factors (Pushing toward Retained):")
    for d in high_c["protective_drivers"]:
        print(f"  [-] {d['feature']:25s} (SHAP: {d['shap_value']:.3f})")
    print("=" * 60)
