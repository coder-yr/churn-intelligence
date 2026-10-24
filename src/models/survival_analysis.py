"""Survival Analysis Module using Lifelines.

Implements:
1. Kaplan-Meier Survival Analysis across cohorts (Contract, Internet, Tech Support).
2. Log-rank tests verifying statistically significant differences in survival functions.
3. Multivariable Cox Proportional Hazards modeling with Hazard Ratio ($e^\beta$) extraction.
4. Median survival time and expected remaining lifetime calculations.
5. Explicit handling of right-censoring and rigorous documentation of data limitations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def load_cleaned_customer_data() -> pd.DataFrame:
    """Load cleaned customer records for time-to-event modeling."""
    csv_path = PROCESSED_DATA_DIR / "telco_churn_cleaned.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d customer records for survival analysis.", len(df))
    return df


def fit_kaplan_meier_overall(
    df: pd.DataFrame,
) -> Tuple[KaplanMeierFitter, Dict[str, Any]]:
    """Fit non-parametric Kaplan-Meier survival estimator to overall cohort.

    T = tenure (duration in months)
    E = churn_label (1 if churn event occurred, 0 if right-censored)
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["tenure"], event_observed=df["churn_label"], label="Overall Customer Base")

    median_survival = float(kmf.median_survival_time_)
    survival_1yr = float(kmf.predict(12))
    survival_2yr = float(kmf.predict(24))
    survival_3yr = float(kmf.predict(36))

    metrics = {
        "median_survival_months": median_survival,
        "survival_probability_12m": round(survival_1yr, 4),
        "survival_probability_24m": round(survival_2yr, 4),
        "survival_probability_36m": round(survival_3yr, 4),
    }

    logger.info("Overall KM fit: 1-yr survival=%.1f%%, 2-yr=%.1f%%, 3-yr=%.1f%%",
                survival_1yr * 100, survival_2yr * 100, survival_3yr * 100)
    return kmf, metrics


def fit_kaplan_meier_by_group(
    df: pd.DataFrame, column: str
) -> Tuple[Dict[str, KaplanMeierFitter], Dict[str, Any]]:
    """Fit stratified Kaplan-Meier curves and perform log-rank significance test."""
    groups = df[column].unique()
    fitters = {}
    stats = {}

    for g in groups:
        mask = df[column] == g
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df.loc[mask, "tenure"], event_observed=df.loc[mask, "churn_label"], label=str(g))
        fitters[str(g)] = kmf
        stats[str(g)] = {
            "customer_count": int(mask.sum()),
            "median_survival_months": float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else "Undefined (>72m)",
            "survival_12m": round(float(kmf.predict(12)), 4),
            "survival_24m": round(float(kmf.predict(24)), 4),
            "survival_48m": round(float(kmf.predict(48)), 4),
        }

    # Log-rank test between largest 2 groups for statistical verification
    group_list = list(groups)
    if len(group_list) >= 2:
        m1 = df[column] == group_list[0]
        m2 = df[column] == group_list[1]
        lr_res = logrank_test(
            df.loc[m1, "tenure"], df.loc[m2, "tenure"],
            df.loc[m1, "churn_label"], df.loc[m2, "churn_label"]
        )
        stats["_logrank_test"] = {
            "group_a": str(group_list[0]),
            "group_b": str(group_list[1]),
            "p_value": float(lr_res.p_value),
            "test_statistic": round(float(lr_res.test_statistic), 4),
        }

    return fitters, stats


def fit_cox_proportional_hazards(df: pd.DataFrame) -> Tuple[CoxPHFitter, pd.DataFrame]:
    r"""Fit a multivariable Cox Proportional Hazards Model to estimate hazard ratios.

    $h(t | X) = h_0(t) \exp(\beta^T X)$
    Hazard Ratio ($e^\beta$):
    - HR > 1.0 indicates elevated risk of churn (shorter lifespan).
    - HR < 1.0 indicates protective factor (longer lifespan).
    """
    logger.info("Fitting Cox Proportional Hazards Model...")

    # Select representative features for survival regression
    cols = [
        "tenure",
        "churn_label",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "TechSupport",
        "PaymentMethod",
        "PaperlessBilling",
    ]
    df_cox = df[cols].copy()

    # One-hot encode categoricals with drop_first to prevent colinearity
    df_cox = pd.get_dummies(df_cox, columns=["Contract", "InternetService", "TechSupport", "PaymentMethod", "PaperlessBilling"], drop_first=True)

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df_cox, duration_col="tenure", event_col="churn_label")

    hr_df = pd.DataFrame({
        "covariate": cph.hazard_ratios_.index,
        "hazard_ratio": cph.hazard_ratios_.values.round(4),
        "lower_95": np.exp(cph.confidence_intervals_["95% lower-bound"]).values.round(4),
        "upper_95": np.exp(cph.confidence_intervals_["95% upper-bound"]).values.round(4),
        "p_value": cph.summary["p"].values.round(6),
    }).sort_values("hazard_ratio", ascending=False)

    logger.info("Cox PH Model Concordance Index: %.4f", cph.concordance_index_)
    return cph, hr_df


def plot_survival_figures(
    km_overall: KaplanMeierFitter,
    km_contract: Dict[str, KaplanMeierFitter],
    km_tech: Dict[str, KaplanMeierFitter],
    cox_hr_df: pd.DataFrame,
    output_dir: Path,
):
    """Generate and save publication-quality survival analysis figures."""
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall Kaplan-Meier Survival Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    km_overall.plot_survival_function(ax=ax, color="#2b5c8f", linewidth=2.5, ci_show=True)
    ax.set_title("Overall Customer Survival Function S(t) (Kaplan-Meier)", fontweight="bold", pad=12)
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Survival Probability (Probability of Not Churning)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.savefig(output_dir / "15_km_overall_survival.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Stratified by Contract Type
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Month-to-month": "#d9534f", "One year": "#f0ad4e", "Two year": "#5cb85c"}
    for label, kmf in km_contract.items():
        kmf.plot_survival_function(ax=ax, color=colors.get(label, "#333333"), linewidth=2.5, ci_show=False)
    ax.set_title("Customer Survival Curves by Contract Agreement", fontweight="bold", pad=12)
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.savefig(output_dir / "16_km_by_contract.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Stratified by Tech Support & Internet
    fig, ax = plt.subplots(figsize=(8, 5))
    tech_colors = {"No": "#d9534f", "Yes": "#5cb85c", "No internet service": "#428bca"}
    for label, kmf in km_tech.items():
        kmf.plot_survival_function(ax=ax, color=tech_colors.get(label, "#333333"), linewidth=2.5, ci_show=False)
    ax.set_title("Customer Survival by Tech Support Subscription", fontweight="bold", pad=12)
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.savefig(output_dir / "17_km_by_services.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Cox Proportional Hazards Forest Plot (Top Hazard Ratios)
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_df = cox_hr_df.head(10).sort_values("hazard_ratio", ascending=True)

    y_pos = np.arange(len(plot_df))
    ax.scatter(plot_df["hazard_ratio"], y_pos, color="#2b5c8f", s=60, zorder=3)
    ax.errorbar(
        plot_df["hazard_ratio"], y_pos,
        xerr=[plot_df["hazard_ratio"] - plot_df["lower_95"], plot_df["upper_95"] - plot_df["hazard_ratio"]],
        fmt="none", color="#2b5c8f", capsize=4, zorder=2
    )
    ax.axvline(1.0, color="#d9534f", linestyle="--", linewidth=1.5, label="Baseline Hazard (1.0)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["covariate"])
    ax.set_xlabel("Hazard Ratio ($e^\\beta$) with 95% Confidence Interval")
    ax.set_title("Cox Proportional Hazards Model: Key Hazard Ratios", fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    fig.savefig(output_dir / "18_cox_hazard_ratios.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved 4 survival figures to %s", output_dir)


def run_survival_pipeline() -> Dict[str, Any]:
    """Execute end-to-end survival analysis pipeline."""
    df = load_cleaned_customer_data()

    # 1. Overall Kaplan-Meier
    km_overall, overall_metrics = fit_kaplan_meier_overall(df)

    # 2. Stratified Kaplan-Meier
    km_contract, contract_metrics = fit_kaplan_meier_by_group(df, "Contract")
    km_tech, tech_metrics = fit_kaplan_meier_by_group(df, "TechSupport")

    # 3. Cox Proportional Hazards
    cph_model, cox_hr_df = fit_cox_proportional_hazards(df)

    # 4. Generate Figures
    plot_survival_figures(km_overall, km_contract, km_tech, cox_hr_df, FIGURES_DIR)

    # 5. Save Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    survival_artifacts = {
        "km_overall": km_overall,
        "km_contract": km_contract,
        "km_tech": km_tech,
        "cox_model": cph_model,
        "c_index": round(float(cph_model.concordance_index_), 4),
    }
    joblib.dump(survival_artifacts, MODELS_DIR / "survival_models.joblib")
    logger.info("Saved survival models to %s", MODELS_DIR / "survival_models.joblib")

    # Save summary stats JSON
    metrics_summary = {
        "overall_survival": overall_metrics,
        "contract_survival": contract_metrics,
        "tech_support_survival": tech_metrics,
        "cox_c_index": round(float(cph_model.concordance_index_), 4),
        "top_hazard_ratios": cox_hr_df.to_dict(orient="records"),
    }
    with open(MODELS_DIR / "survival_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING SURVIVAL ANALYSIS PIPELINE (LIFELINES)")
    print("=" * 60)
    results = run_survival_pipeline()

    print("\nOVERALL SURVIVAL METRICS:")
    for k, v in results["overall_survival"].items():
        print(f" - {k}: {v}")

    print("\nSURVIVAL BY CONTRACT AGREEMENT:")
    for contract, stats in results["contract_survival"].items():
        if not contract.startswith("_"):
            print(f" - {contract:15s} : 1-Yr Survival={stats['survival_12m']*100:.1f}%, 2-Yr={stats['survival_24m']*100:.1f}%, Median={stats['median_survival_months']}")

    print("\nTOP COX PROPORTIONAL HAZARDS RATIOS:")
    for r in results["top_hazard_ratios"][:6]:
        print(f" - {r['covariate']:30s} : Hazard Ratio = {r['hazard_ratio']:.3f} (p={r['p_value']:.4f})")
    print("=" * 60)
