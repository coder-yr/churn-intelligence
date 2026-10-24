r"""Customer Lifetime Value (CLV) & Customer Segmentation Module.

Implements:
1. Discounted Cash Flow Forward CLV:
   CLV_i = \sum_{t=1}^{H} [ (Margin * MonthlyCharges_i * S_i(t | tenure_i)) / (1 + d)^t ]
   - Strictly separates revenue from gross profit.
   - Conditions survival on current tenure (Bayesian conditional survival).
   - High-performance memoized survival matrix for instantaneous calculation.
2. Multi-dimensional sensitivity analysis across margins, discount rates, and horizons.
3. K-Means Customer Segmentation profiling accounts along Risk, Value, and Tenure.
4. Generates diagnostic figures and serialized segmentation artifacts.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def load_data_and_models() -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """Load cleaned customer data, trained LightGBM model, and survival fitters."""
    csv_path = PROCESSED_DATA_DIR / "telco_churn_cleaned.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)

    # Load churn model
    churn_artifact = joblib.load(MODELS_DIR / "churn_lightgbm.joblib")
    churn_model = churn_artifact["model"]

    # Load survival models
    survival_artifacts = joblib.load(MODELS_DIR / "survival_models.joblib")
    km_contract = survival_artifacts["km_contract"]

    return df, churn_model, km_contract


def build_survival_lookup_cache(
    km_contract: Dict[str, Any], max_tenure: int = 75, max_horizon: int = 36
) -> Dict[Tuple[str, int, int], np.ndarray]:
    """Precompute conditional survival curves for all (contract, tenure, horizon) pairs.

    Because tenure is bounded in [0, 72] and contracts in {Month-to-month, One year, Two year},
    there are only ~220 unique combinations. Precomputing this cache reduces execution time
    from minutes down to a few milliseconds.
    """
    cache = {}
    horizons = [12, 24, 36]

    for contract, fitter in km_contract.items():
        # Pre-extract survival function series for O(1) indexed lookups
        s_func = fitter.survival_function_
        max_time = s_func.index.max()

        def get_s(t: int) -> float:
            if t in s_func.index:
                return float(s_func.loc[t].iloc[0])
            elif t > max_time:
                return float(s_func.iloc[-1].iloc[0])
            else:
                # Find closest earlier timestamp
                earlier = s_func.index[s_func.index <= t]
                if len(earlier) > 0:
                    return float(s_func.loc[earlier[-1]].iloc[0])
                return 1.0

        for tenure in range(max_tenure + 1):
            s_cur = max(get_s(tenure), 0.01)
            for h in horizons:
                t_future = [tenure + t for t in range(1, h + 1)]
                s_future = np.array([get_s(t) for t in t_future])
                cond_s = np.clip(s_future / s_cur, 0.0, 1.0)
                cache[(contract, tenure, h)] = cond_s

    return cache


def calculate_discounted_clv_fast(
    monthly_charge: float,
    current_tenure: int,
    contract: str,
    survival_cache: Dict[Tuple[str, int, int], np.ndarray],
    margin_rate: float = 0.70,
    annual_discount_rate: float = 0.10,
    horizon_months: int = 24,
) -> float:
    """Calculate forward-looking expected CLV using precomputed survival cache."""
    monthly_discount_rate = (1.0 + annual_discount_rate) ** (1.0 / 12.0) - 1.0
    monthly_profit = monthly_charge * margin_rate

    cond_survival = survival_cache.get(
        (contract, current_tenure, horizon_months),
        survival_cache.get(("Month-to-month", current_tenure, horizon_months)),
    )

    timeline = np.arange(1, horizon_months + 1)
    discount_factors = (1.0 + monthly_discount_rate) ** timeline

    discounted_cash_flows = (monthly_profit * cond_survival) / discount_factors
    return float(np.sum(discounted_cash_flows))


def enrich_dataset_with_clv_and_risk(
    df: pd.DataFrame,
    churn_model: Any,
    survival_cache: Dict[Tuple[str, int, int], np.ndarray],
) -> pd.DataFrame:
    """Compute churn risk probability, historical realized profit, and forward CLV."""
    logger.info("Computing churn probability and forward CLV for all customers...")
    df = df.copy()

    # 1. Feature Preprocessing for Churn Model
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    from src.features.build_features import engineer_domain_features, get_feature_lists
    df_feat = engineer_domain_features(df)
    num_cols, cat_cols = get_feature_lists()
    X_mat = preprocessor.transform(df_feat[num_cols + cat_cols])
    df["churn_probability"] = churn_model.predict_proba(X_mat)[:, 1].round(4)

    # 2. Historical Realized Value (Gross Profit to date = 70% of TotalCharges)
    df["historical_realized_profit"] = (df["TotalCharges"] * 0.70).round(2)

    # 3. Forward-Looking 24-Month Expected CLV (Conditional on contract survival)
    clv_24m = [
        calculate_discounted_clv_fast(
            monthly_charge=row["MonthlyCharges"],
            current_tenure=int(row["tenure"]),
            contract=row["Contract"],
            survival_cache=survival_cache,
            margin_rate=0.70,
            annual_discount_rate=0.10,
            horizon_months=24,
        )
        for _, row in df.iterrows()
    ]

    df["clv_24m"] = np.round(clv_24m, 2)
    logger.info("Average 24-month forward CLV: $%.2f (Median: $%.2f)", df["clv_24m"].mean(), df["clv_24m"].median())
    return df


def run_clv_sensitivity_analysis(
    df: pd.DataFrame, survival_cache: Dict[Tuple[str, int, int], np.ndarray]
) -> pd.DataFrame:
    """Compute sensitivity matrix of average CLV across margin and discount assumptions."""
    margin_rates = [0.60, 0.70, 0.80]
    discount_rates = [0.08, 0.10, 0.12]
    horizons = [12, 24, 36]

    records = []
    for m in margin_rates:
        for d in discount_rates:
            for h in horizons:
                clvs = [
                    calculate_discounted_clv_fast(
                        monthly_charge=row["MonthlyCharges"],
                        current_tenure=int(row["tenure"]),
                        contract=row["Contract"],
                        survival_cache=survival_cache,
                        margin_rate=m,
                        annual_discount_rate=d,
                        horizon_months=h,
                    )
                    for _, row in df.iterrows()
                ]
                records.append({
                    "margin_rate": f"{int(m*100)}%",
                    "annual_discount_rate": f"{int(d*100)}%",
                    "horizon_months": f"{h}m",
                    "avg_clv": round(float(np.mean(clvs)), 2),
                })

    return pd.DataFrame(records)


def perform_customer_segmentation(df: pd.DataFrame) -> Tuple[KMeans, pd.DataFrame, Dict[str, Any]]:
    """Segment customers using K-Means clustering on behavioral and economic dimensions."""
    logger.info("Executing K-Means Customer Segmentation...")
    segment_features = [
        "tenure",
        "MonthlyCharges",
        "clv_24m",
        "churn_probability",
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[segment_features])

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["segment_id"] = kmeans.fit_predict(X_scaled)

    # Clustering evaluation metrics
    sil_score = float(silhouette_score(X_scaled, df["segment_id"]))
    db_score = float(davies_bouldin_score(X_scaled, df["segment_id"]))
    logger.info("Clustering metrics: Silhouette Score=%.3f, Davies-Bouldin=%.3f", sil_score, db_score)

    # Profile segments
    profile = df.groupby("segment_id").agg({
        "customerID": "count",
        "tenure": "mean",
        "MonthlyCharges": "mean",
        "clv_24m": "mean",
        "churn_probability": "mean",
    }).rename(columns={"customerID": "customer_count"}).round(2)

    # Map clusters to clear business labels based on centroids:
    # 0: Long tenure (58m), High spend ($90), Low churn (24%) -> High-Value Loyalists
    # 1: Long tenure (45m), Low spend ($31), Lowest churn (14%) -> Budget Steady Anchors
    # 2: Mid tenure (17m), High spend ($85), Highest churn (63%) -> Vulnerable High-Spenders
    # 3: Short tenure (9m), Low/Mid spend ($38), High churn (43%) -> New Onboarding Risks
    segment_names = {}
    for seg_id, row in profile.iterrows():
        if row["churn_probability"] >= 0.50:
            segment_names[seg_id] = "Vulnerable High-Spenders"
        elif row["MonthlyCharges"] >= 60 and row["clv_24m"] >= 900:
            segment_names[seg_id] = "High-Value Loyalists"
        elif row["tenure"] >= 30 and row["MonthlyCharges"] < 60:
            segment_names[seg_id] = "Budget Steady Anchors"
        else:
            segment_names[seg_id] = "New Onboarding Risks"

    df["segment_name"] = df["segment_id"].map(segment_names)
    profile["segment_name"] = profile.index.map(segment_names)

    clustering_metadata = {
        "silhouette_score": round(sil_score, 4),
        "davies_bouldin_index": round(db_score, 4),
        "features": segment_features,
        "segment_names": segment_names,
        "profile": profile.to_dict(orient="records"),
    }

    return kmeans, df, clustering_metadata


def plot_clv_and_segmentation_figures(
    df: pd.DataFrame, sensitivity_df: pd.DataFrame, output_dir: Path
):
    """Generate and save CLV distributions, sensitivity heatmap, and segmentation scatter."""
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Forward CLV Distribution Histogram / KDE
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["clv_24m"], kde=True, color="#2b5c8f", bins=30, ax=ax)
    ax.axvline(df["clv_24m"].mean(), color="#d9534f", linestyle="--", linewidth=2, label=f"Mean CLV: ${df['clv_24m'].mean():.2f}")
    ax.axvline(df["clv_24m"].median(), color="#5cb85c", linestyle=":", linewidth=2, label=f"Median CLV: ${df['clv_24m'].median():.2f}")
    ax.set_title("Forward 24-Month Customer Lifetime Value (CLV) Distribution", fontweight="bold", pad=12)
    ax.set_xlabel("Expected 24-Month CLV ($) [Margin: 70%, WACC: 10%]")
    ax.set_ylabel("Customer Count")
    ax.legend(loc="upper right")
    fig.savefig(output_dir / "19_clv_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Sensitivity Analysis Heatmap (24m Horizon)
    fig, ax = plt.subplots(figsize=(7, 5))
    heat_df = sensitivity_df[sensitivity_df["horizon_months"] == "24m"].pivot(
        index="margin_rate", columns="annual_discount_rate", values="avg_clv"
    )
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Average CLV ($)"}, ax=ax)
    ax.set_title("CLV Sensitivity: Contribution Margin vs. Discount Rate (24m)", fontweight="bold", pad=12)
    ax.set_xlabel("Annual Discount Rate (%)")
    ax.set_ylabel("Contribution Margin (%)")
    fig.savefig(output_dir / "20_clv_sensitivity_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Customer Segments Scatter Plot (Churn Risk vs Forward CLV)
    fig, ax = plt.subplots(figsize=(9, 6))
    unique_segments = sorted(df["segment_name"].unique())
    sns.scatterplot(
        data=df,
        x="clv_24m",
        y="churn_probability",
        hue="segment_name",
        palette="tab10",
        alpha=0.6,
        s=40,
        ax=ax,
    )
    ax.axhline(0.28, color="#d9534f", linestyle="--", alpha=0.7, label="Churn Decision Threshold (0.28)")
    ax.set_title("Customer Segmentation: 24-Month CLV vs. Predicted Churn Risk", fontweight="bold", pad=12)
    ax.set_xlabel("Expected 24-Month CLV ($)")
    ax.set_ylabel("Predicted Churn Probability")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(output_dir / "21_customer_segments_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Segment Average Profiles Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    seg_summary = df.groupby("segment_name")[["clv_24m", "MonthlyCharges", "tenure"]].mean().reset_index()
    plot_df = pd.melt(seg_summary, id_vars="segment_name", var_name="Metric", value_name="Average")
    sns.barplot(data=plot_df, x="segment_name", y="Average", hue="Metric", ax=ax)
    ax.set_title("Segment Comparison: Tenure, Monthly Charges, and Expected CLV", fontweight="bold", pad=12)
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Average Metric Value")
    ax.legend(title="Metric")
    plt.xticks(rotation=15)
    fig.savefig(output_dir / "22_segment_profiles_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved 4 CLV and segmentation figures to %s", output_dir)


def run_clv_pipeline() -> Dict[str, Any]:
    """Execute end-to-end CLV estimation, sensitivity analysis, and segmentation."""
    df, churn_model, km_contract = load_data_and_models()

    # 1. Build fast survival lookup cache
    survival_cache = build_survival_lookup_cache(km_contract)

    # 2. Compute CLV and Churn Risk
    df_enriched = enrich_dataset_with_clv_and_risk(df, churn_model, survival_cache)

    # 3. Run Sensitivity Analysis
    sensitivity_df = run_clv_sensitivity_analysis(df_enriched, survival_cache)

    # 4. Perform Segmentation
    kmeans_model, df_segmented, cluster_meta = perform_customer_segmentation(df_enriched)

    # 5. Generate Diagnostic Figures
    plot_clv_and_segmentation_figures(df_segmented, sensitivity_df, FIGURES_DIR)

    # 6. Save Artifacts
    enriched_output_path = PROCESSED_DATA_DIR / "telco_customer_clv_segmented.csv"
    df_segmented.to_csv(enriched_output_path, index=False)
    logger.info("Saved segmented customer dataset with CLV to %s", enriched_output_path)

    joblib.dump(kmeans_model, MODELS_DIR / "segmentation_kmeans.joblib")

    summary_data = {
        "overall_clv": {
            "mean_clv_24m": round(float(df_segmented["clv_24m"].mean()), 2),
            "median_clv_24m": round(float(df_segmented["clv_24m"].median()), 2),
            "total_portfolio_clv_24m": round(float(df_segmented["clv_24m"].sum()), 2),
            "mean_historical_profit": round(float(df_segmented["historical_realized_profit"].mean()), 2),
        },
        "clustering_evaluation": cluster_meta,
        "sensitivity_sample": sensitivity_df.to_dict(orient="records"),
    }
    with open(MODELS_DIR / "clv_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING CUSTOMER LIFETIME VALUE & SEGMENTATION PIPELINE")
    print("=" * 60)
    results = run_clv_pipeline()

    print("\nOVERALL PORTFOLIO CLV SUMMARY:")
    for k, v in results["overall_clv"].items():
        print(f" - {k}: ${v:,.2f}" if isinstance(v, (int, float)) else f" - {k}: {v}")

    print("\nCUSTOMER SEGMENTS:")
    for profile in results["clustering_evaluation"]["profile"]:
        print(f"\n[{profile['segment_name'].upper()}]")
        print(f" - Customers: {profile['customer_count']:,}")
        print(f" - Mean Tenure: {profile['tenure']} months")
        print(f" - Mean Monthly Charge: ${profile['MonthlyCharges']:.2f}")
        print(f" - Expected 24m CLV: ${profile['clv_24m']:.2f}")
        print(f" - Churn Risk: {profile['churn_probability']*100:.1f}%")
    print("=" * 60)
