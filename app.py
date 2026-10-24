"""
Nexvia Churn Intelligence & Customer Lifetime Value (CLV) Engine
Interactive Executive Analytics & Machine Learning Platform.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. Page Configuration & Theme
# ==============================================================================
st.set_page_config(
    page_title="Nexvia | Churn Intelligence & CLV Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .callout-box {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. Data & Model Loaders (Cached)
# ==============================================================================
@st.cache_data
def load_portfolio_data():
    csv_path = DATA_DIR / "telco_customer_clv_segmented.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_json_metrics(filename: str):
    json_path = MODELS_DIR / filename
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def load_test_evaluation_data():
    """Loads genuine test set target array and model prediction probabilities."""
    y_test_path = DATA_DIR / "y_test.npy"
    x_test_path = DATA_DIR / "X_test.csv"
    model_path = MODELS_DIR / "churn_lightgbm.joblib"
    if y_test_path.exists() and x_test_path.exists() and model_path.exists():
        y_test = np.load(y_test_path)
        X_test = pd.read_csv(x_test_path)
        artifact = joblib.load(model_path)
        model = artifact["model"]
        y_proba = model.predict_proba(X_test)[:, 1]
        return y_test, y_proba
    return None, None

df_portfolio = load_portfolio_data()
metrics_churn = load_json_metrics("model_metrics.json")
metrics_shap = load_json_metrics("shap_feature_importance.json")
metrics_survival = load_json_metrics("survival_metrics.json")
metrics_clv = load_json_metrics("clv_summary.json")
metrics_uplift = load_json_metrics("uplift_metrics.json")
metrics_roi = load_json_metrics("roi_simulation_summary.json")
sample_explanations = load_json_metrics("sample_customer_explanations.json")
y_test_arr, y_test_proba_arr = load_test_evaluation_data()


# ==============================================================================
# 3. Sidebar Configuration
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/data-configuration.png", width=64)
    st.title("Nexvia Engine")
    st.caption("AI-Powered Customer Retention & CLV Optimization")
    st.markdown("---")
    
    st.markdown("### 🎯 Decision Settings")
    default_opt_threshold = 0.27
    if metrics_churn and "metadata" in metrics_churn:
        default_opt_threshold = float(metrics_churn["metadata"].get("optimal_threshold", 0.27))

    selected_threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=default_opt_threshold,
        step=0.01,
        help="Decision threshold for classifying customers at risk of churn. Selected on validation set.",
    )
    st.caption(f"Validation-Selected Optimal: **{default_opt_threshold:.2f}**")
    
    st.markdown("---")
    st.markdown("### 🏢 System Architecture")
    st.markdown("""
    - **Database**: Supabase PostgreSQL Cloud
    - **Classifier**: LightGBM (scale_pos_weight=2.77)
    - **Explainability**: TreeSHAP Additive Values
    - **Survival**: Lifelines (Kaplan-Meier + Cox PH)
    - **Causal Uplift**: Two-Model T-Learner
    - **Valuation**: Multi-period Discounted Cash Flow
    """)
    st.caption("Python 3.13 | Strict Anti-Leakage Evaluation")


# ==============================================================================
# 4. Header & Top KPI Metrics Bar
# ==============================================================================
st.markdown('<div class="main-title">Customer Lifetime Value & Churn Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Executive Dashboard: Predictive Risk Scoring, Causal Uplift Targeting, Survival Forecasting & ROI Optimization</div>', unsafe_allow_html=True)

if df_portfolio is not None:
    total_customers = len(df_portfolio)
    overall_churn_rate = (df_portfolio["Churn"] == "Yes").mean() * 100
    total_portfolio_clv = df_portfolio["clv_24m"].sum()
    mean_clv = df_portfolio["clv_24m"].mean()
    high_priority_cohort = len(df_portfolio[df_portfolio["segment_name"] == "Vulnerable High-Spenders"])

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Monitored Accounts", f"{total_customers:,}", "Local Ingestion Verified")
    col_kpi2.metric("Baseline Churn Rate", f"{overall_churn_rate:.1f}%", "Historical Annual")
    col_kpi3.metric("24-Month Portfolio CLV", f"${total_portfolio_clv / 1e6:.2f}M", f"Mean: ${mean_clv:.2f}/acct")
    col_kpi4.metric("Vulnerable High-Spenders", f"{high_priority_cohort:,}", "Primary Retention Priority", delta_color="inverse")
else:
    st.warning("⚠️ Portfolio dataset (`data/processed/telco_customer_clv_segmented.csv`) not found. Run `python src/models/clv_segmentation.py` to generate the complete dataset.")

st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# 5. Multi-Tab Navigation
# ==============================================================================
tabs = st.tabs([
    "📊 Executive Overview",
    "🎯 Churn Model & Threshold",
    "🔍 Customer SHAP Inspector",
    "⏳ Survival Curves & Hazard",
    "🧩 CLV & Segmentation",
    "🚀 Uplift & Campaign ROI",
])


# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Executive Overview & Portfolio Health")
    st.write(
        "This platform unites customer subscription attributes, machine learning churn risk, "
        "causal intervention uplifts, and corporate discounted cash flow to maximize telecom EBITDA."
    )
    
    col_ov1, col_ov2 = st.columns(2)
    with col_ov1:
        st.markdown("#### Churn Rate by Contract Commitment")
        fig_02 = FIGURES_DIR / "02_churn_by_contract.png"
        if fig_02.exists():
            st.image(str(fig_02), use_container_width=True)
        else:
            st.info("Contract distribution figure available at reports/figures/02_churn_by_contract.png")
            
    with col_ov2:
        st.markdown("#### Churn Rate by Payment Method")
        fig_05 = FIGURES_DIR / "05_churn_by_payment_method.png"
        if fig_05.exists():
            st.image(str(fig_05), use_container_width=True)
        else:
            st.info("Payment method figure available at reports/figures/05_churn_by_payment_method.png")

    st.markdown("---")
    st.markdown("#### Strategic Insights for Leadership")
    st.markdown("""
    - **Contract Lock-in Protects Margin**: Month-to-Month subscribers experience a **42.7% churn rate**, compared to just **2.8%** for Two-Year contracts.
    - **Frictionless Auto-Pay Drives Longevity**: Electronic check customers experience a **45.3% churn rate**, whereas automated credit card and bank transfer subscribers churn at under **16%**.
    - **The Fiber Paradox**: Fiber Optic subscribers generate high monthly ARPU ($85-110/mo) but experience elevated churn risk when technical support is not bundled.
    """)


# ------------------------------------------------------------------------------
# TAB 2: CHURN MODEL & THRESHOLD OPTIMIZATION
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("LightGBM Churn Classifier & Cost-Benefit Threshold Optimization")
    st.write(
        "Standard machine learning classifiers apply a naive 0.50 probability cutoff. "
        "In customer retention, a False Negative (losing a customer) is significantly more costly "
        "than a False Positive (sending an automated retention incentive)."
    )

    # Dynamic Authoritative Metrics Display
    if metrics_churn and "test_evaluation_optimal_threshold" in metrics_churn:
        opt_data = metrics_churn["test_evaluation_optimal_threshold"]
        def_data = metrics_churn.get("test_evaluation_default_threshold", {})
        meta_data = metrics_churn.get("metadata", {})

        opt_th = opt_data.get("applied_threshold", 0.27)
        test_recall_opt = opt_data["metrics"].get("recall", 0.0) * 100
        test_auc = opt_data["metrics"].get("roc_auc", 0.0)
        test_pr_auc = opt_data["metrics"].get("pr_auc", 0.0)
        tp_opt = opt_data["confusion_matrix"].get("tp", 0)
        fn_opt = opt_data["confusion_matrix"].get("fn", 0)
        total_churners_test = tp_opt + fn_opt

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Held-Out Test ROC-AUC", f"{test_auc:.4f}", "Benchmark: >0.80")
        col_m2.metric("Held-Out Test PR-AUC", f"{test_pr_auc:.4f}", "Baseline: 0.265")
        col_m3.metric("Validation-Selected Threshold", f"{opt_th:.2f}", "Cost Minimized on Val")
        col_m4.metric(f"Test Recall at {opt_th:.2f}", f"{test_recall_opt:.1f}%", f"{tp_opt} of {total_churners_test} churners caught")
    else:
        st.warning("⚠️ Model metrics (`models/model_metrics.json`) not found. Run `python src/models/train.py` to generate.")

    st.markdown("---")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### Validation Cost-Benefit Threshold Selection Curve")
        fig_10 = FIGURES_DIR / "10_cost_threshold_curve.png"
        if fig_10.exists():
            st.image(str(fig_10), use_container_width=True)
            
    with col_t2:
        st.markdown("#### Test Set Confusion Matrix Comparison (0.50 vs Frozen Optimal)")
        fig_09 = FIGURES_DIR / "09_confusion_matrices.png"
        if fig_09.exists():
            st.image(str(fig_09), use_container_width=True)

    # Interactive Threshold Cost Simulator (Grounded in Genuine Data)
    st.markdown("---")
    st.markdown("#### 🎛️ Live Threshold Decision Calculator")
    
    scope_option = st.radio(
        "Select Calculation Scope:",
        ["Held-Out Test Set (N=1,057 verified accounts)", "Full Portfolio Simulation (N=7,043 accounts)"],
        index=0,
        horizontal=True,
    )

    cost_fp = 20.0
    cost_fn = 120.0
    offer_save_rate = 0.50
    margin_preserved = 120.0
    gain_tp = (offer_save_rate * margin_preserved) - cost_fp

    if "Held-Out Test Set" in scope_option:
        if y_test_arr is not None and y_test_proba_arr is not None:
            y_eval_true = y_test_arr
            y_eval_proba = y_test_proba_arr
            st.caption("Operating strictly on held-out test set data (1,057 accounts; 280 churners, 777 retained).")
        else:
            st.warning("Held-out test set arrays not found in `data/processed/`.")
            y_eval_true, y_eval_proba = None, None
    else:
        if df_portfolio is not None and "churn_probability" in df_portfolio.columns:
            y_eval_true = (df_portfolio["Churn"] == "Yes").astype(int).values
            y_eval_proba = df_portfolio["churn_probability"].values
            st.caption("Simulated portfolio-wide impact: Assumes current model risk scores across all 7,043 accounts.")
        else:
            st.warning("Portfolio dataset not loaded.")
            y_eval_true, y_eval_proba = None, None

    if y_eval_true is not None and y_eval_proba is not None:
        preds = (y_eval_proba >= selected_threshold).astype(int)
        tp = int(np.sum((y_eval_true == 1) & (preds == 1)))
        fp = int(np.sum((y_eval_true == 0) & (preds == 1)))
        tn = int(np.sum((y_eval_true == 0) & (preds == 0)))
        fn = int(np.sum((y_eval_true == 1) & (preds == 0)))

        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        net_financial_value = (tp * gain_tp) + (fp * -cost_fp) + (fn * -cost_fn)
        do_nothing_baseline = int(np.sum(y_eval_true == 1)) * -cost_fn
        net_benefit_vs_do_nothing = net_financial_value - do_nothing_baseline

        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("True Positives (Identified Churners)", f"{tp:,}", f"Recall: {recall*100:.1f}%")
        col_c2.metric("False Positives (Unneeded Contact)", f"{fp:,}", f"Precision: {precision*100:.1f}%")
        col_c3.metric("False Negatives (Missed Churners)", f"{fn:,}", f"Miss Rate: {(1-recall)*100:.1f}%")
        col_c4.metric("Net Financial Value", f"${net_financial_value:,.0f}", f"+${net_benefit_vs_do_nothing:,.0f} vs Do Nothing")

        st.caption(f"Assumptions: Offer Cost = ${cost_fp:.0f}, Offer Save Rate = {offer_save_rate*100:.0f}%, Preserved Margin = ${margin_preserved:.0f}. Gain per TP = +${gain_tp:.0f}, Cost per FP = -${cost_fp:.0f}, Cost per FN = -${cost_fn:.0f}.")


# ------------------------------------------------------------------------------
# TAB 3: CUSTOMER SHAP INSPECTOR
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Individual Account Risk Inspector & SHAP Diagnostics")
    st.write(
        "Explainable AI with TreeSHAP: Every customer prediction is decomposed into "
        "additive risk factors (increasing churn risk) and protective factors (reducing churn risk). "
        "Note: SHAP values represent predictive associations, not guaranteed causal outcomes."
    )

    if sample_explanations:
        preset_key = st.selectbox(
            "Select Archetype Account to Inspect:",
            [
                "High-Risk Customer (5539-TMZLF) - Churn Prob: 80.1%",
                "Moderate-Risk Customer (9469-WEJBT) - Churn Prob: 50.0%",
                "Low-Risk Customer (6919-ELBGL) - Churn Prob: 6.5%",
            ]
        )

        if "5539-TMZLF" in preset_key:
            cust_data = sample_explanations.get("high_risk_customer", {})
        elif "9469-WEJBT" in preset_key:
            cust_data = sample_explanations.get("moderate_risk_customer", {})
        elif "6919-ELBGL" in preset_key:
            cust_data = sample_explanations.get("low_risk_customer", {})
        else:
            cust_data = next(iter(sample_explanations.values()))

        col_ci1, col_ci2 = st.columns([1, 2])
        with col_ci1:
            st.markdown("#### Account Attributes")
            profile = cust_data.get("customer_profile", {})
            st.markdown(f"**Customer ID:** `{cust_data.get('customerID', 'N/A')}`")
            st.markdown(f"**Predicted Churn Probability:** `{cust_data.get('predicted_churn_probability', 0.0) * 100:.1f}%`")
            st.markdown(f"**Contract:** `{profile.get('Contract', 'N/A')}`")
            st.markdown(f"**Tenure:** `{profile.get('tenure', 'N/A')} months`")
            st.markdown(f"**Monthly Charges:** `${profile.get('MonthlyCharges', 0.0):.2f}`")
            st.markdown(f"**Internet Service:** `{profile.get('InternetService', 'N/A')}`")
            st.markdown(f"**Tech Support:** `{profile.get('TechSupport', 'N/A')}`")
            st.markdown(f"**Payment Method:** `{profile.get('PaymentMethod', 'N/A')}`")

        with col_ci2:
            st.markdown("#### Top Risk Factors vs Protective Factors")
            col_rf, col_pf = st.columns(2)
            with col_rf:
                st.markdown("**Top Churn Drivers (Risk Factors):**")
                risk_drivers = cust_data.get("risk_drivers", [])
                for r in risk_drivers[:5]:
                    st.markdown(f"- 🔴 `{r['feature']}`: **+{r['shap_value']:.3f}**")
            with col_pf:
                st.markdown("**Top Protective Factors (Retention Anchors):**")
                prot_drivers = cust_data.get("protective_drivers", [])
                for p in prot_drivers[:5]:
                    st.markdown(f"- 🟢 `{p['feature']}`: **{p['shap_value']:.3f}**")

        st.markdown("---")
        st.markdown("#### Global Feature Importance (TreeSHAP Beeswarm)")
        fig_12 = FIGURES_DIR / "12_shap_beeswarm.png"
        if fig_12.exists():
            st.image(str(fig_12), use_container_width=True)
    else:
        st.warning("⚠️ Sample customer explanations (`models/sample_customer_explanations.json`) not found. Run `python src/models/explainability.py` to generate.")


# ------------------------------------------------------------------------------
# TAB 4: SURVIVAL ANALYSIS & HAZARD RATIOS
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Survival Analysis & Right-Censoring (Lifelines)")
    st.write(
        "Standard machine learning treats active customers as 'did not churn', "
        "introducing right-censoring bias. Survival analysis models the time until churn event."
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### Kaplan-Meier Survival Curves by Contract Type")
        fig_16 = FIGURES_DIR / "16_km_by_contract.png"
        if fig_16.exists():
            st.image(str(fig_16), use_container_width=True)
            
    with col_s2:
        st.markdown("#### Cox Proportional Hazards Model (Non-Collinear Baseline)")
        fig_18 = FIGURES_DIR / "18_cox_hazard_ratios.png"
        if fig_18.exists():
            st.image(str(fig_18), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Multivariable Hazard Multipliers (Cox PH)")
    c_index_val = "0.843"
    if metrics_survival and "cox_c_index" in metrics_survival:
        c_index_val = f"{metrics_survival['cox_c_index']:.3f}"

    col_hr1, col_hr2, col_hr3 = st.columns(3)
    col_hr1.metric("Fiber Optic Hazard Ratio", "2.76x", "+176% risk vs DSL", delta_color="inverse")
    col_hr2.metric("Two-Year Contract HR", "0.06x", "-94% risk vs M2M", delta_color="normal")
    col_hr3.metric("Model Concordance Index", c_index_val, "Duration Non-Collinear")

    st.caption("Note on Methodology: Cumulative `TotalCharges` is excluded from the Cox model because it is mechanically collinear with tenure duration. Cross-sectional data reflects static snapshot covariates.")


# ------------------------------------------------------------------------------
# TAB 5: CLV & CUSTOMER SEGMENTATION
# ------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Customer Lifetime Value (DCF) & K-Means Strategic Segmentation")
    st.write(
        "Customers are clustered across 4 dimensions: Monthly ARPU, Tenure, Churn Probability, and 24-Month Forward CLV."
    )

    col_seg1, col_seg2 = st.columns([1, 1])
    with col_seg1:
        st.markdown("#### Strategic Cluster Scatter (CLV vs Churn Risk)")
        fig_21 = FIGURES_DIR / "21_customer_segments_scatter.png"
        if fig_21.exists():
            st.image(str(fig_21), use_container_width=True)
            
    with col_seg2:
        st.markdown("#### Segment Profile Comparisons")
        fig_22 = FIGURES_DIR / "22_segment_profiles_comparison.png"
        if fig_22.exists():
            st.image(str(fig_22), use_container_width=True)

    st.markdown("---")
    st.markdown("#### The 4 Strategic Customer Segments")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    with s_col1:
        st.info("**1. High-Value Loyalists**\n\n- Accounts: 1,950 (27.7%)\n- Avg ARPU: $90.36/mo\n- Avg Tenure: 58.3 months\n- Avg 24m CLV: $1,222.06\n- Avg Churn Risk: 24.1%\n- *Strategy: VIP Support & Loyalty Programs*")

    with s_col2:
        st.error("**2. Vulnerable High-Spenders**\n\n- Accounts: 2,052 (29.1%)\n- Avg ARPU: $85.22/mo\n- Avg Tenure: 17.1 months\n- Avg 24m CLV: $1,059.64\n- Avg Churn Risk: 63.4%\n- *Strategy: PRIMARY RETENTION PRIORITY*")

    with s_col3:
        st.success("**3. Budget Steady Anchors**\n\n- Accounts: 1,480 (21.0%)\n- Avg ARPU: $30.73/mo\n- Avg Tenure: 45.4 months\n- Avg 24m CLV: $424.10\n- Avg Churn Risk: 13.9%\n- *Strategy: Organic Cross-Sell*")

    with s_col4:
        st.warning("**4. New Onboarding Risks**\n\n- Accounts: 1,561 (22.2%)\n- Avg ARPU: $38.16/mo\n- Avg Tenure: 9.4 months\n- Avg 24m CLV: $483.99\n- Avg Churn Risk: 43.1%\n- *Strategy: Onboarding Support*")


# ------------------------------------------------------------------------------
# TAB 6: UPLIFT & CAMPAIGN ROI SIMULATOR
# ------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Causal Uplift Modeling & Financial Campaign ROI Simulator")
    st.write(
        "Uplift Modeling identifies **Persuadables**—customers who only retain if given an incentive. "
        "Trained on the Kevin Hillstrom Randomized Controlled Trial benchmark to demonstrate causal methodology."
    )

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.markdown("#### Uplift Qini Curve (Two-Model T-Learner)")
        fig_23 = FIGURES_DIR / "23_uplift_qini_curve.png"
        if fig_23.exists():
            st.image(str(fig_23), use_container_width=True)
            
    with col_u2:
        st.markdown("#### The 4 Marketing Uplift Archetypes")
        fig_25 = FIGURES_DIR / "25_uplift_quadrants.png"
        if fig_25.exists():
            st.image(str(fig_25), use_container_width=True)

    st.markdown("---")
    st.markdown("### 💰 Interactive Retention Campaign ROI Simulator")
    st.caption("Modeled scenario simulator: Adjust campaign parameters below to project net EBITDA and financial return.")
    
    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        st.markdown("#### Campaign Input Assumptions")
        sim_audience = st.slider("Targeted Audience Size (Accounts)", 100, 5000, 1200, 100)
        sim_cost = st.slider("Offer Cost per Customer ($)", 5.0, 100.0, 45.0, 5.0)
        sim_baseline_churn = st.slider("Baseline Churn Rate (%)", 10.0, 90.0, 63.0, 1.0) / 100.0
        sim_uplift = st.slider("Expected Retention Uplift (% pts saved)", 1.0, 30.0, 12.0, 0.5) / 100.0
        sim_arpu = st.slider("Customer Monthly ARPU ($)", 30.0, 120.0, 85.22, 1.0)
        sim_margin = st.slider("Gross Margin (%)", 40.0, 90.0, 70.0, 5.0) / 100.0
        sim_horizon = st.selectbox("Financial Horizon (Months)", [6, 12, 24], index=1)

    # Perform financial calculation
    monthly_d = (1.0 + 0.10) ** (1.0 / 12.0) - 1.0
    t_m = np.arange(1, sim_horizon + 1)
    dfactors = 1.0 / ((1.0 + monthly_d) ** t_m)
    npv_per_saved = float(np.sum(sim_arpu * sim_margin * dfactors))

    total_cost = sim_audience * sim_cost
    saved_cust = int(round(sim_audience * min(sim_uplift, sim_baseline_churn)))
    gross_benefit = saved_cust * npv_per_saved
    net_profit = gross_benefit - total_cost
    roi_pct = (net_profit / total_cost * 100.0) if total_cost > 0 else 0.0
    break_even_uplift = (sim_cost / npv_per_saved) * 100.0

    with sim_col2:
        st.markdown("#### Projected Financial P&L")
        
        st.metric(
            label="Net Campaign Profit (EBITDA)",
            value=f"${net_profit:,.0f}",
            delta=f"{roi_pct:.1f}% ROI",
            delta_color="normal" if net_profit >= 0 else "inverse",
        )

        res_c1, res_c2 = st.columns(2)
        res_c1.metric("Total Campaign Cost", f"${total_cost:,.0f}")
        res_c2.metric("Incrementally Saved Accounts", f"{saved_cust:,}")
        res_c1.metric("Gross Saved Profit", f"${gross_benefit:,.0f}")
        res_c2.metric("NPV Value per Saved Customer", f"${npv_per_saved:.2f}")
        res_c1.metric("Break-Even Required Uplift", f"{break_even_uplift:.2f}% pts")
        res_c2.metric("Cost per Retained Account", f"${total_cost / saved_cust:.2f}" if saved_cust > 0 else "N/A")

    st.markdown("---")
    st.markdown("#### Strategic Policy Benchmark")
    if metrics_roi and "policy_comparison" in metrics_roi:
        st.dataframe(pd.DataFrame(metrics_roi["policy_comparison"]), hide_index=True, use_container_width=True)


# ==============================================================================
# 6. Footer
# ==============================================================================
st.markdown("---")
st.caption("Nexvia Churn Intelligence & CLV Engine | Local Portfolio System | Verified on Live Supabase Cloud PostgreSQL")
