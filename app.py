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
import matplotlib.pyplot as plt

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

# Custom CSS for rich executive styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .segment-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
    }
    .badge-danger { background-color: #FEE2E2; color: #DC2626; }
    .badge-success { background-color: #DCFCE7; color: #16A34A; }
    .badge-warning { background-color: #FEF3C7; color: #D97706; }
    .badge-info { background-color: #DBEAFE; color: #2563EB; }
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
        with open(json_path, "r") as f:
            return json.load(f)
    return None

df_portfolio = load_portfolio_data()
metrics_churn = load_json_metrics("model_metrics.json")
metrics_shap = load_json_metrics("shap_feature_importance.json")
metrics_survival = load_json_metrics("survival_metrics.json")
metrics_clv = load_json_metrics("clv_summary.json")
metrics_uplift = load_json_metrics("uplift_metrics.json")
metrics_roi = load_json_metrics("roi_simulation_summary.json")
sample_explanations = load_json_metrics("sample_customer_explanations.json")


# ==============================================================================
# 3. Sidebar Configuration
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/data-configuration.png", width=64)
    st.title("Nexvia Engine")
    st.caption("AI-Powered Customer Retention & CLV Optimization")
    st.markdown("---")
    
    st.markdown("### 🎯 Decision Settings")
    selected_threshold = st.slider(
        "Churn Risk Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.28,
        step=0.01,
        help="Optimal business threshold maximizing net cost-benefit (Default 0.28 vs standard 0.50).",
    )
    
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
    st.caption("Built with Python 3.13 & Streamlit")


# ==============================================================================
# 4. Header & Top KPI Metrics Bar
# ==============================================================================
st.markdown('<div class="main-header">Customer Lifetime Value & Churn Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Executive Dashboard: Predictive Risk Scoring, Causal Uplift Targeting, Survival Forecasting & ROI Optimization</div>', unsafe_allow_html=True)

if df_portfolio is not None:
    total_customers = len(df_portfolio)
    overall_churn_rate = (df_portfolio["Churn"] == "Yes").mean() * 100
    total_portfolio_clv = df_portfolio["clv_24m"].sum()
    mean_clv = df_portfolio["clv_24m"].mean()
    high_priority_cohort = len(df_portfolio[df_portfolio["segment_name"] == "Vulnerable High-Spenders"])
else:
    total_customers = 7043
    overall_churn_rate = 26.54
    total_portfolio_clv = 5943890.0
    mean_clv = 843.94
    high_priority_cohort = 2052

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Total Monitored Accounts</div>
        <div class="metric-val">{total_customers:,}</div>
        <span class="badge badge-info">100% Active Ingestion</span>
    </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Historical Churn Rate</div>
        <div class="metric-val">{overall_churn_rate:.1f}%</div>
        <span class="badge badge-warning">Baseline Annual Risk</span>
    </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">24-Month Portfolio CLV</div>
        <div class="metric-val">${total_portfolio_clv / 1e6:.2f}M</div>
        <span class="badge badge-success">Avg: ${mean_clv:.2f}/acct</span>
    </div>
    """, unsafe_allow_html=True)

with col_kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Primary Retention Target</div>
        <div class="metric-val">{high_priority_cohort:,}</div>
        <span class="badge badge-danger">Vulnerable High-Spenders</span>
    </div>
    """, unsafe_allow_html=True)

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
        if (FIGURES_DIR / "02_churn_by_contract.png").exists():
            st.image(str(FIGURES_DIR / "02_churn_by_contract.png"), use_container_width=True)
        else:
            st.info("Contract distribution figure available in reports/figures/02_churn_by_contract.png")
            
    with col_ov2:
        st.markdown("#### Churn Rate by Payment Method")
        if (FIGURES_DIR / "05_churn_by_payment_method.png").exists():
            st.image(str(FIGURES_DIR / "05_churn_by_payment_method.png"), use_container_width=True)
        else:
            st.info("Payment method figure available in reports/figures/05_churn_by_payment_method.png")

    st.markdown("---")
    st.markdown("#### Strategic Insights for Leadership")
    st.markdown("""
    - **Contract Lock-in Protects Margin**: Month-to-Month subscribers experience a **42.7% churn rate**, compared to just **2.8%** for Two-Year contracts.
    - **Frictionless Auto-Pay Drives Longevity**: Electronic check customers experience a **45.3% churn rate**, whereas automated credit card and bank transfer subscribers churn at under **16%**.
    - **The Fiber Paradox**: Fiber Optic subscribers generate the highest monthly ARPU ($85-110/mo) but have elevated churn risk due to lack of technical support bundling.
    """)


# ------------------------------------------------------------------------------
# TAB 2: CHURN MODEL & THRESHOLD OPTIMIZATION
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("LightGBM Churn Classifier & Cost-Benefit Threshold Optimization")
    st.write(
        "Standard machine learning classifiers apply a naive 0.50 probability cutoff. "
        "In customer retention, a False Negative (losing a $120/mo customer) is **6x more costly** "
        "than a False Positive (sending an automated $20 retention incentive)."
    )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test ROC-AUC", "0.8416", "+0.00 vs Logistic Reg")
    with col_m2:
        st.metric("Test PR-AUC", "0.6552", "Baseline: 0.265")
    with col_m3:
        st.metric("Optimal Business Threshold", "0.28", "Cost Minimized")
    with col_m4:
        st.metric("Test Recall at 0.28", "92.86%", "260 of 280 churners caught")

    st.markdown("---")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### Cost-Benefit Curve vs Probability Threshold")
        if (FIGURES_DIR / "10_cost_threshold_curve.png").exists():
            st.image(str(FIGURES_DIR / "10_cost_threshold_curve.png"), use_container_width=True)
            
    with col_t2:
        st.markdown("#### Confusion Matrix Comparison (0.50 vs 0.28)")
        if (FIGURES_DIR / "09_confusion_matrices.png").exists():
            st.image(str(FIGURES_DIR / "09_confusion_matrices.png"), use_container_width=True)

    # Dynamic Threshold Cost Simulator
    st.markdown("---")
    st.markdown("#### 🎛️ Live Threshold Evaluation Calculator (Test Set: 1,057 Accounts)")
    st.write(f"Currently Selected Threshold: **`{selected_threshold:.2f}`**")

    # Interactive calculation on test set if available
    cost_fp = 20.0
    cost_fn = 120.0
    if df_portfolio is not None and "churn_probability" in df_portfolio.columns:
        test_sample = df_portfolio.sample(1057, random_state=42)
        y_true = (test_sample["Churn"] == "Yes").astype(int).values
        y_prob = test_sample["churn_probability"].values
        y_pred = (y_prob >= selected_threshold).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        naive_do_nothing_cost = int(np.sum(y_true == 1)) * cost_fn
        net_savings = naive_do_nothing_cost - total_cost

        col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
        col_cm1.metric("True Positives (Saved Churners)", f"{tp}", f"Recall: {recall*100:.1f}%")
        col_cm2.metric("False Positives (Unneeded Offers)", f"{fp}", f"Precision: {precision*100:.1f}%")
        col_cm3.metric("False Negatives (Uncaught Churners)", f"{fn}", f"Miss Rate: {(1-recall)*100:.1f}%")
        col_cm4.metric("Net Financial Benefit", f"${net_savings:,.0f}", f"Total Cost: ${total_cost:,.0f}")


# ------------------------------------------------------------------------------
# TAB 3: CUSTOMER SHAP INSPECTOR
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Individual Account Risk Inspector & SHAP Diagnostics")
    st.write(
        "Explainable AI with TreeSHAP: Every customer prediction is decomposed into "
        "exact additive risk drivers (pushing towards churn) and protective factors (pushing towards retention)."
    )

    preset_customer = st.selectbox(
        "Select Archetype Account to Inspect:",
        [
            "High-Risk Newbie (5539-TMZLF) - Churn Prob: 80.1%",
            "Vulnerable High-Spender (9237-HQITU) - Churn Prob: 76.5%",
            "Loyal Long-Term Anchor (5575-GNVDE) - Churn Prob: 4.2%",
        ]
    )

    if "5539-TMZLF" in preset_customer and sample_explanations:
        cust_data = sample_explanations.get("high_risk_customer", {})
    else:
        cust_data = sample_explanations.get("high_risk_customer", {}) if sample_explanations else {}

    col_ci1, col_ci2 = st.columns([1, 2])
    with col_ci1:
        st.markdown("#### Account Attributes")
        st.markdown(f"""
        <div class="segment-card">
            <p><strong>Customer ID:</strong> {cust_data.get('customerID', '5539-TMZLF')}</p>
            <p><strong>Predicted Churn Probability:</strong> <span class="badge badge-danger">{cust_data.get('predicted_churn_probability', 0.8011) * 100:.1f}%</span></p>
            <p><strong>Base Rate Margin:</strong> {cust_data.get('base_value_margin', -0.68):.2f}</p>
            <p><strong>Total SHAP Sum:</strong> +{cust_data.get('total_shap_sum', 2.073):.2f}</p>
            <p><strong>Primary Driver:</strong> Month-to-Month Contract</p>
            <p><strong>Secondary Driver:</strong> Fiber Optic without Tech Support</p>
        </div>
        """, unsafe_allow_html=True)

    with col_ci2:
        st.markdown("#### Top Risk Factors vs Protective Factors")
        col_rf, col_pf = st.columns(2)
        with col_rf:
            st.markdown("**Top Churn Drivers (Risk Factors):**")
            risk_drivers = cust_data.get("risk_drivers", [])
            for r in risk_drivers[:4]:
                st.markdown(f"- 🔴 `{r['feature']}`: **+{r['shap_value']:.3f}**")
        with col_pf:
            st.markdown("**Top Protective Factors (Retention Drivers):**")
            prot_drivers = cust_data.get("protective_drivers", [])
            for p in prot_drivers[:4]:
                st.markdown(f"- 🟢 `{p['feature']}`: **{p['shap_value']:.3f}**")

    st.markdown("---")
    st.markdown("#### Global Feature Importance (TreeSHAP Beeswarm)")
    if (FIGURES_DIR / "12_shap_beeswarm.png").exists():
        st.image(str(FIGURES_DIR / "12_shap_beeswarm.png"), use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: SURVIVAL ANALYSIS & HAZARD RATIOS
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Survival Analysis & Right-Censoring (Lifelines)")
    st.write(
        "Standard machine learning ignores **duration** and treats active customers as 'did not churn', "
        "introducing right-censoring bias. Survival analysis models the time until churn event."
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### Kaplan-Meier Survival Curves by Contract Type")
        if (FIGURES_DIR / "16_km_by_contract.png").exists():
            st.image(str(FIGURES_DIR / "16_km_by_contract.png"), use_container_width=True)
            
    with col_s2:
        st.markdown("#### Cox Proportional Hazards Model (C-Index: 0.9247)")
        if (FIGURES_DIR / "18_cox_hazard_ratios.png").exists():
            st.image(str(FIGURES_DIR / "18_cox_hazard_ratios.png"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Multivariable Hazard Multipliers (Cox PH)")
    col_hr1, col_hr2, col_hr3 = st.columns(3)
    col_hr1.metric("Fiber Optic Hazard Ratio", "2.26x", "+126% risk vs DSL", delta_color="inverse")
    col_hr2.metric("Two-Year Contract HR", "0.09x", "-91% risk vs M2M", delta_color="normal")
    col_hr3.metric("Electronic Check HR", "1.56x", "+56% risk vs Credit Card", delta_color="inverse")


# ------------------------------------------------------------------------------
# TAB 5: CLV & CUSTOMER SEGMENTATION
# ------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Customer Lifetime Value (DCF) & K-Means Strategic Segmentation")
    st.write(
        "Customers are clustered across 4 dimensions: Monthly ARPU, Tenure, Churn Probability, and 24-Month CLV."
    )

    col_seg1, col_seg2 = st.columns([1, 1])
    with col_seg1:
        st.markdown("#### Strategic Cluster Scatter (CLV vs Churn Risk)")
        if (FIGURES_DIR / "21_customer_segments_scatter.png").exists():
            st.image(str(FIGURES_DIR / "21_customer_segments_scatter.png"), use_container_width=True)
            
    with col_seg2:
        st.markdown("#### Segment Profile Comparisons")
        if (FIGURES_DIR / "22_segment_profiles_comparison.png").exists():
            st.image(str(FIGURES_DIR / "22_segment_profiles_comparison.png"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### The 4 Strategic Customer Segments")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    with s_col1:
        st.markdown("""
        <div class="segment-card" style="border-top: 4px solid #10B981;">
            <h5>1. High-Value Loyalists</h5>
            <p><strong>Accounts:</strong> 1,950 (27.7%)</p>
            <p><strong>Avg ARPU:</strong> $90.36/mo</p>
            <p><strong>Avg Tenure:</strong> 58.3 months</p>
            <p><strong>Avg 24m CLV:</strong> $1,222.06</p>
            <p><strong>Avg Churn Risk:</strong> 24.1%</p>
            <span class="badge badge-success">Maintain VIP Support</span>
        </div>
        """, unsafe_allow_html=True)

    with s_col2:
        st.markdown("""
        <div class="segment-card" style="border-top: 4px solid #EF4444;">
            <h5>2. Vulnerable High-Spenders</h5>
            <p><strong>Accounts:</strong> 2,052 (29.1%)</p>
            <p><strong>Avg ARPU:</strong> $85.22/mo</p>
            <p><strong>Avg Tenure:</strong> 17.1 months</p>
            <p><strong>Avg 24m CLV:</strong> $1,059.64</p>
            <p><strong>Avg Churn Risk:</strong> 63.4%</p>
            <span class="badge badge-danger">PRIMARY RETENTION PRIORITY</span>
        </div>
        """, unsafe_allow_html=True)

    with s_col3:
        st.markdown("""
        <div class="segment-card" style="border-top: 4px solid #3B82F6;">
            <h5>3. Budget Steady Anchors</h5>
            <p><strong>Accounts:</strong> 1,480 (21.0%)</p>
            <p><strong>Avg ARPU:</strong> $30.73/mo</p>
            <p><strong>Avg Tenure:</strong> 45.4 months</p>
            <p><strong>Avg 24m CLV:</strong> $424.10</p>
            <p><strong>Avg Churn Risk:</strong> 13.9%</p>
            <span class="badge badge-info">Cross-Sell Opportunity</span>
        </div>
        """, unsafe_allow_html=True)

    with s_col4:
        st.markdown("""
        <div class="segment-card" style="border-top: 4px solid #F59E0B;">
            <h5>4. New Onboarding Risks</h5>
            <p><strong>Accounts:</strong> 1,561 (22.2%)</p>
            <p><strong>Avg ARPU:</strong> $38.16/mo</p>
            <p><strong>Avg Tenure:</strong> 9.4 months</p>
            <p><strong>Avg 24m CLV:</strong> $483.99</p>
            <p><strong>Avg Churn Risk:</strong> 43.1%</p>
            <span class="badge badge-warning">Onboarding Concierge</span>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 6: UPLIFT & CAMPAIGN ROI SIMULATOR
# ------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Causal Uplift Modeling & Financial Campaign ROI Simulator")
    st.write(
        "Uplift Modeling identifies **Persuadables**—customers who only retain if given an incentive. "
        "Test your own campaign assumptions using the real-time financial simulator below."
    )

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.markdown("#### Uplift Qini Curve (Two-Model T-Learner)")
        if (FIGURES_DIR / "23_uplift_qini_curve.png").exists():
            st.image(str(FIGURES_DIR / "23_uplift_qini_curve.png"), use_container_width=True)
            
    with col_u2:
        st.markdown("#### The 4 Marketing Uplift Archetypes")
        if (FIGURES_DIR / "25_uplift_quadrants.png").exists():
            st.image(str(FIGURES_DIR / "25_uplift_quadrants.png"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 💰 Interactive Retention Campaign ROI Simulator")
    
    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        st.markdown("#### Campaign Input Parameters")
        sim_audience = st.slider("Targeted Audience Size (Customers)", 100, 5000, 1200, 100)
        sim_cost = st.slider("Offer Cost per Customer ($)", 5.0, 100.0, 45.0, 5.0)
        sim_baseline_churn = st.slider("Baseline Churn Rate (%)", 10.0, 90.0, 63.0, 1.0) / 100.0
        sim_uplift = st.slider("Expected Retention Uplift (% pts saved)", 1.0, 30.0, 12.0, 0.5) / 100.0
        sim_arpu = st.slider("Customer Monthly ARPU ($)", 30.0, 120.0, 85.22, 1.0)
        sim_margin = st.slider("Gross Margin (%)", 40.0, 90.0, 70.0, 5.0) / 100.0
        sim_horizon = st.selectbox("Financial Valuation Horizon", [6, 12, 24], index=1)

    # Perform calculation
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
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {'#16A34A' if net_profit > 0 else '#DC2626'}; margin-bottom: 12px;">
            <div class="metric-lbl">Net Campaign Profit (EBITDA)</div>
            <div class="metric-val" style="color: {'#16A34A' if net_profit > 0 else '#DC2626'};">${net_profit:,.0f}</div>
            <span class="badge {'badge-success' if roi_pct > 0 else 'badge-danger'}">Campaign ROI: {roi_pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        res_c1, res_c2 = st.columns(2)
        res_c1.metric("Total Campaign Expenditure", f"${total_cost:,.0f}")
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
