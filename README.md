# Customer Lifetime Value (CLV) & Churn Intelligence System

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/database-Supabase%20Postgres-3ECF8E.svg)](https://supabase.com)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/pytest-44%20passed-brightgreen.svg)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, local-first Machine Learning and Decision Intelligence platform engineered to predict customer churn, explain individual churn drivers, model survival timelines under right-censoring, calculate forward Customer Lifetime Value (CLV), identify causal retention uplifts, and simulate campaign financial ROI.

---

## 1. System Architecture & End-to-End Workflow

```
                               +-------------------------------------+
                               |   Raw Data (IBM Telco & Hillstrom)  |
                               +-------------------------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |   Phase 2: Cleaning & Validation    |
                               |  - Imputed 11 whitespace charges    |
                               |  - Prevented survivorship bias      |
                               +-------------------------------------+
                                                  |
                                                  v
                      +-------------------------------------------------------+
                      |       Phase 3: Supabase Cloud PostgreSQL Store        |
                      |  - SQLAlchemy 2.0 ORM + Idempotent Upsert Pipeline    |
                      +-------------------------------------------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |   Phase 4: Feature Engineering      |
                               |  - Stratified 70/15/15 Data Split   |
                               |  - 7 Domain Features (41 Post-OHE)  |
                               +-------------------------------------+
                                                  |
                     +----------------------------+----------------------------+
                     |                                                         |
                     v                                                         v
    +---------------------------------+                       +---------------------------------+
    |   Phase 5: LightGBM Classifier  |                       |   Phase 7: Survival Analysis    |
    |  - ROC-AUC: 0.842 | PR-AUC: 0.655|                       |  - Kaplan-Meier Hazard Curves   |
    |  - Cost-Optimal Cutoff: 0.28    |                       |  - Cox PH Model (C-Index: 0.925)|
    |  - Recall: 92.86% (Caught 260)  |                       +---------------------------------+
    +---------------------------------+                                        |
                     |                                                         |
                     v                                                         v
    +---------------------------------+                       +---------------------------------+
    |   Phase 6: TreeSHAP Explain     |                       |   Phase 8: CLV & Segmentation   |
    |  - Additive Attributions        |                       |  - 24m Discounted Cash Flow     |
    |  - Global & Local Waterfall     |                       |  - K-Means 4 Strategic Clusters |
    +---------------------------------+                       +---------------------------------+
                     |                                                         |
                     +----------------------------+----------------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |   Phase 9: Causal Uplift Modeling   |
                               |  - Two-Model T-Learner (LightGBM)   |
                               |  - Qini Score: +37.46 | 4 Archetypes|
                               +-------------------------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |   Phase 10: Retention ROI Simulator |
                               |  - Precision Policy: +81.3% ROI     |
                               |  - vs Mass Blanket: -38.1% Loss     |
                               +-------------------------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |   Phase 11: Streamlit Dashboard     |
                               |  - 6 Multi-Tab Executive Interface  |
                               +-------------------------------------+
```

---

## 2. All 12 Milestones Summary

| Phase | Milestone | Status | Key Deliverable / Measured Result |
| :--- | :--- | :---: | :--- |
| **Phase 1** | **Environment & Scaffolding** | **Complete** | Windows `.venv` (Python 3.13), Git repository, modular folders, and `.gitignore`. |
| **Phase 2** | **Data Cleaning & EDA** | **Complete** | 11 whitespace `TotalCharges` imputed with 0.0; 6 publication figures generated in `reports/figures/`. |
| **Phase 3** | **Supabase PostgreSQL** | **Complete** | Ingested 7,043 customers into live Supabase pooler via SQLAlchemy 2.0 idempotent upsert. |
| **Phase 4** | **Feature Engineering** | **Complete** | Anti-leakage 70/15/15 stratified split; 7 domain features (41 encoded); `preprocessor.joblib`. |
| **Phase 5** | **LightGBM Churn Model** | **Complete** | ROC-AUC: `0.842`, PR-AUC: `0.655`; optimal threshold `0.28` catches **92.86% of churners**. |
| **Phase 6** | **Explainability (SHAP)** | **Complete** | TreeSHAP exact additivity; beeswarm plots; single-customer waterfall attribution. |
| **Phase 7** | **Survival Analysis** | **Complete** | Kaplan-Meier by Contract ($p < 0.0001$); multivariable Cox PH **C-index: 0.9247**. |
| **Phase 8** | **CLV & Segmentation** | **Complete** | DCF CLV ($843.94 mean); K-Means ($K=4$, Silhouette 0.389) isolated *Vulnerable High-Spenders*. |
| **Phase 9** | **Uplift Modeling** | **Complete** | Two-Model T-Learner; Qini Score `+37.46`; classified *Persuadables* (61.9%) vs *Sleeping Dogs* (5.9%). |
| **Phase 10** | **Retention ROI Simulator**| **Complete** | Benchmarked policies: Precision Targeting achieves **+81.3% ROI** vs Mass Blanket **-38.1% loss**. |
| **Phase 11** | **Streamlit Web Dashboard**| **Complete** | Interactive executive app with live threshold sliders, SHAP inspector, and P&L simulator. |
| **Phase 12** | **Testing & Presentation** | **Complete** | **44 automated unit & integration tests passing**; 10-minute presentation guide in `docs/`. |

---

## 3. Key Findings & Business Insights

1. **The Accuracy Paradox & Threshold Tuning**:
   - Predicting "No Churn" yields 73.5% baseline accuracy while catching **0% of churners**.
   - Moving from standard `0.50` threshold to cost-optimal **`0.28`** increases churn recall from **71.79% to 92.86%** on the test set, capturing **260 out of 280 churners** and minimizing combined false positive/negative financial penalties.
2. **Contract Lock-in vs Payment Friction**:
   - Month-to-Month contracts exhibit a **42.7% churn rate**, compared to **2.8%** for Two-Year contracts (Cox Hazard Ratio: `0.091`, -91% risk).
   - Electronic Check users experience a **45.3% churn rate**, compared to < 16% for automated credit cards (Cox Hazard Ratio: `1.56`, +56% risk).
3. **The 4 Strategic Customer Segments**:
   - **High-Value Loyalists (1,950 accounts)**: $90.36 ARPU, 58m tenure, $1,222 CLV, 24% churn risk.
   - **Vulnerable High-Spenders (2,052 accounts)**: $85.22 ARPU, 17m tenure, $1,060 CLV, **63.4% churn risk** (**Primary Retention Priority!**).
   - **Budget Steady Anchors (1,480 accounts)**: $30.73 ARPU, 45m tenure, $424 CLV, 14% churn risk.
   - **New Onboarding Risks (1,561 accounts)**: $38.16 ARPU, 9m tenure, $484 CLV, 43% churn risk.
4. **Causal Uplift & The ROI Breakthrough**:
   - Mass blanket discounting destroys **-$67,028** (-38.1% ROI) by giving unneeded discounts to "Sure Things" and subsidizing "Lost Causes".
   - Precision targeting (Vulnerable High-Spenders who are Persuadables) yields **+$43,932 in net profit (+81.3% ROI)**.

---

## 4. Quickstart & Installation

### Prerequisites
- Python 3.13+ on Windows / macOS / Linux
- Git

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/nexvia/churn-intelligence.git
cd churn-intelligence

# Create virtual environment
python -m venv .venv

# Activate on Windows
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Database Credentials (Optional for Local Re-runs)
Copy `.env.example` to `.env` and configure your Supabase connection string:
```bash
DATABASE_URL=postgresql+psycopg://postgres.YOUR_PROJECT_REF:YOUR_PASSWORD@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
```

### 3. Run Automated Test Suite
Verify that all 44 unit and integration tests are passing:
```bash
python -m pytest tests/ -v
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Directory Structure

```text
churn-intelligence/
├── app.py                      # Interactive Streamlit Executive Dashboard (Phase 11)
├── requirements.txt            # Project dependencies
├── README.md                   # System documentation & quickstart
├── data/
│   ├── raw/                    # Original IBM Telco & Hillstrom RCT CSVs
│   └── processed/              # Cleaned, split, and segmented CSV datasets
├── docs/                       # Technical milestone documentation
│   ├── database_setup.md       # Phase 3 Supabase integration
│   ├── eda_findings.md         # Phase 2 Exploratory data analysis
│   ├── feature_engineering.md  # Phase 4 Feature engineering & splits
│   ├── model_evaluation.md     # Phase 5 LightGBM & threshold optimization
│   ├── shap_explanations.md    # Phase 6 TreeSHAP interpretability
│   ├── survival_analysis.md    # Phase 7 Kaplan-Meier & Cox Proportional Hazards
│   ├── customer_lifetime_value.md # Phase 8 DCF CLV & K-Means segmentation
│   ├── uplift_modeling.md      # Phase 9 Two-Model T-Learner & Qini analysis
│   ├── retention_roi_simulator.md # Phase 10 Campaign financial simulation
│   └── presentation_guide.md   # Phase 12 10-Minute Interview Presentation Guide
├── models/                     # Serialized pipelines, joblibs, and metric summaries
├── reports/figures/            # 26 publication-grade figures (01 to 26)
├── src/
│   ├── data/                   # Data cleaning and EDA visualizers
│   ├── database/               # SQLAlchemy 2.0 schema and idempotent ingestion
│   ├── features/               # Anti-leakage preprocessing pipelines
│   ├── models/                 # Model training, explainability, survival, and CLV
│   └── retention/              # Uplift T-Learner and ROI simulator
└── tests/                      # 44 comprehensive unit and integration tests
```

---

## 6. Portfolio & Evaluation Guide

For a structured walkthrough to present this system during **internship evaluations, technical interviews, and stakeholder demos**, review:
👉 [`docs/presentation_guide.md`](docs/presentation_guide.md)
