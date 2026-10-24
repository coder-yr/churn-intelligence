# Customer Lifetime Value (CLV) & Churn Intelligence System

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/database-Supabase%20Postgres-3ECF8E.svg)](https://supabase.com)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/pytest-47%20passed-brightgreen.svg)](https://pytest.org)
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
    |  - ROC-AUC: 0.843 | PR-AUC: 0.635|                       |  - Kaplan-Meier Hazard Curves   |
    |  - Val-Tuned Cutoff: 0.27       |                       |  - Cox PH Model (C-Index: 0.843)|
    |  - Test Recall: 93.57% (262/280)|                       +---------------------------------+
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
| **Phase 2** | **Data Cleaning & EDA** | **Complete** | 11 whitespace `TotalCharges` imputed with 0.0 (zero tenure); 6 publication figures generated. |
| **Phase 3** | **Supabase PostgreSQL** | **Complete** | Ingested 7,043 customers into live Supabase pooler via SQLAlchemy 2.0 idempotent upsert with SSL. |
| **Phase 4** | **Feature Engineering** | **Complete** | Anti-leakage 70/15/15 stratified split; 7 domain features (41 encoded); `preprocessor.joblib`. |
| **Phase 5** | **LightGBM Churn Model** | **Complete** | Strict anti-leakage: optimal threshold `0.27` tuned on validation, yielding **93.57% Test Recall** (262 of 280 churners caught). |
| **Phase 6** | **Explainability (SHAP)** | **Complete** | TreeSHAP exact additivity; beeswarm plots; 3 customer archetype waterfall explanations. |
| **Phase 7** | **Survival Analysis** | **Complete** | Kaplan-Meier by Contract ($p < 0.0001$); non-collinear Cox PH **C-index: 0.8432** (excluding cumulative `TotalCharges`). |
| **Phase 8** | **CLV & Segmentation** | **Complete** | DCF CLV ($843.94 mean); K-Means ($K=4$, Silhouette 0.389) isolated *Vulnerable High-Spenders*. |
| **Phase 9** | **Uplift Modeling** | **Complete** | Two-Model T-Learner on Hillstrom RCT; Qini Score `+37.46`; classified *Persuadables* (61.9%) vs *Sleeping Dogs* (5.9%). |
| **Phase 10** | **Retention ROI Simulator**| **Complete** | Benchmarked policies: Precision Targeting achieves **+81.3% ROI** vs Mass Blanket **-38.1% loss**. |
| **Phase 11** | **Streamlit Web Dashboard**| **Complete** | Authoritative dynamic metrics, genuine held-out test data evaluation scope, and fixed customer selector. |
| **Phase 12** | **Testing & Presentation** | **Complete** | **47 automated unit & integration tests passing**; 10-minute presentation guide in `docs/`. |

---

## 3. Data Acquisition, Licensing & Placement

This platform uses two established, publicly available datasets:

1. **IBM Telco Customer Churn Dataset**:
   - **Source**: Kaggle / IBM Business Analytics Community.
   - **Licensing**: Public Domain / Open Database.
   - **Placement**: Save `WA_Fn-UseC_-Telco-Customer-Churn.csv` into `data/raw/`.
   - **Dimensions**: 7,043 records, 21 initial attributes.
2. **Kevin Hillstrom MineThatData E-Mail Analytics Dataset**:
   - **Source**: Kevin Hillstrom / MineThatData Marketing Analytics Blog.
   - **Licensing**: Public Open Research Benchmark for Causal Inference & Uplift Modeling.
   - **Placement**: Save `hillstrom_raw.csv` into `data/raw/`.
   - **Dimensions**: 64,000 customers randomly assigned to Email treatment arms or Control.

---

## 4. Quickstart & Reproducible Pipeline

### Prerequisites
- Python 3.13+ on Windows / macOS / Linux
- Git

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/coder-yr/churn-intelligence.git
cd churn-intelligence

# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Database Credentials (Optional for Local Re-runs)
Copy `.env.example` to `.env` and configure your Supabase connection string:
```bash
DATABASE_URL=postgresql+psycopg://postgres.YOUR_PROJECT_REF:YOUR_PASSWORD@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
```

### 3. Run the End-to-End Master Pipeline (One Command)
Rebuild all cleaned datasets, feature matrices, machine learning models, SHAP explanations, survival curves, CLV clusters, uplift models, and figures from scratch:
```bash
python src/run_pipeline.py
```

### 4. Run Automated Test Suite (47 Tests)
Verify that all unit and integration tests pass:
```bash
python -m pytest tests/ -v
```

### 5. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Critical Technical & Causal Assumptions

1. **Anti-Leakage Decision Threshold Tuning**:
   - In traditional naive implementations, thresholds are tuned on test data, introducing optimistic bias.
   - In our system, the profit-maximizing threshold (`0.27`) was tuned exclusively on the **validation partition**.
   - The frozen model and threshold were then evaluated on the **held-out test partition**, catching **262 out of 280 churners (93.57% Recall)** and achieving **+$700.00 net benefit**.
2. **Survival Analysis & Non-Collinear Duration Covariates**:
   - `TotalCharges` is mechanically cumulative over duration ($\text{TotalCharges} \approx \text{tenure} \times \text{MonthlyCharges}$). Including it in Cox Proportional Hazards artificially inflates concordance due to leakage of $T$.
   - Excluding `TotalCharges` eliminates multi-collinearity, yielding an authentic **Concordance Index of 0.8432**.
   - Cross-sectional Telco data is subject to left-truncation and survivor selection bias; static covariates reflect snapshot state rather than longitudinal transitions.
3. **Causal Uplift vs Predictive Association**:
   - SHAP values measure **predictive association**, not causal guarantees.
   - Because the observational Telco dataset lacks randomized retention treatments, the causal uplift model is trained on the **Hillstrom Randomized Controlled Trial (RCT)** benchmark to prove causal methodology before applying targeting rules to Telco CLV segments.
4. **Financial Simulation Parameters**:
   - Retention offer costs ($20-$45), save rate (50%), gross margin (70%), and annual discount rate (10%) are explicitly modeled business assumptions parameterized for sensitivity analysis.

---

## 6. Directory Structure

```text
churn-intelligence/
├── app.py                      # Interactive Streamlit Executive Dashboard (Phase 11)
├── requirements.txt            # Project dependencies
├── README.md                   # System documentation & quickstart
├── data/
│   ├── raw/                    # Raw IBM Telco & Hillstrom RCT CSVs
│   └── processed/              # Cleaned, split, and segmented CSV datasets
├── docs/                       # Technical milestone documentation
│   ├── database_setup.md       # Phase 3 Supabase integration
│   ├── eda_findings.md         # Phase 2 Exploratory data analysis
│   ├── feature_engineering.md  # Phase 4 Feature engineering & splits
│   ├── model_evaluation.md     # Phase 5 LightGBM & anti-leakage threshold tuning
│   ├── shap_explanations.md    # Phase 6 TreeSHAP interpretability
│   ├── survival_analysis.md    # Phase 7 Kaplan-Meier & Cox Proportional Hazards
│   ├── customer_lifetime_value.md # Phase 8 DCF CLV & K-Means segmentation
│   ├── uplift_modeling.md      # Phase 9 Two-Model T-Learner & Qini analysis
│   ├── retention_roi_simulator.md # Phase 10 Campaign financial simulation
│   └── presentation_guide.md   # Phase 12 10-Minute Interview Presentation Guide
├── models/                     # Serialized pipelines, joblibs, and metric summaries
├── reports/figures/            # 26 publication-grade figures (01 to 26)
├── src/
│   ├── run_pipeline.py         # Master pipeline orchestrator
│   ├── data/                   # Data cleaning and EDA visualizers
│   ├── database/               # SQLAlchemy 2.0 schema and idempotent ingestion
│   ├── features/               # Anti-leakage preprocessing pipelines
│   ├── models/                 # Model training, explainability, survival, and CLV
│   └── retention/              # Uplift T-Learner and ROI simulator
└── tests/                      # 47 comprehensive unit and integration tests
```

---

## 7. Portfolio & Presentation Guide

For a structured walkthrough to present this system during **internship evaluations, technical interviews, and stakeholder demos**, review:
👉 [`docs/presentation_guide.md`](docs/presentation_guide.md)
