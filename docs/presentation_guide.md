# 10-Minute Portfolio Presentation & Technical Interview Guide
## Customer Lifetime Value (CLV) & Churn Intelligence Engine

This document provides a structured, time-stamped script and technical defense strategy to present this project during **internship evaluations, technical machine learning interviews, and executive stakeholder demonstrations**.

---

## 1. Executive Summary Elevator Pitch (30 Seconds)

> *"In subscription telecommunications, acquiring a new customer costs 5 to 7 times more than retaining an existing one. Most companies build simple churn models that predict who is likely to leave, and then blindly distribute discounts.*  
>  
> *I built an end-to-end Decision Intelligence system that goes far beyond simple classification. It ingests customer data into a live Supabase PostgreSQL cloud database, trains a cost-optimized LightGBM classifier that catches 93% of churners, explains predictions using TreeSHAP, models customer survival timelines under right-censoring, values accounts via Discounted Cash Flow CLV, and applies Causal Uplift Modeling to identify exactly who can be persuaded to stay—demonstrating an 81.3% retention campaign ROI versus a 38.1% loss for traditional blanket discounting."*

---

## 2. 10-Minute Presentation Script & Flow

### Minute 0:00 – 1:30 | The Business Context & The Naive ML Trap
- **The Hook**: Show the business consequence of churn. Telecom churn rates average 25-30% annually. Losing a $100/mo customer destroys thousands of dollars in lifetime value.
- **The Naive ML Trap**:
  - Explain the **Accuracy Paradox**: A naive model that predicts "No Churn" achieves 73.5% accuracy but catches **0% of churners**, bankrupting the business.
  - Explain the **0.50 Threshold Failure**: Standard machine learning packages use a 0.50 probability cutoff. But in business, losing a customer ($120 cost) is 6x worse than sending an unneeded email or call ($20 cost).
- **Core Principle**: Data science must be evaluated by **financial impact and EBITDA**, not just F1-score.

---

### Minute 1:30 – 3:30 | Data Engineering, Cloud Database & Feature Pipelines
- **Cloud Architecture**:
  - Data ingested into a live **Supabase Cloud PostgreSQL** instance (`aws-0-ap-south-1.pooler.supabase.com:6543/postgres`).
  - Implemented an idempotent upsert pipeline (`ON CONFLICT ("customerID") DO UPDATE`) using SQLAlchemy 2.0 and `psycopg3`.
  - Addressed SQL column quoting for camelCase columns (`"MonthlyCharges" >= 0`) to prevent Postgres lowercase folding.
- **Handling Data Quality**:
  - Identified 11 whitespace `TotalCharges` records.
  - Highlighted that all 11 records had `tenure = 0` (brand-new accounts who haven't received their first bill).
  - *Technical Defense*: Imputed `0.0` instead of dropping or using the mean. Dropping them introduces survivorship bias; using the mean distorts zero-tenure reality.
- **Anti-Leakage Feature Engineering**:
  - Split data into **70% Train, 15% Validation, 15% Test** using stratified sampling *before* any preprocessing.
  - Fitted all encoders and scalers strictly on the training set.
  - Engineered 7 domain-specific features (e.g., `has_fiber_no_support`, `charge_per_tenure_month`, `monthly_to_total_ratio`), expanding from 20 raw features to 41 clean model features.

---

### Minute 3:30 – 5:00 | Churn Modeling, Anti-Leakage Threshold Optimization & TreeSHAP
- **Model Selection**:
  - Evaluated Dummy Classifier (73.5% accuracy, 0% recall), Balanced Logistic Regression (0.845 ROC-AUC), and **LightGBM** with `scale_pos_weight = 2.77`.
  - LightGBM achieved **0.843 ROC-AUC** and **0.635 PR-AUC** on the held-out test set.
- **Anti-Leakage Threshold Optimization**:
  - Simulated business cost curve exclusively on the **validation partition**: $\text{Total Cost} = 20 \cdot FP + 120 \cdot FN$.
  - The profit-maximizing threshold selected on validation is **`0.27`** (not a naive 0.50 cutoff!).
  - Evaluating this frozen `0.27` threshold on the **held-out test set**, test recall reached **93.57%**, catching **262 out of 280 churners** and creating **+$700.00** in net financial benefit on the test slice alone.
- **Model Explainability (TreeSHAP)**:
  - Verified exact additivity: $\text{Expected Margin} + \sum \text{SHAP} = \text{Model Output}$.
  - Top churn drivers: Month-to-Month contracts (+0.59 SHAP), Fiber Optic (+0.16 SHAP), High Monthly-to-Total charge ratio (+0.15 SHAP).
  - Top retention drivers: Two-Year contracts (-0.62 SHAP), One-Year contracts (-0.29 SHAP).
  - *Technical Rigor*: Explicitly note that SHAP measures predictive feature association, not causal mechanisms.

---

### Minute 5:00 – 7:00 | Survival Analysis (Lifelines) & Forward CLV
- **Addressing Right-Censoring**:
  - 5,174 active customers have not yet churned. Standard classifiers treat them as "negative" outcomes, ignoring that they will eventually churn in future months.
  - Fitted **Kaplan-Meier survival curves**:
    - Month-to-Month customers have a median survival half-life of **35 months**.
    - One-Year and Two-Year contract customers have median survival exceeding **72 months** (Log-rank test: $p < 0.0001$).
  - Fitted multivariable **Cox Proportional Hazards** (**C-index: 0.8432**):
    - *Methodological Defense*: Cumulative `TotalCharges` was explicitly excluded from covariates because $\text{TotalCharges} \approx \text{tenure} \times \text{MonthlyCharges}$, which creates artificial collinearity with the duration variable $T$.
    - Fiber Optic hazard multiplier: **2.76x** (+176% risk).
    - Electronic Check payment multiplier: **1.88x** (+88% risk).
    - Two-Year Contract multiplier: **0.064x** (-93.6% risk).
- **Forward Discounted Cash Flow (DCF) CLV**:
  - Separated revenue from gross profit using 70% gross margin.
  - Discounted future cash flows using corporate WACC (10% annual, 0.797% monthly).
  - Conditioned monthly survival probability on contract survival curves.
  - Mean 24-month forward CLV: **$843.94**; Total portfolio valuation: **$5.94M**.
- **K-Means Strategic Segmentation ($K=4$)**:
  - Isolated **Segment 2: Vulnerable High-Spenders** (2,052 accounts, $85.22/mo, 17m tenure, $1,060 CLV, **63.4% churn risk**).
  - *Strategic Recommendation*: This segment generates massive cash flow but is leaking out of the company—our primary retention priority!

---

### Minute 7:00 – 8:45 | Causal Uplift Modeling & The ROI Simulator
- **The Causal Dilemma**:
  - "Predicting churn is not the same as deciding who to target."
  - Blind discounts give money to *Sure Things* (organic renewals) and waste budget on *Lost Causes* (irreversible churners), while agitating *Sleeping Dogs*.
- **The Solution (Two-Model T-Learner)**:
  - Trained on the Kevin Hillstrom MineThatData Randomized Controlled Trial benchmark (64,000 customers).
  - Predicted Individual Treatment Effect: $\hat{\tau}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$.
  - Achieved a **Qini Score of +37.46**; the top 20% of customers ranked by predicted uplift generated **+8.5% incremental conversions**.
  - Categorized customers into the 4 archetypes: **Persuadables (61.9%)**, **Lost Causes (26.6%)**, **Sleeping Dogs (5.9%)**, **Sure Things (5.6%)**.
- **Campaign Financial Simulator**:
  - **Policy 1: Mass Blanket (All 7,043 customers)**: Cost $176k, Net Profit **-$67,028 (-38.1% ROI)**.
  - **Policy 2: Risk-Only (All Prob $\ge$ 0.28)**: Cost $80.5k, Net Profit **+$1,480 (+1.8% ROI)**.
  - **Policy 3: Precision (Vulnerable High-Spenders + Persuadables)**: Cost $54k, Net Profit **+$43,932 (+81.3% ROI)**.

---

### Minute 8:45 – 10:00 | Live Dashboard Walkthrough & Conclusion
- Open Streamlit Dashboard (`streamlit run app.py`).
- Show the 6 tabs:
  1. Executive KPIs ($5.94M portfolio CLV).
  2. Live Threshold Slider (watch confusion matrix and financial cost recalculate in real time).
  3. Individual Customer SHAP Inspector.
  4. Kaplan-Meier survival curves.
  5. 4 K-Means Segment cards.
  6. Financial ROI Simulator with interactive cost and uplift sliders.
- Conclude: *"This project connects data engineering, predictive machine learning, explainable AI, survival statistics, and causal inference directly to corporate EBITDA."*

---

## 3. Tough Interview Questions & Bulletproof Answers

### Q1: "Why did you use LightGBM instead of Deep Learning or XGBoost?"
> **Answer**:  
> *"For tabular data with mixed continuous and categorical features, gradient boosted decision trees consistently match or outperform deep neural networks without requiring extensive hyperparameter tuning or large compute overhead. I chose LightGBM over XGBoost because LightGBM uses leaf-wise tree growth with Histogram-based binning, which is significantly faster and more memory-efficient on Windows, while providing native support for categorical features and built-in `scale_pos_weight` for class imbalance."*

---

### Q2: "Why couldn't you train your Uplift Model directly on the IBM Telco dataset?"
> **Answer**:  
> *"Because the IBM Telco dataset is cross-sectional observational data without a randomized intervention indicator. In observational data, if discounts are given only to customers who complain or are already churning, the treatment assignment is confounded with customer intent. Training an uplift model on confounded data produces biased causal estimates. To ensure scientific rigor, I trained the T-Learner on the Kevin Hillstrom randomized A/B test benchmark, established the Qini curves and quadrant segmentation, and then combined those causal targeting rules with the Telco CLV segments."*

---

### Q3: "What is right-censoring in survival analysis and why does standard ML ignore it?"
> **Answer**:  
> *"Right-censoring occurs when an observation ends before the event of interest occurs. In Telco, 5,174 customers are still active at the time of observation. A standard binary classification model treats them as 'Did Not Churn' (label 0). However, many of those active customers will churn 1 or 2 months later. Standard ML introduces survivorship bias by treating recent sign-ups and 5-year loyalists as identical 'non-churners'. Survival analysis explicitly accounts for censoring by estimating the conditional probability of surviving past time $t$ given that the customer has survived up to time $t$."*

---

### Q4: "How does your CLV formula differ from simple historical CLV?"
> **Answer**:  
> *"Historical CLV simply sums past charges, which is backward-looking and penalizes high-value new customers who joined recently. My implementation uses a forward-looking Discounted Cash Flow (DCF) model over a 24-month horizon. It isolates gross profit margin (70%) from revenue, discounts future cash flows at the corporate Weighted Average Cost of Capital (10% annual), and conditions each future month's cash flow on the customer's contract-specific Kaplan-Meier survival probability. This prevents revenue inflation and accurately reflects the time value of money."*

---

### Q5: "How do you explain the trade-off of lowering your threshold from 0.50 to 0.27, and how did you prevent evaluation leakage?"
> **Answer**:  
> *"First, to prevent evaluation leakage, I tuned the decision threshold strictly on the validation set, never touching test data. The profit-maximizing threshold on validation was 0.27. Second, when evaluated on the held-out test set, lowering the threshold from 0.50 to 0.27 increased False Positives (from 165 to 381) and lowered precision (from 55.8% to 40.8%), but dramatically reduced False Negatives (from 72 to 18), boosting test recall from 74.3% to 93.6%. Because missing a churner costs $120 while sending an offer costs only $20, a False Negative is 6 times more expensive than a False Positive. Minimizing the business loss function strictly favors 0.27, generating a net financial benefit of +$700 on the test slice."*
