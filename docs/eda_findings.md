# Exploratory Data Analysis & Data Quality Report

**Project**: Customer Lifetime Value and Churn Intelligence System  
**Dataset**: IBM Telco Customer Churn (`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`)  
**Processed Output**: `data/processed/telco_churn_cleaned.csv`  
**Figures**: Stored in `reports/figures/`

---

## 1. Executive Summary & Key Metrics

The dataset contains account records for **7,043 customers** across 21 features capturing customer demographics, subscribed services, account tenure, payment arrangements, and billing charges.

| Metric | Measured Value | Business Interpretation |
| :--- | :---: | :--- |
| **Total Customers** | 7,043 | Full dataset with 100% unique `customerID`s |
| **Retained Customers (`Churn=No`)** | 5,174 (73.46%) | Baseline retention rate |
| **Churned Customers (`Churn=Yes`)** | 1,869 (26.54%) | Churn rate benchmark |
| **Class Imbalance Ratio** | ~2.77 : 1 | Moderate imbalance; PR-AUC and cost-based thresholds are essential |
| **Median Tenure (Churned)** | **10.0 months** | Churn hazard is concentrated heavily in the first year |
| **Median Tenure (Retained)** | **38.0 months** | Established customers exhibit strong retention inertia |
| **Mean Monthly Charge (Churned)** | **$74.44 / mo** | At-risk customers pay significantly higher monthly rates |
| **Mean Monthly Charge (Retained)** | **$61.27 / mo** | Retained customers have lower monthly billing averages |

---

## 2. Data Quality Analysis & Cleaning Decisions

### 2.1 Whitespace in `TotalCharges`
- **Anomaly**: `TotalCharges` was initially imported as text/object. Inspection revealed **11 customer records** where `TotalCharges` contained whitespace spaces (`" "`).
- **Investigation**:
  - All 11 records have `tenure == 0`.
  - All 11 records have `Churn == "No"`.
  - All 11 records have standard positive `MonthlyCharges` (ranging from $19.70 to $80.85).
- **Business Rationale**:
  - These are brand-new customers who enrolled during the current billing cycle and have not yet received or completed their first monthly billing cycle.
  - **Why dropping them is wrong**: Deleting these records would introduce *survivorship bias* and discard valid, currently active customers.
  - **Why mean/median imputation is wrong**: Imputing the dataset average (~$2,283) would falsely misrepresent new customers as having paid thousands of dollars.
  - **Defensible Solution**: Impute `TotalCharges = 0.0`. This accurately reflects zero historical dollars billed.

### 2.2 Uniqueness and Null Audits
- Exactly 7,043 unique `customerID` values out of 7,043 rows (0 duplicates).
- Following numeric conversion of `TotalCharges` and zero-imputation, the cleaned dataset contains **0 missing values** across all 22 columns.

---

## 3. Exploratory Data Findings by Business Domain

### 3.1 Contract Duration (Figure: `02_churn_by_contract.png`)
- **Month-to-month**: **42.71% churn** (N = 3,875)
- **One year**: **11.27% churn** (N = 1,473)
- **Two year**: **2.83% churn** (N = 1,695)
- **Insight**: More than half of the customer base is on a month-to-month plan, and their churn rate is more than 15× higher than customers on two-year agreements.

### 3.2 Tenure Cohorts & Hazard Timing (Figure: `03_churn_by_tenure_cohorts.png`)
- **0–12 months**: **47.44% churn**
- **13–24 months**: **28.71% churn**
- **25–36 months**: **21.32% churn**
- **37–48 months**: **14.53% churn**
- **49–60 months**: **10.05% churn**
- **61–72 months**: **6.61% churn**
- **Insight**: The highest attrition hazard occurs in the first 12 months (onboarding phase). Survival curves (Phase 7) will model this decay explicitly.

### 3.3 Internet Services & Technical Support (Figure: `04_churn_by_internet_and_tech_support.png`)
- **Fiber Optic Customers**: **41.89% churn** (N = 3,096)
- **DSL Customers**: **18.96% churn** (N = 2,421)
- **No Internet (Phone only)**: **7.40% churn** (N = 1,526)
- **Tech Support Subscription**:
  - Customers without Tech Support: **41.64% churn**
  - Customers with Tech Support: **15.17% churn**
- **Insight**: High-speed fiber customers experience high churn. However, value-added protective services (Tech Support, Online Security) correlate strongly with customer stickiness.

### 3.4 Payment Methods & Billing (Figure: `05_churn_by_payment_method.png`)
- **Electronic Check**: **45.29% churn** (N = 2,365)
- **Mailed Check**: **19.11% churn** (N = 1,612)
- **Bank Transfer (Automatic)**: **16.71% churn** (N = 1,544)
- **Credit Card (Automatic)**: **15.24% churn** (N = 1,522)
- **Insight**: Customers using manual electronic checks churn at nearly triple the rate of customers utilizing automated payment instruments.

### 3.5 Pricing & Billing Magnitude (Figure: `06_charges_distribution_by_churn.png`)
- Churned customers are heavily skewed toward higher monthly bills (peaking between $70–$100/mo).
- Retained customers form a bimodal distribution: a large low-cost cluster (~$20/mo phone-only users) and a stable premium cluster.

---

## 4. Critical Methodological Distinction: Correlation vs. Causation

In beginner data science projects, analysts often look at strong correlations and jump to naive causal recommendations. In an executive presentation or technical interview, maintaining rigorous causal hygiene is essential:

### 1. Contract Type: Correlation vs. Causation
- **Observed Correlation**: Month-to-month contracts strongly correlate with high churn (42.7%), whereas two-year contracts have low churn (2.8%).
- **Flawed Causal Conclusion**: *"If we force all month-to-month customers onto two-year contracts, we will reduce churn to 2.8%."*
- **True Explanation**: Month-to-month contracts allow dissatisfied or transient customers to cancel immediately with zero penalty. Conversely, satisfied, stable customers willingly enter two-year contracts to lock in lower rates. Forcing long-term contracts would likely kill acquisition rates rather than solve underlying customer dissatisfaction.

### 2. Fiber Optic Internet: Correlation vs. Causation
- **Observed Correlation**: Fiber optic subscribers churn at 41.9%, compared to 19.0% for DSL.
- **Flawed Causal Conclusion**: *"Fiber optic service causes customers to leave; we should discourage fiber signups."*
- **True Explanation**: Fiber optic service carries the highest monthly charge (averaging ~$91/mo vs ~$58/mo for DSL). High bills heighten price sensitivity. Furthermore, fiber is installed in competitive urban areas where competitor alternatives (cable, 5G home internet) are readily available.

### 3. Tech Support: Correlation vs. Causation
- **Observed Correlation**: Customers with Tech Support churn at 15.2% vs 41.6% without.
- **Flawed Causal Conclusion**: *"Giving free tech support to everyone will automatically slash churn by 26 percentage points."*
- **True Explanation**: Customers who opt to purchase tech support are typically more invested in their home network setup or less price-sensitive. In Phase 9 (Uplift Modeling), we will rigorously evaluate whether offering support or discounts creates genuine incremental uplift or merely subsidizes customers who would stay anyway.

---

## 5. Architectural Implications for Future Phases

1. **Feature Engineering (Phase 4)**:
   - Engineer `tenure_group` cohorts (0–12m, 13–24m, etc.).
   - Calculate `monthly_to_total_ratio` and service bundle density (sum of active security/support add-ons).
   - Ensure `customerID` is excluded from feature matrices to prevent identity leakage.
2. **Modeling & Thresholding (Phase 5)**:
   - Handle the ~26.5% class imbalance using weighted loss or tuning decision thresholds based on actual business retention economics rather than an arbitrary 0.5 probability cutoff.
3. **Survival Analysis (Phase 7)**:
   - Use Kaplan-Meier and Cox Proportional Hazards to model expected remaining lifetime as a function of contract and service tiers.
4. **Targeting & Uplift (Phase 9 & 10)**:
   - High churn probability alone does **not** equal responsiveness to retention incentives. We must isolate the "Persuadables" to maximize campaign ROI.
