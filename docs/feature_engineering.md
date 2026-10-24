# Feature Engineering & Preprocessing Architecture

**Project**: Customer Lifetime Value and Churn Intelligence System  
**Pipeline Source**: [`src/features/build_features.py`](file:///f:/yash/projects/nexvia/churn-intelligence/src/features/build_features.py)  
**Serialized Preprocessor**: `models/preprocessor.joblib`  
**Data Matrices**: `data/processed/X_train.csv`, `X_val.csv`, `X_test.csv`

---

## 1. Overview & Anti-Leakage Protocol

Feature engineering transforms raw customer records into predictive, numeric signals that gradient boosting trees (LightGBM) can exploit effectively.

### The Leakage Prevention Mandate
To ensure strict adherence to industry best practices and interview standards:
1. **Split-First Architecture**: We split the full dataset into:
   - **Training Set (70%)**: 4,929 customers
   - **Validation Set (15%)**: 1,057 customers
   - **Test Set (15%)**: 1,057 customers
   using stratified sampling on `churn_label` (ensuring each split contains exactly ~26.5% churned accounts).
2. **Fit-Only-On-Train**: The `StandardScaler` and `OneHotEncoder` are fit **only** on `train_df`.
3. **Identity Exclusion**: `customerID`, `Churn` (text label), and `churn_label` (target integer) are excluded from the feature matrix `X` to eliminate identity leakage and target leakage.

---

## 2. Engineered Domain Features

| Feature | Formula / Logic | Business & Empirical Rationale |
| :--- | :--- | :--- |
| **`tenure_cohort`** | Bins: `[0-12m, 13-24m, 25-36m, 37-48m, 49-60m, 61-72m]` | Captures non-linear hazard decay. Customers in months 0–12 experience 47.4% churn vs 6.6% in months 61–72. |
| **`num_services_subscribed`** | Sum of active phone, internet, security, and streaming add-ons (0 to 9) | Measures total service adoption. Higher service bundling increases switching friction. |
| **`num_security_services`** | Sum of `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` (0 to 4) | Security services correlate with extreme customer stickiness (15.2% churn with Tech Support vs 41.6% without). |
| **`is_automatic_payment`** | 1 if payment method contains `"automatic"`, else 0 | Distinguishes hands-off autopay (15.2%–16.7% churn) from high-friction manual check billing (45.3% churn). |
| **`has_fiber_no_support`** | `(InternetService == 'Fiber optic') & (TechSupport != 'Yes')` | Pinpoints the single highest-risk customer segment identified in EDA (expensive fiber bill without technical assistance). |
| **`charge_per_tenure_month`** | `TotalCharges / (tenure + 1)` | Represents historical monthly burn rate without division-by-zero errors for new customers (`tenure=0`). |
| **`monthly_to_total_ratio`** | `MonthlyCharges / (TotalCharges + 1)` | Normalizes current commitment against historical spend. High values flag new accounts at peak churn vulnerability. |

---

## 3. Encoding & Transformation Specifications

The preprocessing pipeline (`ColumnTransformer`) standardizes 9 continuous features and One-Hot Encodes 17 categorical features, producing **41 total modeling features**:

1. **StandardScaler**:
   - `tenure`, `MonthlyCharges`, `TotalCharges`, `num_services_subscribed`, `num_security_services`, `charge_per_tenure_month`, `monthly_to_total_ratio`, etc.
2. **OneHotEncoder (`drop='first'`)**:
   - Dropping the first dummy column prevents multicollinearity (the dummy variable trap).
   - Unknown categories encountered during future inference are safely handled (`handle_unknown='ignore'`).

---

## 4. Verification Benchmarks

| Metric | Training Set | Validation Set | Test Set |
| :--- | :---: | :---: | :---: |
| **Customer Count** | 4,929 | 1,057 | 1,057 |
| **Feature Dimensions** | 41 | 41 | 41 |
| **Churn Rate (%)** | 26.54% | 26.58% | 26.49% |
| **Missing Values (NaN)** | 0 | 0 | 0 |
| **Customer ID Overlap** | **0%** | **0%** | **0%** |
