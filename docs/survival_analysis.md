# Survival Analysis & Time-to-Event Modeling Report

**Project**: Customer Lifetime Value and Churn Intelligence System  
**Pipeline Source**: [`src/models/survival_analysis.py`](file:///f:/yash/projects/nexvia/churn-intelligence/src/models/survival_analysis.py)  
**Methodology**: Kaplan-Meier Estimator, Log-Rank Hypothesis Testing, Cox Proportional Hazards  
**Artifacts**: `models/survival_models.joblib`, `models/survival_metrics.json`  
**Figures**: Stored in `reports/figures/` (`15_km_overall_survival.png`, `16_km_by_contract.png`, `17_km_by_services.png`, `18_cox_hazard_ratios.png`)

---

## 1. Core Concepts & Data Science Foundations

In standard classification (Phase 5), we predicted a binary question: *"Will this customer churn in the current window?"*  
**Survival Analysis** models the dynamic temporal question: *"When is this customer expected to churn, and what is their expected remaining relationship duration?"*

### 1.1 Time-to-Event Terminology
- **Duration ($T$)**: The duration in months a customer has maintained an account (`tenure`).
- **Event ($E$)**: Binary indicator `churn_label` ($E=1$ if account churned, $E=0$ if active).
- **Survival Function ($S(t)$)**: The probability that a customer remains active longer than month $t$:
  $$S(t) = P(T > t)$$
- **Hazard Rate ($h(t)$)**: The instantaneous rate of churn at month $t$, conditional on having survived up to month $t$.

### 1.2 Right-Censoring: Why Standard Linear Regression Fails
If you run standard linear regression to predict `tenure`, you commit a fatal error: **ignoring censoring**.
- 5,174 customers (73.5% of the dataset) have `Churn = 'No'`. Their `tenure` is not their final lifetime; it is only their observed duration up to the snapshot date.
- Their eventual churn will happen in the future ($T > \text{tenure}_{\text{current}}$). This phenomenon is called **Right-Censoring**.
- Survival analysis models correctly account for censored observations without biasing duration estimates downward.

---

## 2. Crucial Methodological Limitation of the Telco Dataset

In technical interviews and senior architecture reviews, highlighting data limitations demonstrates mature engineering judgment:

> [!WARNING]
> **Cross-Sectional Snapshot Limitation**:
> 1. The IBM Telco dataset is an observational cross-sectional snapshot, not a longitudinal tracking panel. We do not have calendar timestamps (e.g. signup date vs cancellation date).
> 2. Customers who joined 5 years ago entered under different market pricing than customers who joined 2 months ago (cohort vintage effects).
> 3. Features like `Contract` or `MonthlyCharges` are recorded as of the snapshot date. We cannot observe whether a churned customer switched from a 1-year contract to month-to-month right before canceling (time-varying covariates).
> 4. **Best Practice Applied**: We treat `tenure` strictly as duration-at-observation, use non-parametric Kaplan-Meier for cohort survival curves, and apply regularized Cox Proportional Hazards for relative risk comparisons rather than claiming deterministic calendar survival dates.

---

## 3. Kaplan-Meier Empirical Survival Benchmarks

### 3.1 Overall Customer Base (`15_km_overall_survival.png`)
- **12-Month Survival Probability**: **84.32%**
- **24-Month Survival Probability**: **78.87%**
- **36-Month Survival Probability**: **74.85%**
- *Observation*: The steepest survival drop occurs in months 1–12 (the onboarding friction window). Once past month 24, the survival curve flattens significantly.

### 3.2 Survival Stratified by Contract Type (`16_km_by_contract.png`)

| Contract Agreement | Customer Count | 12-Month Survival | 24-Month Survival | Median Lifetime |
| :--- | :---: | :---: | :---: | :---: |
| **Month-to-month** | 3,875 | **70.28%** | **58.60%** | **35.0 months** |
| **One year** | 1,473 | **99.12%** | **97.83%** | **> 72 months** |
| **Two year** | 1,695 | **100.00%** | **100.00%** | **> 72 months** |

- **Log-Rank Test**: Comparing Month-to-Month vs One-Year survival yields a test statistic of **$1,172.5$** ($p < 0.0001$), confirming that contract structure creates a massive, statistically significant survival divergence.

### 3.3 Survival by Tech Support & Internet (`17_km_by_services.png`)
- Customers with **Tech Support** achieve **93.2% survival at 24 months**, compared to **68.4% for customers without support**.

---

## 4. Cox Proportional Hazards Model (`18_cox_hazard_ratios.png`)

The multivariable Cox model estimates the relative risk multiplier (Hazard Ratio $HR = e^\beta$) for each customer attribute, controlling for other variables simultaneously:

$$\text{Concordance Index (C-Index)} = \mathbf{0.9247}$$
*(A C-index of 0.925 indicates outstanding discriminative ability across customer survival pairs).*

### Top Hazard Ratios ($HR = e^\beta$)

| Covariate | Hazard Ratio ($HR$) | 95% Confidence Interval | $p$-value | Risk Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **`InternetService_Fiber optic`** | **2.263** | [1.95, 2.63] | $< 0.0001$ | **$2.26\times$ hazard multiplier**. Fiber users churn at more than double the rate of DSL users, ceteris paribus. |
| **`PaymentMethod_Mailed check`** | **1.603** | [1.32, 1.95] | $< 0.0001$ | **$60\%$ elevated hazard** compared to automated bank transfers. |
| **`PaymentMethod_Electronic check`** | **1.564** | [1.35, 1.81] | $< 0.0001$ | **$56\%$ elevated hazard** compared to automated bank transfers. |
| **`PaperlessBilling_Yes`** | **1.231** | [1.11, 1.37] | $0.0001$ | **$23\%$ elevated hazard** compared to paper billing. |
| **`MonthlyCharges`** | **1.018** | [1.01, 1.02] | $< 0.0001$ | Each $+\$10/\text{mo}$ increases churn hazard by **$\approx 18\%$**. |
| **`Contract_Two year`** | **0.091** | [0.06, 0.14] | $< 0.0001$ | **$91\%$ hazard reduction** ($HR < 0.10$). Extreme protective retention anchor. |

---

## 5. Architectural Bridge to Phase 8 (Customer Lifetime Value)

The survival functions estimated here provide the empirical foundation for **Customer Lifetime Value (CLV)** in Phase 8:
- Rather than assuming a constant, naive retention rate for all customers, CLV will integrate the **expected survival curve $S(t)$** specific to each customer's contract, tenure, and service tier over a discounted future cash flow horizon.
