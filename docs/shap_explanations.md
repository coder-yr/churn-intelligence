# Model Explainability & Interpretability Report (SHAP)

**Project**: Customer Lifetime Value and Churn Intelligence System  
**Pipeline Source**: [`src/models/explainability.py`](file:///f:/yash/projects/nexvia/churn-intelligence/src/models/explainability.py)  
**Method**: TreeSHAP via `shap.TreeExplainer` on the fitted LightGBM Classifier  
**Artifacts**: `models/shap_feature_importance.json`, `models/sample_customer_explanations.json`  
**Figures**: Stored in `reports/figures/` (`11_shap_summary_bar.png`, `12_shap_beeswarm.png`, `13_shap_dependence_tenure.png`, `14_shap_dependence_monthly_charges.png`)

---

## 1. What Are SHAP Values? (Beginner-Friendly Explanation)

Machine Learning models like Gradient Boosted Trees (LightGBM) can be "black boxes" because they combine hundreds of decision trees with thousands of split rules. 

**SHAP (SHapley Additive exPlanations)** is grounded in cooperative game theory (developed by Nobel Laureate Lloyd Shapley). It treats each feature (e.g. `tenure`, `MonthlyCharges`, `Contract`) as a "player" in a team game whose goal is to predict customer churn. SHAP calculates the exact marginal contribution of each player:

- **Baseline ($E[f(X)]$ / Base Value)**: The average churn expectation across all customers before looking at any individual customer's features.
- **Positive SHAP Value ($+SHAP$)**: A risk factor that pulls the prediction **higher** toward Churn.
- **Negative SHAP Value ($-SHAP$)**: A protective factor that pushes the prediction **lower** toward Retained.
- **Additivity Property**: For every single customer, the sum of all their SHAP values plus the baseline equals the exact model output:
  $$\text{Base Value} + \sum_{i=1}^{M} \text{SHAP}_i = \text{Model Output (Margin)}$$
  We formally verified this mathematical property across our test set.

---

## 2. Global Feature Importance (Top 10 Churn Drivers)

Based on mean absolute SHAP values across the 1,057 test customers:

| Rank | Feature | Mean \|SHAP\| | Primary Impact Direction | Business Explanation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `Contract_Two year` | **0.6232** | Strong Negative (Protective) | Two-year contractual commitments create immense retention inertia. |
| **2** | `Contract_One year` | **0.2901** | Negative (Protective) | One-year agreements substantially reduce churn hazard relative to month-to-month. |
| **3** | `InternetService_Fiber optic` | **0.1607** | Positive (Risk) | Fiber optic subscribers carry high churn hazard due to high bills ($70–$100/mo) and urban competitor switching options. |
| **4** | `monthly_to_total_ratio` | **0.1495** | Positive (Risk) | High ratio indicates new accounts that have not yet established relationship depth or amortized acquisition friction. |
| **5** | `has_fiber_no_support` | **0.1149** | Positive (Risk) | High-cost fiber subscribers who lack tech support face technical frustration without immediate resolution. |
| **6** | `tenure` | **0.0899** | Negative (Protective) | Each additional month of active tenure systematically decreases churn probability. |
| **7** | `PaymentMethod_Electronic check` | **0.0870** | Positive (Risk) | Manual monthly payment friction increases bill scrutiny compared to automated payments. |
| **8** | `MonthlyCharges` | **0.0757** | Positive (Risk) | Higher monthly financial commitment consistently correlates with elevated churn sensitivity. |
| **9** | `StreamingMovies_Yes` | **0.0683** | Directional / Modulating | Media consumption interacts with high data tiers. |
| **10** | `InternetService_No` | **0.0467** | Negative (Protective) | Low-cost phone-only subscribers ($20/mo) rarely switch carriers. |

---

## 3. Visual Interpretability Insights

### 3.1 Beeswarm Plot (`12_shap_beeswarm.png`)
The beeswarm plot displays every customer as a dot:
- **Red dots** = High feature value; **Blue dots** = Low feature value.
- **Horizontal position** = SHAP value (impact on churn).
- *Observation*: High `Contract_Two year` (Red dots) exclusively sit on the left ($-SHAP$), proving that two-year contracts protect against churn. Conversely, High `has_fiber_no_support` (Red dots) sit on the right ($+SHAP$), proving that fiber without support drives churn.

### 3.2 Dependence Plots (`13_shap_dependence_tenure.png` & `14_shap_dependence_monthly_charges.png`)
- **Tenure vs. Contract**: Between months 0 and 12, tenure has positive or neutral SHAP values (high risk). Beyond month 24, SHAP values become sharply negative, confirming that surviving past the 1-year mark cements customer loyalty.
- **Monthly Charges vs. Fiber Interaction**: For monthly charges above $70, the churn risk escalates rapidly, with customers lacking tech support experiencing the steepest upward risk slope.

---

## 4. Local Explanations: Individual Customer Waterfall Examples

In production and in the upcoming Streamlit application, customer support and account managers need actionable, customer-specific explanations:

### Customer Case Study: High-Risk Account (`5539-TMZLF`)
- **Predicted Churn Probability**: **80.1%** (Flagged by model at the 0.28 threshold)
- **Top Risk Factors**:
  1. `Contract_Two year` = 0 (Month-to-month plan): $+0.599$ SHAP
  2. `tenure` = Low (New customer): $+0.314$ SHAP
  3. `monthly_to_total_ratio` = High: $+0.288$ SHAP
  4. `InternetService_Fiber optic` = Yes: $+0.164$ SHAP
- **Actionable Retention Recommendation**:
  - Do not merely offer a generic discount.
  - Transition this customer from month-to-month to a discounted 1-year plan and bundle complimentary Tech Support to eliminate their primary churn drivers.

---

## 5. Artifacts Created

- Serialized global rankings: [`models/shap_feature_importance.json`](file:///f:/yash/projects/nexvia/churn-intelligence/models/shap_feature_importance.json)
- Sample customer case studies: [`models/sample_customer_explanations.json`](file:///f:/yash/projects/nexvia/churn-intelligence/models/sample_customer_explanations.json)
- Figures in [`reports/figures/`](file:///f:/yash/projects/nexvia/churn-intelligence/reports/figures):
  - `11_shap_summary_bar.png`
  - `12_shap_beeswarm.png`
  - `13_shap_dependence_tenure.png`
  - `14_shap_dependence_monthly_charges.png`
