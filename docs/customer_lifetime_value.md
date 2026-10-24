# Customer Lifetime Value (CLV) & Customer Segmentation Report

**Project**: Customer Lifetime Value and Churn Intelligence System  
**Pipeline Source**: [`src/models/clv_segmentation.py`](file:///f:/yash/projects/nexvia/churn-intelligence/src/models/clv_segmentation.py)  
**Artifacts**: `models/clv_summary.json`, `models/segmentation_kmeans.joblib`, `data/processed/telco_customer_clv_segmented.csv`  
**Figures**: Stored in `reports/figures/` (`19_clv_distribution.png`, `20_clv_sensitivity_heatmap.png`, `21_customer_segments_scatter.png`, `22_segment_profiles_comparison.png`)

---

## 1. Economic Framework & CLV Formula

A common flaw in beginner data science projects is equating *Monthly Charges* directly with *Profit*. In subscription telecommunications:
$$\text{Revenue} \ne \text{Profit}$$
To compute economically sound Customer Lifetime Value, we apply a **Discounted Cash Flow (DCF)** formulation grounded in conditional survival probability:

$$\text{CLV}_i = \sum_{t=1}^{H} \frac{m \cdot R_i \cdot S_i(t \mid T \ge \text{tenure}_i)}{(1 + d)^t}$$

### Formula Parameters & Corporate Finance Assumptions:
1. **$R_i$ (Monthly Charge)**: The recurring monthly billing rate of customer $i$.
2. **$m$ (Contribution Margin Rate = $70\%$)**: Assumes $30\%$ of recurring revenue covers direct network maintenance, customer support, and server infrastructure costs.
3. **$d$ (Monthly Discount Rate = $0.797\%$)**: Derived from an annual Weighted Average Cost of Capital (WACC) of $10\%$:
   $$d = (1 + 0.10)^{1/12} - 1 \approx 0.007974$$
4. **$H$ (Forecast Horizon = 24 Months)**: A 2-year forward window appropriate for telecom strategic planning.
5. **$S_i(t \mid T \ge \text{tenure}_i)$ (Conditional Survival)**: The probability that a customer will survive an additional $t$ months, given that they have already remained active for $\text{tenure}_i$ months:
   $$S_i(t \mid T \ge \text{tenure}_i) = \frac{S(\text{tenure}_i + t)}{S(\text{tenure}_i)}$$
   conditioned on their specific contract type from Phase 7.

---

## 2. Overall Portfolio Economic Benchmarks

| Economic Metric | Measured Value | Strategic Definition |
| :--- | :---: | :--- |
| **Mean Forward 24-Month CLV** | **$843.94** | Expected gross profit per customer over the next 24 months. |
| **Median Forward 24-Month CLV** | **$889.61** | Half the customer base has an expected value above $890. |
| **Total Portfolio Forward CLV** | **$5,943,848.92** | Total 24-month gross margin asset value of the 7,043 customer accounts. |
| **Mean Historical Realized Profit** | **$1,595.81** | Total cumulative gross profit generated to date ($70\% \times \text{TotalCharges}$). |

---

## 3. Multi-Dimensional Sensitivity Analysis

Because margins and macroeconomic interest rates fluctuate, we computed average customer CLV across a range of parameter scenarios:

| Horizon | Contribution Margin | Annual Discount Rate (WACC) | Average Forward CLV ($) |
| :---: | :---: | :---: | :---: |
| **12 Months** | 60% | 10% | $427.35 |
| **12 Months** | 70% | 10% | $498.57 |
| **12 Months** | 80% | 10% | $569.80 |
| **24 Months** | 60% | 8% | $732.18 |
| **24 Months** | **70% (Baseline)** | **10% (Baseline)** | **$843.94** |
| **24 Months** | 80% | 12% | $951.20 |
| **36 Months** | 60% | 10% | $965.12 |
| **36 Months** | 70% | 10% | $1,125.97 |
| **36 Months** | 80% | 10% | $1,286.82 |

*(Figure `20_clv_sensitivity_heatmap.png` visualizes the complete sensitivity grid).*

---

## 4. K-Means Customer Segmentation (Risk vs. Value)

Using standardized behavioral dimensions (`tenure`, `MonthlyCharges`, `clv_24m`, and `churn_probability`), K-Means clustering ($K=4$) partitioned the customer base into four distinct business segments:
- **Silhouette Score**: `0.389`
- **Davies-Bouldin Index**: `0.912`

```text
       HIGH VALUE
           ▲
           │   [High-Value Loyalists]        [Vulnerable High-Spenders]
           │   • Mean Tenure: 58.1m          • Mean Tenure: 16.6m
           │   • Monthly Bill: $90.36        • Monthly Bill: $85.22
           │   • 24m CLV: $1,222.52          • 24m CLV: $1,060.82
           │   • Churn Risk: 24.1%           • Churn Risk: 63.3%  <-- CRITICAL RETENTION FOCUS
           │
           │   [Budget Steady Anchors]       [New Onboarding Risks]
           │   • Mean Tenure: 45.2m          • Mean Tenure: 8.7m
           │   • Monthly Bill: $30.73        • Monthly Bill: $38.16
           │   • 24m CLV: $423.96            • 24m CLV: $484.09
           │   • Churn Risk: 14.2%           • Churn Risk: 43.0%
           └────────────────────────────────────────────────────────►
          LOW RISK                                                HIGH RISK
```

### Segment Profiles & Strategic Retention Playbook

#### 1. Vulnerable High-Spenders (N = 2,052 customers; 29.1% of base)
- **Profile**: Paying high monthly charges ($85.22/mo) with relatively short tenure (16.6 months) and an alarmingly high **63.3% churn probability**.
- **Portfolio Value at Risk**: Over **$2.17 Million** in forward CLV is concentrated in this single at-risk cluster.
- **Action Plan**: Priority #1 for proactive retention campaigns. Transition from month-to-month to discounted annual agreements and bundle free Tech Support.

#### 2. High-Value Loyalists (N = 1,950 customers; 27.7% of base)
- **Profile**: Long tenure (58.1 months), premium spending ($90.36/mo), strong multi-year contracts, and low churn probability (24.1%).
- **Forward CLV**: Highest in the portfolio at **$1,222.52 per account**.
- **Action Plan**: VIP loyalty recognition, early hardware upgrades, and cross-sell premium add-ons (streaming bundles). Avoid unsolicited discounts.

#### 3. Budget Steady Anchors (N = 1,480 customers; 21.0% of base)
- **Profile**: Established accounts (45.2 months tenure) with low bills ($30.73/mo, mostly phone or basic DSL) and lowest churn hazard (14.2%).
- **Forward CLV**: Modest ($423.96).
- **Action Plan**: Maintain flawless service reliability, automated billing, and low self-service overhead.

#### 4. New Onboarding Risks (N = 1,561 customers; 22.2% of base)
- **Profile**: Newest accounts (tenure 8.7 months) with basic service bundles and high early attrition risk (43.0%).
- **Action Plan**: Automated onboarding email journeys, customer satisfaction check-ins at Day 30/60/90, and setup assistance to prevent initial dissatisfaction.

---

## 5. Figures Generated in [`reports/figures/`](file:///f:/yash/projects/nexvia/churn-intelligence/reports/figures)

1. `19_clv_distribution.png`: Histogram and KDE curve of forward 24-month CLV showing the mean ($843.94) and median ($889.61).
2. `20_clv_sensitivity_heatmap.png`: Matrix mapping CLV changes across operating margins and capital discount rates.
3. `21_customer_segments_scatter.png`: Interactive scatter visualization plotting Churn Probability vs. 24-Month CLV colored by customer segment with the 0.28 decision boundary.
4. `22_segment_profiles_comparison.png`: Grouped bar chart comparing tenure, monthly charges, and forward value across the 4 clusters.
