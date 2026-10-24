# Phase 10: Retention Campaign Financial ROI Simulator

## 1. Executive Problem Statement

Data Science models are only as valuable as their business impact. In an executive review or C-suite presentation, stakeholders do not evaluate retention programs by ROC-AUC or log-loss. They ask:
> **"If we invest $50,000 in customer retention this quarter, how many customers do we save, what is our net profit after offer expenses, and what is our return on investment (ROI)?"**

The **Retention Campaign Financial ROI Simulator** bridges predictive machine learning with corporate finance (EBITDA, Discounted Cash Flow, and margin preservation).

---

## 2. Financial Methodology & Accounting Formulas

### Discounted Value of a Retained Customer (NPV)
A saved customer generates monthly gross profit rather than revenue alone. Given monthly ARPU ($R$) and gross profit margin ($m$):
$$\text{Monthly Gross Profit} = R \times m$$

Future months are discounted using corporate Weighted Average Cost of Capital (WACC, default 10% annual, converted to monthly $d = (1+0.10)^{1/12} - 1 \approx 0.797\%$):
$$\text{NPV}_{\text{Saved}} = \sum_{t=1}^{H} \frac{R \cdot m}{(1 + d)^t}$$
For a high-value customer ($ARPU = \$85.22$, $m = 70\%$, 12-month horizon), saving one customer preserves **$680.08** in net present gross profit.

### Campaign P&L Identities
1. **Total Campaign Cost**:
   $$\text{Cost} = N_{\text{targeted}} \times \text{Cost}_{\text{offer}}$$
2. **Incrementally Saved Accounts**:
   $$\Delta N_{\text{saved}} = N_{\text{targeted}} \times \text{Uplift}$$
3. **Gross Financial Benefit**:
   $$\text{Benefit} = \Delta N_{\text{saved}} \times \text{NPV}_{\text{Saved}}$$
4. **Net Campaign Profit**:
   $$\text{Net Profit} = \text{Benefit} - \text{Cost}$$
5. **Campaign ROI Percentage**:
   $$\text{ROI} = \left(\frac{\text{Net Profit}}{\text{Cost}}\right) \times 100\%$$
6. **Break-Even Retention Uplift**:
   $$\text{Uplift}_{\text{break-even}} = \frac{\text{Cost}_{\text{offer}}}{\text{NPV}_{\text{Saved}}}$$

---

## 3. Targeting Policy Benchmark: Mass vs Risk-Only vs Precision

We evaluated three strategic policies across the IBM Telco customer base:

| Policy | Targeted Accounts | Cost per Offer | Total Campaign Cost | Incrementally Saved Accounts | Gross Benefit Saved | Net Profit | Return on Investment (ROI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Mass Blanket (Untargeted)** | 7,043 | $25 | $176,075 | 211 | $109,047 | **-$67,028** | **-38.1% (Loss)** |
| **2. Risk-Only (All Prob $\ge$ 0.28)** | 2,300 | $35 | $80,500 | 138 | $81,980 | **+$1,480** | **+1.8% (Barely Breakeven)** |
| **3. Precision (High-CLV Persuadables)** | **1,200** | **$45** | **$54,000** | **144** | **$97,932** | **+$43,932** | **+81.3% (Highly Profitable)** |

### Strategic Insights:
- **Mass Blanket Destroys Value**: Distributing discounts to everyone wastes $176k. Most customers were going to stay anyway (Sure Things) or were low-value accounts with low margin, causing a **-$67k net loss**.
- **Risk-Only Barely Breaks Even**: Targeting all high-probability churners wastes expensive offers on *Lost Causes* (who churn regardless) and budget subscribers whose ARPU is too low to justify a $35 offer.
- **Precision Targeting Wins**: By combining **Segment 2 (Vulnerable High-Spenders, ARPU $85.22)** with **Uplift Modeling (Persuadables)**, the operator saves more high-value accounts (144 vs 138), spends $26,500 less in campaign expenditure, and generates **+$43,932 in clean EBITDA profit (+81.3% ROI)**!

---

## 4. Visualizations & Sensitivity Analysis

- **Figure 26**: [`reports/figures/26_roi_sensitivity_curves.png`](file:///f:/yash/projects/nexvia/churn-intelligence/reports/figures/26_roi_sensitivity_curves.png)
  - Illustrates net profit across uplift percentages for $25, $40, and $60 offer costs on 1,000 high-value accounts.
  - Clearly demonstrates the break-even threshold where positive return begins.
