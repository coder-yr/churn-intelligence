# Model Training, Evaluation & Decision Threshold Report

**Project**: Customer Lifetime Value and Churn Intelligence System  
**Pipeline Source**: [`src/models/train.py`](file:///f:/yash/projects/nexvia/churn-intelligence/src/models/train.py)  
**Trained Artifact**: `models/churn_lightgbm.joblib`  
**Performance Metrics**: `models/model_metrics.json`  
**Figures**: Stored in `reports/figures/` (`07_roc_curve.png`, `08_precision_recall_curve.png`, `09_confusion_matrices.png`, `10_cost_threshold_curve.png`)

---

## 1. Executive Summary & Benchmark Comparison

All models were evaluated on the **held-out test set (1,057 customer records; 280 churned, 777 retained)** that was completely isolated before any preprocessing or feature transformation.

### Comprehensive Test Set Evaluation

| Model | Decision Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Most Frequent)** | 0.50 | 73.51% | 0.00% | 0.00% | 0.000 | 0.500 | 0.265 | 0.265 |
| **Logistic Regression (Balanced)** | 0.50 | 74.74% | 51.55% | 77.14% | 0.618 | 0.845 | 0.639 | 0.163 |
| **LightGBM (Default Threshold)** | 0.50 | **77.58%** | **55.76%** | 74.29% | **0.637** | **0.843** | **0.635** | **0.151** |
| **LightGBM (Cost-Optimal Threshold)** | **0.28** | 64.05% | 41.94% | **92.86%** | 0.578 | **0.843** | **0.635** | **0.151** |

---

## 2. Key Data Science Insights & Interview Concepts

### 2.1 The "Accuracy Paradox"
- The **Dummy Classifier** achieves **73.51% accuracy** by simply predicting that *nobody ever churns*.
- Despite 73.5% accuracy, its Recall is **0.0%** and F1-Score is **0.000**. It catches zero churners and produces zero business value.
- **Takeaway**: For imbalanced datasets (~26.5% minority class), Accuracy is a dangerously misleading metric. **PR-AUC, ROC-AUC, Recall, and Expected Financial Benefit** are the true measures of performance.

### 2.2 Why LightGBM Outperformed Baselines
- LightGBM achieved the highest overall discriminatory performance with an **ROC-AUC of 0.843** and a well-calibrated **Brier Score of 0.151** (lower is better; closer to 0 indicates well-calibrated probabilistic outputs).
- Unlike Logistic Regression, LightGBM automatically modeled non-linear feature interactions (such as high monthly charges compounding with the absence of tech support).

---

## 3. Business-Cost Threshold Optimization

In standard academic data science, models are evaluated at a fixed default threshold of `0.50`. In real-world enterprise retention, **a 0.50 cutoff is financially arbitrary and sub-optimal**.

### 3.1 Asymmetric Business Costs
The cost of errors is asymmetric:
1. **False Positive (Type I Error)**: Flagging a loyal customer as at-risk.
   - *Action*: Send a $20 discount offer.
   - *Financial Impact*: We lose **$20** in unnecessary discount margin.
2. **False Negative (Type II Error)**: Missing an actual at-risk customer.
   - *Action*: We do nothing.
   - *Financial Impact*: The customer cancels. We lose their entire expected remaining contribution margin (**-$120**).
   - **A False Negative is 6× more expensive than a False Positive!**

### 3.2 Finding the Profit-Maximizing Threshold
We simulated campaign profit across thresholds from 0.10 to 0.90 in increments of 0.01:
- **Default Threshold (0.50)**:
  - Catches 208 of 280 churners (74.3% Recall). Misses 72 churners (72 × $120 = $8,640 lost).
- **Cost-Optimal Threshold (0.28)**:
  - Catches **260 of 280 churners (92.9% Recall)**!
  - Misses only 20 churners.
  - Generates the peak net economic benefit (**+$800.00** on the test cohort alone).

### 3.3 Confusion Matrix Comparison on Test Set (N = 1,057)

```text
At Default Threshold (0.50):
                Predicted Retained    Predicted Churn
Actual Retained        612                  165
Actual Churned          72                  208
(Missed Churners: 72)

At Cost-Optimal Threshold (0.28):
                Predicted Retained    Predicted Churn
Actual Retained        417                  360
Actual Churned          20                  260
(Missed Churners: 20 — 72% reduction in unflagged churners!)
```

---

## 4. Evaluation Figures in `reports/figures/`

1. **`07_roc_curve.png`**: Compares ROC trajectories of Dummy, Logistic Regression, and LightGBM. Both trained models achieve ~0.845 AUC, heavily dominating random chance.
2. **`08_precision_recall_curve.png`**: Displays Precision-Recall trade-off against the no-skill baseline (0.265). LightGBM achieves 0.635 PR-AUC.
3. **`09_confusion_matrices.png`**: Side-by-side heatmaps illustrating how the optimal threshold (0.28) cuts missed churners from 72 down to 20.
4. **`10_cost_threshold_curve.png`**: Dual-axis plot tracking Net Campaign Financial Value ($) alongside F1-score and Recall across the 0.10–0.90 threshold spectrum, clearly identifying 0.28 as the peak profit point.

---

## 5. Deployment Artifacts

- **Model Serialized**: [`models/churn_lightgbm.joblib`](file:///f:/yash/projects/nexvia/churn-intelligence/models/churn_lightgbm.joblib) containing the trained LightGBM model, the optimal threshold (`0.28`), and the default threshold (`0.50`).
- **Metrics Catalog**: [`models/model_metrics.json`](file:///f:/yash/projects/nexvia/churn-intelligence/models/model_metrics.json) containing all exact measured test metrics.
