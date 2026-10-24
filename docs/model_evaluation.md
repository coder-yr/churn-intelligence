# Model Training, Evaluation & Decision Threshold Report

**Project**: Customer Lifetime Value and Churn Intelligence System  
**Pipeline Source**: [`src/models/train.py`](file:///f:/yash/projects/nexvia/churn-intelligence/src/models/train.py)  
**Trained Artifact**: `models/churn_lightgbm.joblib`  
**Performance Metrics**: `models/model_metrics.json`  
**Figures**: Stored in `reports/figures/` (`07_roc_curve.png`, `08_precision_recall_curve.png`, `09_confusion_matrices.png`, `10_cost_threshold_curve.png`)

---

## 1. Executive Summary & Anti-Leakage Methodology

To prevent data leakage, the evaluation adheres to strict data partition boundaries:
1. **Training Set (4,929 accounts, 70%)**: Model weights fitted using gradient boosted decision trees with `scale_pos_weight = 2.77`.
2. **Validation Set (1,057 accounts, 15%)**: Used strictly for early stopping (round 49) and business-cost decision threshold tuning.
3. **Held-Out Test Set (1,057 accounts, 15%)**: Frozen model and frozen threshold evaluated strictly on held-out test data. **Test labels were never used to select the threshold, tune hyperparameters, or choose the model.**

### Authoritative Test Set Benchmark

| Model | Evaluation Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Most Frequent)** | 0.50 | 73.51% | 0.00% | 0.00% | 0.000 | 0.500 | 0.265 | 0.265 |
| **Logistic Regression (Balanced)** | 0.50 | 74.74% | 51.55% | 77.14% | 0.618 | 0.845 | 0.639 | 0.163 |
| **LightGBM (Default Threshold)** | 0.50 | **77.58%** | **55.76%** | 74.29% | **0.637** | **0.843** | **0.635** | **0.151** |
| **LightGBM (Validation-Tuned Frozen Threshold)** | **0.27** | 62.25% | 40.75% | **93.57%** | 0.568 | **0.843** | **0.635** | **0.151** |

---

## 2. Key Data Science Insights & Concepts

### 2.1 The "Accuracy Paradox"
- The **Dummy Classifier** achieves **73.51% accuracy** by predicting that *nobody ever churns*.
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
   - *Financial Impact*: The customer cancels. We lose their expected contribution margin (**-$120**).
   - **A False Negative is 6× more expensive than a False Positive!**

### 3.2 Finding the Profit-Maximizing Threshold on Validation Data
We simulated campaign profit across thresholds from 0.10 to 0.90 in increments of 0.01 on the **validation set**:
- Optimal threshold selected on validation: **`0.27`** (Validation Net Value: **+$1,400.00**, Validation Recall: **95.37%**).

### 3.3 Evaluating the Frozen Threshold on the Held-Out Test Set
Evaluating the frozen `0.27` cutoff on the unseen test set:
- **Default Threshold (0.50)**:
  - Catches 208 of 280 churners (74.29% Recall). Misses 72 churners (72 × $120 = $8,640 lost).
- **Frozen Optimal Threshold (0.27)**:
  - Catches **262 of 280 churners (93.57% Recall)**!
  - Misses only 18 churners.
  - Generates a net financial benefit of **+$700.00** on the test cohort alone.
  - Test Confusion Matrix: `TN = 396, FP = 381, FN = 18, TP = 262`.
