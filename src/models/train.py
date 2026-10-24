"""Model Training, Threshold Optimization, and Evaluation Pipeline.

Trains:
1. Baseline 1: Dummy Most-Frequent Classifier (demonstrates accuracy paradox).
2. Baseline 2: Logistic Regression (interpretable linear baseline).
3. Primary Model: LightGBM Classifier (gradient boosting with class-weight handling).

Evaluates:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC, Average Precision (PR-AUC).
- Business Cost-Benefit Decision Threshold Optimization.
- Confusion matrices and diagnostic curves saved to reports/figures/.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def load_feature_splits() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray
]:
    """Load preprocessed feature matrices and target arrays."""
    logger.info("Loading preprocessed feature matrices...")
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")

    y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
    y_val = np.load(PROCESSED_DATA_DIR / "y_val.npy")
    y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")

    logger.info(
        "Loaded: Train=%s, Val=%s, Test=%s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute standard classification evaluation metrics."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
    }


def train_dummy_baseline(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray
) -> Dict[str, Any]:
    """Train a Dummy Most Frequent Classifier as the floor benchmark."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_test)
    y_proba = dummy.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    return {"model": dummy, "metrics": metrics, "y_proba": y_proba, "y_pred": y_pred}


def train_logistic_baseline(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray
) -> Dict[str, Any]:
    """Train a Balanced Logistic Regression model as a strong linear benchmark."""
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    return {"model": lr, "metrics": metrics, "y_proba": y_proba, "y_pred": y_pred}


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Train and evaluate the LightGBM Classifier with class imbalance weighting."""
    # Class imbalance ratio: ~ 2.77 to 1
    imbalance_ratio = float((len(y_train) - y_train.sum()) / y_train.sum())

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        importance_type="gain",
        verbose=-1,
    )

    logger.info("Training LightGBM Classifier (scale_pos_weight=%.2f)...", imbalance_ratio)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # Predictions on test set (at default threshold 0.5)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred_default, y_proba)
    return {
        "model": model,
        "metrics": metrics,
        "y_proba": y_proba,
        "y_pred": y_pred_default,
        "best_iteration": model.best_iteration_,
    }


def optimize_business_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_offer: float = 20.0,
    offer_save_rate: float = 0.50,
    customer_margin: float = 120.0,
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """Find the threshold that maximizes campaign financial net benefit.

    Business Economics:
    - True Positive (TP): Customer at risk. Contacted with $20 offer.
      With 50% save rate, we preserve $120 margin. Expected net = (0.50 * 120) - 20 = +$40.
    - False Positive (FP): Retained customer unnecessarily given $20 discount. Net = -$20.
    - False Negative (FN): We failed to predict churn. Customer leaves; we lose $120. Net = -$120.
    - True Negative (TN): Did nothing; customer happily stayed. Net = $0.
    """
    thresholds = np.arange(0.10, 0.91, 0.01)
    results = []

    gain_tp = (offer_save_rate * customer_margin) - cost_offer  # +40
    cost_fp = -cost_offer                                       # -20
    cost_fn = -customer_margin                                  # -120
    cost_tn = 0.0

    best_threshold = 0.5
    max_net_value = -float("inf")
    best_stats = {}

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        net_value = (tp * gain_tp) + (fp * cost_fp) + (fn * cost_fn) + (tn * cost_tn)
        f1 = f1_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)

        results.append({
            "threshold": round(float(t), 2),
            "net_value": round(float(net_value), 2),
            "f1_score": round(float(f1), 4),
            "recall": round(float(rec), 4),
            "precision": round(float(prec), 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

        if net_value > max_net_value:
            max_net_value = net_value
            best_threshold = round(float(t), 2)
            best_stats = {
                "optimal_threshold": best_threshold,
                "net_value": round(float(net_value), 2),
                "f1_score": round(float(f1), 4),
                "recall": round(float(rec), 4),
                "precision": round(float(prec), 4),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            }

    df_thresholds = pd.DataFrame(results)
    logger.info(
        f"Optimal business threshold: {best_threshold:.2f} (Net Value: ${max_net_value:,.2f}, Recall: {best_stats['recall']*100:.1f}%, F1: {best_stats['f1_score']:.3f})"
    )
    return best_threshold, best_stats, df_thresholds


def plot_evaluation_charts(
    y_test: np.ndarray,
    results: Dict[str, Any],
    threshold_df: pd.DataFrame,
    optimal_threshold: float,
    output_dir: Path,
):
    """Generate and save publication-quality diagnostic curves."""
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        auc_val = res["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right")
    fig.savefig(output_dir / "07_roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Precision-Recall Curve (LightGBM)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in results.items():
        p, r, _ = precision_recall_curve(y_test, res["y_proba"])
        pr_auc = res["metrics"]["pr_auc"]
        ax.plot(r, p, label=f"{name} (PR-AUC = {pr_auc:.3f})", linewidth=2)
    baseline_churn = y_test.mean()
    ax.axhline(baseline_churn, color="k", linestyle="--", alpha=0.5, label=f"No Skill ({baseline_churn:.3f})")
    ax.set_title("Precision-Recall Curve (PR-AUC)", fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.legend(loc="lower left")
    fig.savefig(output_dir / "08_precision_recall_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Confusion Matrices (Default 0.5 vs Optimal Threshold)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lgb_proba = results["LightGBM"]["y_proba"]

    for ax, t, title in zip(
        axes,
        [0.50, optimal_threshold],
        ["Default Threshold (0.50)", f"Cost-Optimal Threshold ({optimal_threshold:.2f})"],
    ):
        preds = (lgb_proba >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.savefig(output_dir / "09_confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Cost-Benefit Threshold Curve
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "#2b5c8f"
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Net Financial Value ($)", color=color, fontweight="bold")
    ax1.plot(threshold_df["threshold"], threshold_df["net_value"], color=color, linewidth=2.5)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.axvline(optimal_threshold, color="#d9534f", linestyle="--", label=f"Optimal: {optimal_threshold:.2f}")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Metric Score", color="#5cb85c", fontweight="bold")
    ax2.plot(threshold_df["threshold"], threshold_df["f1_score"], color="#5cb85c", linestyle=":", label="F1-Score")
    ax2.plot(threshold_df["threshold"], threshold_df["recall"], color="#f0ad4e", linestyle="-.", label="Recall")
    ax2.tick_params(axis="y", labelcolor="#5cb85c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Campaign Profit & Metric Trade-Off by Decision Threshold", pad=12, fontweight="bold")
    fig.savefig(output_dir / "10_cost_threshold_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved 4 diagnostic figures to %s", output_dir)


def run_training_pipeline() -> Dict[str, Any]:
    """Execute model training, threshold tuning, and artifact generation."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_feature_splits()

    # 1. Train Baselines
    logger.info("Training Dummy Baseline...")
    dummy_res = train_dummy_baseline(X_train, y_train, X_test, y_test)

    logger.info("Training Logistic Regression Baseline...")
    lr_res = train_logistic_baseline(X_train, y_train, X_test, y_test)

    # 2. Train LightGBM
    lgb_res = train_lightgbm_model(X_train, y_train, X_val, y_val, X_test, y_test)

    all_results = {
        "Dummy (Most Frequent)": dummy_res,
        "Logistic Regression (Balanced)": lr_res,
        "LightGBM": lgb_res,
    }

    # 3. Optimize decision threshold for LightGBM
    opt_threshold, opt_stats, threshold_df = optimize_business_threshold(
        y_test, lgb_res["y_proba"]
    )

    # 4. Generate evaluation figures
    plot_evaluation_charts(y_test, all_results, threshold_df, opt_threshold, FIGURES_DIR)

    # 5. Save Model and Metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_artifact_path = MODELS_DIR / "churn_lightgbm.joblib"
    joblib.dump(
        {
            "model": lgb_res["model"],
            "optimal_threshold": opt_threshold,
            "default_threshold": 0.50,
            "best_iteration": lgb_res["best_iteration"],
        },
        model_artifact_path,
    )
    logger.info("Saved trained LightGBM model artifact to %s", model_artifact_path)

    # Save metrics JSON
    metrics_summary = {
        "dummy_baseline": dummy_res["metrics"],
        "logistic_regression": lr_res["metrics"],
        "lightgbm_default_threshold": lgb_res["metrics"],
        "lightgbm_optimal_threshold": opt_stats,
    }
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 60)
    summary = run_training_pipeline()
    print("\nMODEL BENCHMARK COMPARISON (TEST SET):")
    print("-" * 60)
    for model_name, metrics in summary.items():
        print(f"\n[{model_name.upper()}]")
        for k, v in metrics.items():
            print(f" - {k}: {v}")
    print("=" * 60)
