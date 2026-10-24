"""
Uplift Modeling and Treatment Effect Estimation.

Implements:
1. Two-Model T-Learner using LightGBM estimators for Treatment and Control groups.
2. Individual Treatment Effect (ITE / CATE): tau_hat(X) = mu_1(X) - mu_0(X).
3. Uplift Evaluation: Decile Analysis, Cumulative Gain Curve, and Qini Curve.
4. Uplift Quadrant Segmentation: Persuadables, Sure Things, Lost Causes, Sleeping Dogs.
5. Telecom retention mapping and cost-effective targeted policy.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "uplift_train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "uplift_test.csv"

FEATURE_COLS = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
    "zip_Suburban",
    "zip_Urban",
    "channel_Phone",
    "channel_Web",
]


class TwoModelTLearner:
    """
    Two-Model T-Learner for Causal Uplift Modeling.
    
    Fits two independent LightGBM estimators:
    - Model 0 (Control): trained on observations where Treatment = 0
    - Model 1 (Treated): trained on observations where Treatment = 1
    
    Predicted CATE tau(X) = P(Y=1 | X, T=1) - P(Y=1 | X, T=0)
    """

    def __init__(self, lgb_params: Optional[Dict[str, Any]] = None):
        if lgb_params is None:
            self.lgb_params = {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "random_state": 42,
                "verbose": -1,
            }
        else:
            self.lgb_params = lgb_params

        self.model_control = lgb.LGBMClassifier(**self.lgb_params)
        self.model_treated = lgb.LGBMClassifier(**self.lgb_params)
        self.is_fitted = False
        self.feature_names = FEATURE_COLS

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series) -> "TwoModelTLearner":
        """
        Fits separate models on control (treatment=0) and treated (treatment=1) subsets.
        """
        control_mask = (treatment == 0)
        treated_mask = (treatment == 1)

        X_control = X[control_mask]
        y_control = y[control_mask]
        X_treated = X[treated_mask]
        y_treated = y[treated_mask]

        self.model_control.fit(X_control, y_control)
        self.model_treated.fit(X_treated, y_treated)
        self.is_fitted = True

        return self

    def predict_counterfactuals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts response probabilities under Control and Treatment:
        Returns:
            (mu_0, mu_1) where:
            mu_0 = P(Y=1 | X, T=0)
            mu_1 = P(Y=1 | X, T=1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        mu_0 = self.model_control.predict_proba(X[self.feature_names])[:, 1]
        mu_1 = self.model_treated.predict_proba(X[self.feature_names])[:, 1]
        return mu_0, mu_1

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculates predicted uplift tau_hat(X) = mu_1(X) - mu_0(X).
        """
        mu_0, mu_1 = self.predict_counterfactuals(X)
        return mu_1 - mu_0

    def save(self, filepath: Path) -> None:
        """
        Serializes models and parameters cleanly to avoid __main__ namespace issues.
        """
        joblib.dump({
            "model_control": self.model_control,
            "model_treated": self.model_treated,
            "feature_names": self.feature_names,
            "lgb_params": self.lgb_params,
            "is_fitted": self.is_fitted,
        }, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "TwoModelTLearner":
        """
        Loads serialized T-Learner models.
        """
        data = joblib.load(filepath)
        instance = cls(lgb_params=data.get("lgb_params"))
        instance.model_control = data["model_control"]
        instance.model_treated = data["model_treated"]
        instance.feature_names = data["feature_names"]
        instance.is_fitted = data.get("is_fitted", True)
        return instance


def compute_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_preds: np.ndarray,
    num_bins: int = 100,
) -> Dict[str, Any]:
    """
    Computes Qini curve points and Qini coefficient.
    
    Qini formula at proportion p of population:
        Q(p) = Y_{T, p} - Y_{C, p} * (N_{T, p} / N_{C, p})
        where Y_{T, p} is sum of positive outcomes among treated in top p,
              Y_{C, p} is sum of positive outcomes among controls in top p.
    """
    df = pd.DataFrame({
        "y": y_true,
        "treatment": treatment,
        "uplift": uplift_preds,
    }).sort_values(by="uplift", ascending=False).reset_index(drop=True)

    n_total = len(df)
    bin_sizes = np.linspace(0, n_total, num_bins + 1, dtype=int)

    fractions = []
    qini_values = []
    random_qini_values = []

    total_treated_pos = df[df["treatment"] == 1]["y"].sum()
    total_control_pos = df[df["treatment"] == 0]["y"].sum()
    total_treated_n = (df["treatment"] == 1).sum()
    total_control_n = (df["treatment"] == 0).sum()

    # Total theoretical campaign gain at 100% targeted
    total_gain = total_treated_pos - (total_control_pos * (total_treated_n / total_control_n))

    for size in bin_sizes:
        if size == 0:
            fractions.append(0.0)
            qini_values.append(0.0)
            random_qini_values.append(0.0)
            continue

        subset = df.iloc[:size]
        n_t = (subset["treatment"] == 1).sum()
        n_c = (subset["treatment"] == 0).sum()
        y_t = subset[subset["treatment"] == 1]["y"].sum()
        y_c = subset[subset["treatment"] == 0]["y"].sum()

        if n_c > 0:
            qini_val = y_t - (y_c * (n_t / n_c))
        else:
            qini_val = y_t

        fraction = size / n_total
        fractions.append(float(fraction))
        qini_values.append(float(qini_val))
        random_qini_values.append(float(total_gain * fraction))

    # Qini score: Area between model curve and random diagonal
    # Using trapezoidal integration (NumPy 2.0 compatible)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area_model = trapz_fn(qini_values, fractions)
    area_random = trapz_fn(random_qini_values, fractions)
    qini_score = area_model - area_random

    return {
        "fractions": fractions,
        "qini_values": qini_values,
        "random_qini_values": random_qini_values,
        "qini_score": float(qini_score),
        "total_campaign_gain": float(total_gain),
    }


def compute_decile_uplift(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_preds: np.ndarray,
    num_deciles: int = 10,
) -> pd.DataFrame:
    """
    Aggregates test set into deciles ranked by predicted uplift.
    Calculates empirical treated conversion, control conversion, and observed uplift.
    """
    df = pd.DataFrame({
        "y": y_true,
        "treatment": treatment,
        "uplift": uplift_preds,
    })

    # Decile 1 = highest predicted uplift, Decile 10 = lowest
    df["decile"] = pd.qcut(df["uplift"].rank(method="first", ascending=False), q=num_deciles, labels=False) + 1

    decile_stats = []
    for d in range(1, num_deciles + 1):
        sub = df[df["decile"] == d]
        t_sub = sub[sub["treatment"] == 1]
        c_sub = sub[sub["treatment"] == 0]

        n_t = len(t_sub)
        n_c = len(c_sub)
        y_t = t_sub["y"].sum()
        y_c = c_sub["y"].sum()

        t_rate = y_t / n_t if n_t > 0 else 0.0
        c_rate = y_c / n_c if n_c > 0 else 0.0
        observed_uplift = t_rate - c_rate

        decile_stats.append({
            "decile": int(d),
            "size": len(sub),
            "n_treated": int(n_t),
            "n_control": int(n_c),
            "mean_pred_uplift": float(sub["uplift"].mean()),
            "treated_rate": float(t_rate),
            "control_rate": float(c_rate),
            "observed_uplift": float(observed_uplift),
        })

    return pd.DataFrame(decile_stats)


def classify_uplift_quadrants(
    mu_0: np.ndarray,
    mu_1: np.ndarray,
    uplift: np.ndarray,
    uplift_threshold: float = 0.05,
    baseline_threshold: float = 0.15,
) -> pd.Series:
    """
    Assigns each customer to one of four behavioral marketing quadrants:
    1. Persuadables: High uplift (uplift >= threshold) - targets for retention offers.
    2. Sure Things: High baseline engagement (mu_0 >= threshold) but low uplift.
    3. Lost Causes: Low baseline engagement and low uplift.
    4. Sleeping Dogs: Negative uplift (outreach actually decreases positive outcome).
    """
    quadrants = []
    for m0, m1, up in zip(mu_0, mu_1, uplift):
        if up < 0.0:
            quadrants.append("Sleeping Dogs")
        elif up >= uplift_threshold:
            quadrants.append("Persuadables")
        elif m0 >= baseline_threshold:
            quadrants.append("Sure Things")
        else:
            quadrants.append("Lost Causes")

    return pd.Series(quadrants)


def plot_uplift_curves(
    qini_results: Dict[str, Any],
    decile_df: pd.DataFrame,
    quadrants: pd.Series,
    output_dir: Path = FIGURES_DIR,
):
    """
    Generates and saves the three standard uplift visualizations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Figure 23: Qini Curve
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(
        qini_results["fractions"],
        qini_results["qini_values"],
        color="#1E3A8A",
        linewidth=2.5,
        label=f"Two-Model T-Learner (Qini score = {qini_results['qini_score']:.1f})",
    )
    plt.plot(
        qini_results["fractions"],
        qini_results["random_qini_values"],
        color="#9CA3AF",
        linestyle="--",
        linewidth=1.8,
        label="Random Targeting Baseline",
    )
    plt.title("Figure 23: Uplift Qini Curve (Cumulative Incremental Conversions)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Proportion of Population Targeted (Ranked by Predicted Uplift)", fontsize=11)
    plt.ylabel("Cumulative Incremental Conversions (Qini)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "23_uplift_qini_curve.png")
    plt.close()

    # 2. Figure 24: Cumulative Gain / Observed Uplift by Decile
    plt.figure(figsize=(10, 5), dpi=300)
    x = decile_df["decile"]
    width = 0.35
    plt.bar(
        x - width / 2,
        decile_df["treated_rate"] * 100,
        width,
        label="Treated Conversion Rate (%)",
        color="#2563EB",
        alpha=0.85,
    )
    plt.bar(
        x + width / 2,
        decile_df["control_rate"] * 100,
        width,
        label="Control Conversion Rate (%)",
        color="#94A3B8",
        alpha=0.85,
    )
    plt.plot(
        x,
        decile_df["observed_uplift"] * 100,
        color="#DC2626",
        marker="o",
        linewidth=2.2,
        label="Incremental Uplift (% pts)",
    )
    plt.title("Figure 24: Observed Uplift by Predicted Uplift Decile", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Decile (Decile 1 = Top 10% Highest Predicted Uplift)", fontsize=11)
    plt.ylabel("Rate (%)", fontsize=11)
    plt.xticks(x)
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(output_dir / "24_uplift_cumulative_gain.png")
    plt.close()

    # 3. Figure 25: Uplift Quadrant Distribution
    plt.figure(figsize=(8, 5.5), dpi=300)
    quadrant_counts = quadrants.value_counts()
    colors = {
        "Persuadables": "#10B981",  # Green (Primary target!)
        "Sure Things": "#3B82F6",   # Blue (Organic retention)
        "Lost Causes": "#6B7280",   # Gray (Futile spend)
        "Sleeping Dogs": "#EF4444", # Red (Do not disturb!)
    }
    bar_colors = [colors.get(q, "#94A3B8") for q in quadrant_counts.index]
    bars = plt.bar(quadrant_counts.index, quadrant_counts.values, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.5)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(quadrants)) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + (len(quadrants) * 0.01),
            f"{h:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title("Figure 25: Uplift Quadrant Segmentation (Targeting Archetypes)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Number of Customers", fontsize=11)
    plt.ylim(0, quadrant_counts.max() * 1.15)
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "25_uplift_quadrants.png")
    plt.close()


def train_and_evaluate_uplift():
    """
    Main training and evaluation pipeline for Uplift Modeling.
    """
    print("=" * 65)
    print("Training Uplift T-Learner on Randomized Benchmark Dataset...")
    print("=" * 65)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df[FEATURE_COLS]
    t_train = train_df["treatment"]
    y_train = train_df["visit"]

    X_test = test_df[FEATURE_COLS]
    t_test = test_df["treatment"]
    y_test = test_df["visit"]

    # Initialize and fit T-Learner
    t_learner = TwoModelTLearner()
    t_learner.fit(X_train, t_train, y_train)

    # Compute counterfactual predictions on test set
    mu_0_test, mu_1_test = t_learner.predict_counterfactuals(X_test)
    uplift_test = t_learner.predict_uplift(X_test)

    # Diagnostic model performance
    auc_control = roc_auc_score(y_test[t_test == 0], mu_0_test[t_test == 0])
    auc_treated = roc_auc_score(y_test[t_test == 1], mu_1_test[t_test == 1])
    print(f"Control Model Test ROC-AUC: {auc_control:.4f}")
    print(f"Treated Model Test ROC-AUC: {auc_treated:.4f}")

    # Decile uplift analysis
    decile_df = compute_decile_uplift(y_test.values, t_test.values, uplift_test, num_deciles=10)
    print("\nDecile Uplift Summary (Top 3 vs Bottom 3):")
    print(decile_df[["decile", "mean_pred_uplift", "treated_rate", "control_rate", "observed_uplift"]].to_string(index=False))

    # Qini evaluation
    qini_results = compute_qini_curve(y_test.values, t_test.values, uplift_test)
    print(f"\nQini Score (Model Area - Random Area): {qini_results['qini_score']:.2f}")
    print(f"Total Full-Campaign Incremental Gain: {qini_results['total_campaign_gain']:.1f} visits")

    # Quadrant Classification
    quadrants = classify_uplift_quadrants(mu_0_test, mu_1_test, uplift_test)
    print("\nTargeting Quadrants Distribution (Test Set):")
    for quad, cnt in quadrants.value_counts().items():
        print(f"  {quad:15s}: {cnt:5d} ({cnt / len(quadrants) * 100:5.1f}%)")

    # Generate plots
    plot_uplift_curves(qini_results, decile_df, quadrants)
    print(f"\nSaved Uplift figures 23, 24, 25 to {FIGURES_DIR}")

    # Serialize artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t_learner.save(MODELS_DIR / "uplift_models.joblib")

    # Prepare JSON summary
    uplift_summary = {
        "dataset": "Hillstrom MineThatData Randomized Controlled Trial",
        "sample_size": {
            "train": len(train_df),
            "test": len(test_df),
        },
        "model_performance": {
            "auc_control": round(float(auc_control), 4),
            "auc_treated": round(float(auc_treated), 4),
            "qini_score": round(float(qini_results["qini_score"]), 2),
            "total_campaign_incremental_gain": round(float(qini_results["total_campaign_gain"]), 1),
        },
        "deciles": decile_df.to_dict(orient="records"),
        "quadrant_distribution": {
            quad: int(cnt) for quad, cnt in quadrants.value_counts().items()
        },
        "features": FEATURE_COLS,
    }

    with open(MODELS_DIR / "uplift_metrics.json", "w") as f:
        json.dump(uplift_summary, f, indent=2)

    print(f"Saved model artifact to {MODELS_DIR / 'uplift_models.joblib'}")
    print(f"Saved metrics summary to {MODELS_DIR / 'uplift_metrics.json'}")

    return t_learner, uplift_summary


if __name__ == "__main__":
    train_and_evaluate_uplift()
