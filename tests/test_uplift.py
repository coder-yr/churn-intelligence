"""
Unit tests for Uplift Modeling, T-Learner, Qini Curve, and Quadrant Segmentation.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.retention.uplift_model import (
    TwoModelTLearner,
    compute_qini_curve,
    compute_decile_uplift,
    classify_uplift_quadrants,
    FEATURE_COLS,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


@pytest.fixture
def synthetic_uplift_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "recency": np.random.randint(1, 12, size=n),
        "history": np.random.uniform(50, 800, size=n),
        "mens": np.random.choice([0, 1], size=n),
        "womens": np.random.choice([0, 1], size=n),
        "newbie": np.random.choice([0, 1], size=n),
        "zip_Suburban": np.random.choice([0, 1], size=n),
        "zip_Urban": np.random.choice([0, 1], size=n),
        "channel_Phone": np.random.choice([0, 1], size=n),
        "channel_Web": np.random.choice([0, 1], size=n),
    })
    treatment = np.random.choice([0, 1], size=n, p=[0.33, 0.67])
    # Uplift exists: treated have higher response probability
    p_response = np.where(treatment == 1, 0.25, 0.10)
    y = np.random.binomial(1, p_response)
    return df, pd.Series(treatment), pd.Series(y)


def test_t_learner_fit_and_predict(synthetic_uplift_data):
    X, treatment, y = synthetic_uplift_data
    learner = TwoModelTLearner(lgb_params={"n_estimators": 10, "random_state": 42, "verbose": -1})
    learner.fit(X, treatment, y)

    assert learner.is_fitted is True

    mu_0, mu_1 = learner.predict_counterfactuals(X)
    assert len(mu_0) == len(X)
    assert len(mu_1) == len(X)
    assert np.all((mu_0 >= 0.0) & (mu_0 <= 1.0))
    assert np.all((mu_1 >= 0.0) & (mu_1 <= 1.0))

    uplift = learner.predict_uplift(X)
    assert len(uplift) == len(X)
    # Uplift is strictly mu_1 - mu_0
    np.testing.assert_allclose(uplift, mu_1 - mu_0, atol=1e-6)


def test_qini_curve_properties():
    np.random.seed(42)
    n = 500
    y = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    treatment = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    uplift_preds = np.random.normal(0.05, 0.02, size=n)

    qini = compute_qini_curve(y, treatment, uplift_preds, num_bins=20)

    assert "fractions" in qini
    assert "qini_values" in qini
    assert "random_qini_values" in qini
    assert "qini_score" in qini
    assert len(qini["fractions"]) == 21
    assert qini["fractions"][0] == 0.0
    assert pytest.approx(qini["fractions"][-1], abs=1e-5) == 1.0


def test_compute_decile_uplift():
    np.random.seed(42)
    n = 300
    y = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    treatment = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    uplift_preds = np.linspace(0.20, -0.05, n)

    deciles = compute_decile_uplift(y, treatment, uplift_preds, num_deciles=5)

    assert len(deciles) == 5
    assert set(deciles["decile"]) == {1, 2, 3, 4, 5}
    assert (deciles["size"] > 0).all()
    assert (deciles["treated_rate"] >= 0.0).all()
    assert (deciles["control_rate"] >= 0.0).all()


def test_classify_uplift_quadrants():
    mu_0 = np.array([0.05, 0.25, 0.02, 0.30])
    mu_1 = np.array([0.15, 0.27, 0.03, 0.20])
    uplift = mu_1 - mu_0  # [+0.10, +0.02, +0.01, -0.10]

    quadrants = classify_uplift_quadrants(
        mu_0, mu_1, uplift, uplift_threshold=0.05, baseline_threshold=0.15
    )

    assert len(quadrants) == 4
    assert quadrants[0] == "Persuadables"  # uplift >= 0.05
    assert quadrants[1] == "Sure Things"    # baseline >= 0.15, uplift < 0.05
    assert quadrants[2] == "Lost Causes"    # baseline < 0.15, uplift < 0.05
    assert quadrants[3] == "Sleeping Dogs"  # uplift < 0.0


def test_uplift_artifacts_exist_and_valid():
    model_path = MODELS_DIR / "uplift_models.joblib"
    metrics_path = MODELS_DIR / "uplift_metrics.json"

    assert model_path.exists(), f"Missing model: {model_path}"
    assert metrics_path.exists(), f"Missing metrics: {metrics_path}"

    # Load model
    learner = TwoModelTLearner.load(model_path)
    assert isinstance(learner, TwoModelTLearner)
    assert learner.is_fitted is True

    # Validate metrics JSON
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "model_performance" in metrics
    assert "qini_score" in metrics["model_performance"]
    assert "deciles" in metrics
    assert len(metrics["deciles"]) == 10
    assert "quadrant_distribution" in metrics
    assert "Persuadables" in metrics["quadrant_distribution"]

    # Verify figures
    assert (FIGURES_DIR / "23_uplift_qini_curve.png").exists()
    assert (FIGURES_DIR / "24_uplift_cumulative_gain.png").exists()
    assert (FIGURES_DIR / "25_uplift_quadrants.png").exists()
