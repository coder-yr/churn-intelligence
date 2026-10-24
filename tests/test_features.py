"""Unit tests for Feature Engineering and Preprocessing Pipeline."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    engineer_domain_features,
    get_feature_lists,
    split_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


@pytest.fixture
def sample_customer_df() -> pd.DataFrame:
    """Provide synthetic customer records covering edge cases (e.g. tenure=0)."""
    return pd.DataFrame(
        {
            "customerID": ["C001", "C002", "C003", "C004"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [0, 12, 36, 72],
            "PhoneService": ["Yes", "Yes", "No", "Yes"],
            "MultipleLines": ["No", "Yes", "No phone service", "Yes"],
            "InternetService": ["Fiber optic", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "Yes", "No", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No", "No internet service"],
            "DeviceProtection": ["No", "No", "Yes", "No internet service"],
            "TechSupport": ["No", "No", "Yes", "No internet service"],
            "StreamingTV": ["Yes", "No", "No", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No", "No internet service"],
            "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Mailed check",
            ],
            "MonthlyCharges": [85.50, 95.00, 45.00, 20.00],
            "TotalCharges": [0.0, 1140.0, 1620.0, 1440.0],
            "Churn": ["No", "Yes", "No", "No"],
            "churn_label": [0, 1, 0, 0],
        }
    )


def test_engineer_domain_features(sample_customer_df):
    """Verify that all domain features are calculated accurately."""
    df_feat = engineer_domain_features(sample_customer_df)

    # 1. Tenure cohort check
    assert df_feat.loc[0, "tenure_cohort"] == "0-12m"
    assert df_feat.loc[1, "tenure_cohort"] == "0-12m"
    assert df_feat.loc[2, "tenure_cohort"] == "25-36m"
    assert df_feat.loc[3, "tenure_cohort"] == "61-72m"

    # 2. Payment automation check
    assert df_feat.loc[0, "is_automatic_payment"] == 0  # Electronic check
    assert df_feat.loc[1, "is_automatic_payment"] == 1  # Bank transfer (automatic)
    assert df_feat.loc[2, "is_automatic_payment"] == 1  # Credit card (automatic)
    assert df_feat.loc[3, "is_automatic_payment"] == 0  # Mailed check

    # 3. High-risk flag: Fiber optic without Tech Support
    assert df_feat.loc[0, "has_fiber_no_support"] == 1  # Fiber + TechSupport='No'
    assert df_feat.loc[1, "has_fiber_no_support"] == 1  # Fiber + TechSupport='No'
    assert df_feat.loc[2, "has_fiber_no_support"] == 0  # DSL + TechSupport='Yes'
    assert df_feat.loc[3, "has_fiber_no_support"] == 0  # No Internet

    # 4. Zero division safety for tenure=0
    assert df_feat.loc[0, "charge_per_tenure_month"] == 0.0
    assert not np.isnan(df_feat.loc[0, "monthly_to_total_ratio"])


def test_split_data_leakage_free():
    """Verify that train, validation, and test sets are completely disjoint."""
    csv_path = PROCESSED_DATA_DIR / "telco_churn_cleaned.csv"
    assert csv_path.exists(), "Cleaned CSV must exist."
    df = pd.read_csv(csv_path)

    train_df, val_df, test_df = split_data(df, test_size=0.15, val_size=0.15, random_state=42)

    # Total row counts
    assert len(train_df) + len(val_df) + len(test_df) == 7043

    # Zero overlap in customer IDs (Guarantees isolation)
    train_ids = set(train_df["customerID"])
    val_ids = set(val_df["customerID"])
    test_ids = set(test_df["customerID"])

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # Churn stratification preserved within 0.5% margin
    assert abs(train_df["churn_label"].mean() - 0.2654) < 0.01
    assert abs(val_df["churn_label"].mean() - 0.2654) < 0.01
    assert abs(test_df["churn_label"].mean() - 0.2654) < 0.01


def test_preprocessor_artifact_and_features():
    """Verify that the serialized ColumnTransformer artifact transforms data into 41 features."""
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    assert preprocessor_path.exists(), "Fitted preprocessor artifact must exist."

    preprocessor = joblib.load(preprocessor_path)
    feature_names = preprocessor.get_feature_names_out()

    assert len(feature_names) == 41

    # Check generated feature matrices
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")

    assert X_train.shape == (4929, 41)
    assert X_val.shape == (1057, 41)
    assert X_test.shape == (1057, 41)

    assert X_train.isna().sum().sum() == 0
    assert X_val.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0
