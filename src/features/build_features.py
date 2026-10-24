"""Feature Engineering and Leakage-Free Preprocessing Pipeline.

This module:
1. Loads customer records from Supabase PostgreSQL (or local fallback).
2. Engineers high-signal domain features justified by Phase 2 EDA:
   - tenure_cohort (6 duration brackets capturing onboarding attrition hazard)
   - num_services_subscribed (total count of active subscriptions)
   - num_security_services (count of protective stickiness services)
   - is_automatic_payment (indicator for bank transfer / credit card auto-pay)
   - has_fiber_no_support (flag for high-cost fiber optic lacking tech support)
   - charge_per_tenure_month (historical monthly expenditure rate)
   - monthly_to_total_ratio (monthly financial commitment ratio)
3. Strictly prevents data leakage by executing a Stratified 70/15/15 train/val/test split
   BEFORE fitting any encoding or scaling transformations.
4. Serializes the fitted preprocessor pipeline and split datasets.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.database.connection import get_engine
from src.database.schema import Customer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Services columns considered for service count aggregations
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

SECURITY_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
]


def load_dataset_from_db_or_csv() -> pd.DataFrame:
    """Load dataset from Supabase PostgreSQL, falling back to local cleaned CSV if offline."""
    try:
        engine = get_engine()
        query = "SELECT * FROM customers"
        logger.info("Loading records from Supabase Cloud PostgreSQL...")
        df = pd.read_sql(query, con=engine)
        if len(df) == 7043:
            logger.info("Successfully fetched %d records from Supabase.", len(df))
            return df
        logger.warning("Supabase returned %d records; checking local CSV fallback...", len(df))
    except Exception as e:
        logger.warning("Could not load from Supabase (%s); falling back to local CSV...", e)

    csv_path = PROCESSED_DATA_DIR / "telco_churn_cleaned.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {csv_path}")
    logger.info("Loading records from local CSV: %s", csv_path)
    return pd.read_csv(csv_path)


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain-specific features grounded in Phase 2 EDA findings.

    Does not fit any global statistics, making it safe to apply row-wise
    without risk of cross-split data leakage.
    """
    df = df.copy()

    # 1. Tenure Cohorts (Binned categorical feature)
    bins = [-1, 12, 24, 36, 48, 60, 72]
    labels = ["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61-72m"]
    df["tenure_cohort"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)

    # 2. Number of active subscribed services (Count metric 0 - 9)
    # PhoneService == 'Yes' counts as 1. MultipleLines == 'Yes' counts as additional service.
    # InternetService in ['DSL', 'Fiber optic'] counts as 1.
    active_services = (df["InternetService"].isin(["DSL", "Fiber optic"])).astype(int)
    for col in SERVICE_COLUMNS:
        active_services += (df[col] == "Yes").astype(int)
    df["num_services_subscribed"] = active_services

    # 3. Number of protective security / support services (Count metric 0 - 4)
    security_count = pd.Series(0, index=df.index)
    for col in SECURITY_COLUMNS:
        security_count += (df[col] == "Yes").astype(int)
    df["num_security_services"] = security_count

    # 4. Payment automation flag (1 = Automatic, 0 = Manual)
    df["is_automatic_payment"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)

    # 5. High-Risk Combination: Fiber Optic Internet without Tech Support
    df["has_fiber_no_support"] = (
        (df["InternetService"] == "Fiber optic") & (df["TechSupport"] != "Yes")
    ).astype(int)

    # 6. Charge Ratios (Adding +1 to denominator prevents division-by-zero for tenure=0)
    df["charge_per_tenure_month"] = (df["TotalCharges"] / (df["tenure"] + 1)).round(2)
    df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).round(4)

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute stratified 70/15/15 train, validation, and test split.

    Uses stratification on 'churn_label' to preserve the 26.54% churn proportion
    identically across all three subsets.
    """
    # First split off the test set (15%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["churn_label"],
    )

    # From remaining 85%, split out validation set (0.15 / 0.85 ≈ 0.1765)
    adjusted_val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        random_state=random_state,
        stratify=train_val_df["churn_label"],
    )

    logger.info(
        "Data split complete: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        len(train_df),
        len(train_df) / len(df) * 100,
        len(val_df),
        len(val_df) / len(df) * 100,
        len(test_df),
        len(test_df) / len(df) * 100,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def create_preprocessor_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numeric features and one-hot encodes categoricals."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """Define the explicit list of numeric and categorical predictive features.

    Note: 'customerID', 'Churn' (string target), and 'churn_label' (binary target)
    are strictly excluded from feature inputs.
    """
    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "num_services_subscribed",
        "num_security_services",
        "is_automatic_payment",
        "has_fiber_no_support",
        "charge_per_tenure_month",
        "monthly_to_total_ratio",
    ]

    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "tenure_cohort",
    ]

    return numeric_features, categorical_features


def run_feature_pipeline() -> Dict[str, any]:
    """Execute end-to-end feature engineering and leakage-free dataset creation."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df_raw = load_dataset_from_db_or_csv()

    # 2. Engineer domain features
    df_engineered = engineer_domain_features(df_raw)

    # 3. Split data FIRST to strictly prevent data leakage
    train_df, val_df, test_df = split_data(df_engineered)

    # 4. Define features
    numeric_cols, categorical_cols = get_feature_lists()

    # 5. Fit preprocessor ONLY on the training set
    preprocessor = create_preprocessor_pipeline(numeric_cols, categorical_cols)
    preprocessor.fit(train_df[numeric_cols + categorical_cols])

    # 6. Transform train, val, and test splits
    X_train = preprocessor.transform(train_df[numeric_cols + categorical_cols])
    X_val = preprocessor.transform(val_df[numeric_cols + categorical_cols])
    X_test = preprocessor.transform(test_df[numeric_cols + categorical_cols])

    feature_names = preprocessor.get_feature_names_out().tolist()

    # Convert back to clean DataFrames with feature names for interpretability
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    y_train = train_df["churn_label"].values
    y_val = val_df["churn_label"].values
    y_test = test_df["churn_label"].values

    # 7. Save processed feature matrices and target vectors
    X_train_df.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_val_df.to_csv(PROCESSED_DATA_DIR / "X_val.csv", index=False)
    X_test_df.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)

    np.save(PROCESSED_DATA_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DATA_DIR / "y_test.npy", y_test)

    # Also save raw engineered splits with customerID intact for evaluation & dashboard lookups
    train_df.to_csv(PROCESSED_DATA_DIR / "train_unprocessed.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val_unprocessed.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_unprocessed.csv", index=False)

    # 8. Save fitted preprocessor pipeline artifact
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    logger.info("Saved fitted preprocessor to %s", MODELS_DIR / "preprocessor.joblib")

    summary = {
        "train_rows": len(X_train_df),
        "val_rows": len(X_val_df),
        "test_rows": len(X_test_df),
        "feature_count": len(feature_names),
        "features": feature_names,
        "train_churn_rate_pct": round(float(y_train.mean() * 100), 2),
        "val_churn_rate_pct": round(float(y_val.mean() * 100), 2),
        "test_churn_rate_pct": round(float(y_test.mean() * 100), 2),
    }

    return summary


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING & PREPROCESSING PIPELINE")
    print("=" * 60)
    results = run_feature_pipeline()
    for k, v in results.items():
        if k == "features":
            print(f" - {k} ({len(v)} total): {v[:5]} ... {v[-5:]}")
        else:
            print(f" - {k}: {v}")
    print("=" * 60)
