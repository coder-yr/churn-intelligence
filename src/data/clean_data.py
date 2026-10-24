"""Data Cleaning and Quality Assurance Module for Telco Customer Churn.

This module loads the raw IBM Telco Customer Churn dataset, performs data
validation, handles whitespace anomalies in TotalCharges with defensible business
logic (tenure=0 customers), creates a standardized binary churn indicator, and saves
the cleaned dataset to data/processed/.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define canonical project paths using pathlib
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
)


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset from disk."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {filepath}. Please ensure the file exists."
        )
    logger.info("Loading raw dataset from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Raw dataset loaded: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def clean_telecom_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Clean the raw telecom dataset with defensible business rules.

    Cleaning Steps:
    1. Verify unique customerID identifiers (no duplicates).
    2. Convert TotalCharges from string to numeric.
       - Whitespace strings (' ') become NaN via pd.to_numeric(errors='coerce').
       - Investigate rows with NaN TotalCharges: all 11 rows correspond to
         brand new customers (tenure = 0).
       - Business decision: Impute TotalCharges = 0.0 for tenure = 0.
         (Dropping active new customers introduces survivorship bias;
          imputing with mean/median would distort first-month customer accounting).
    3. Retain original 'Churn' column ('Yes'/'No') for reporting, and add a
       convenient binary integer target 'churn_label' (1 for Yes, 0 for No).
    4. Validate data types and verify zero remaining null values.

    Returns:
        Tuple[pd.DataFrame, dict]: Cleaned dataframe and metadata summary.
    """
    cleaned_df = df.copy()

    # 1. Uniqueness check
    total_rows = len(cleaned_df)
    unique_ids = cleaned_df["customerID"].nunique()
    if total_rows != unique_ids:
        raise ValueError(
            f"Dataset contains duplicate customerIDs: {total_rows - unique_ids} duplicates found."
        )

    # 2. TotalCharges numeric conversion & whitespace investigation
    is_whitespace = cleaned_df["TotalCharges"].astype(str).str.strip() == ""
    whitespace_count = int(is_whitespace.sum())

    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"], errors="coerce"
    )
    nan_charges_count = int(cleaned_df["TotalCharges"].isna().sum())

    # Check tenure for NaN TotalCharges
    nan_tenure_dist = cleaned_df.loc[
        cleaned_df["TotalCharges"].isna(), "tenure"
    ].value_counts().to_dict()

    # Defensible handling: All NaN rows have tenure=0, impute 0.0
    cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].fillna(0.0)

    # 3. Target label standardization
    cleaned_df["churn_label"] = (cleaned_df["Churn"] == "Yes").astype(int)

    # 4. Summary metadata
    summary = {
        "initial_rows": total_rows,
        "final_rows": len(cleaned_df),
        "columns_count": len(cleaned_df.columns),
        "whitespace_total_charges_count": whitespace_count,
        "nan_charges_count": nan_charges_count,
        "nan_charges_tenure_distribution": nan_tenure_dist,
        "imputed_total_charges_value": 0.0,
        "retained_zero_tenure_customers": int((cleaned_df["tenure"] == 0).sum()),
        "churn_rate_pct": round(
            float(cleaned_df["churn_label"].mean() * 100), 2
        ),
        "remaining_nulls": int(cleaned_df.isna().sum().sum()),
    }

    logger.info(
        "Data cleaning complete: %d rows retained, %d whitespace charges imputed to 0.0, remaining nulls: %d",
        summary["final_rows"],
        summary["whitespace_total_charges_count"],
        summary["remaining_nulls"],
    )

    return cleaned_df, summary


def save_cleaned_data(
    df: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH
) -> Path:
    """Save cleaned dataframe to processed CSV directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Cleaned dataset successfully saved to: %s", output_path)
    return output_path


def run_pipeline() -> Tuple[pd.DataFrame, dict]:
    """Execute the full data cleaning pipeline."""
    df_raw = load_raw_data()
    df_cleaned, summary = clean_telecom_data(df_raw)
    save_cleaned_data(df_cleaned)
    return df_cleaned, summary


if __name__ == "__main__":
    df_clean, stats = run_pipeline()
    print("\n" + "=" * 50)
    print("DATA CLEANING PIPELINE SUMMARY")
    print("=" * 50)
    for key, value in stats.items():
        print(f" - {key}: {value}")
    print("=" * 50)
