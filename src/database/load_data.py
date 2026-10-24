"""Database Ingestion Pipeline for Supabase PostgreSQL.

Loads cleaned customer records from `data/processed/telco_churn_cleaned.csv`
into the Supabase `customers` table.

Guarantees:
- Validates data integrity before opening a database transaction.
- Idempotent: Uses PostgreSQL `ON CONFLICT (customerID) DO UPDATE` so rerunning
  never duplicates records.
- Batch processing: Ingests in chunks for optimal network performance.
- Safe rollback: Any failure cleanly rolls back the transaction.
- Zero credential leakage in output or logs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from src.database.connection import get_engine, mask_database_url
from src.database.schema import Customer, create_tables

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
)

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
    "churn_label",
]


def validate_dataframe_for_ingestion(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate dataframe structure, data types, and values prior to database loading.

    Raises:
        ValueError: If required columns are missing, duplicate IDs exist,
                    unexpected nulls are present, or invalid labels are found.
    """
    # 1. Column existence
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset missing required columns for ingestion: {sorted(missing_cols)}"
        )

    # 2. Duplicate customerID check
    total_records = len(df)
    unique_ids = df["customerID"].nunique()
    if total_records != unique_ids:
        raise ValueError(
            f"Integrity check failed: {total_records - unique_ids} duplicate customerIDs detected."
        )

    # 3. Null values check
    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    if null_counts.sum() > 0:
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Integrity check failed: null values found in columns: {bad_cols}")

    # 4. Valid churn labels
    invalid_labels = set(df["churn_label"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(
            f"Invalid churn_label values found: {invalid_labels}. Expected only {0, 1}."
        )

    # 5. Non-negative numeric checks
    if (df["tenure"] < 0).any():
        raise ValueError("Integrity check failed: negative tenure values found.")
    if (df["MonthlyCharges"] < 0).any():
        raise ValueError("Integrity check failed: negative MonthlyCharges values found.")
    if (df["TotalCharges"] < 0).any():
        raise ValueError("Integrity check failed: negative TotalCharges values found.")

    return {
        "valid": True,
        "total_records": total_records,
        "unique_customer_ids": unique_ids,
        "churned_count": int((df["churn_label"] == 1).sum()),
        "retained_count": int((df["churn_label"] == 0).sum()),
    }


def load_cleaned_records(csv_path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load cleaned CSV file from disk."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {csv_path}. Please run clean_data.py first."
        )
    return pd.read_csv(csv_path)


def ingest_records(
    df: pd.DataFrame,
    engine: Engine,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """Ingest records into the PostgreSQL `customers` table using idempotent upsert.

    Uses `ON CONFLICT (customerID) DO UPDATE` so that running ingestion repeatedly
    refreshes records in place without creating duplicates or failing on primary key conflicts.
    """
    logger.info("Starting ingestion of %d customer records...", len(df))

    # Ensure tables are created
    create_tables(engine)

    records = df[EXPECTED_COLUMNS].to_dict(orient="records")
    total_records = len(records)
    inserted_count = 0

    with engine.begin() as conn:
        for i in range(0, total_records, batch_size):
            batch = records[i : i + batch_size]
            stmt = pg_insert(Customer).values(batch)

            # Idempotent upsert: update non-key fields if primary key already exists
            update_dict = {
                col.name: stmt.excluded[col.name]
                for col in Customer.__table__.columns
                if not col.primary_key
            }
            upsert_stmt = stmt.on_conflict_do_update(
                index_elements=["customerID"],
                set_=update_dict,
            )

            conn.execute(upsert_stmt)
            inserted_count += len(batch)
            logger.info("Ingested %d / %d records...", inserted_count, total_records)

    logger.info("Successfully completed ingestion of %d records.", inserted_count)
    return {
        "ingested_records": inserted_count,
        "status": "success",
    }


def verify_database_records(engine: Engine) -> Dict[str, Any]:
    """Execute SQL verification queries against the database to confirm data accuracy."""
    logger.info("Running database verification queries...")
    with engine.connect() as conn:
        # Total rows
        total_rows = conn.scalar(select(func.count()).select_from(Customer))

        # Distinct customer IDs
        distinct_ids = conn.scalar(
            select(func.count(func.distinct(Customer.customerID)))
        )

        # Churned customers
        churned_count = conn.scalar(
            select(func.count())
            .select_from(Customer)
            .where(Customer.churn_label == 1)
        )

        # Retained customers
        retained_count = conn.scalar(
            select(func.count())
            .select_from(Customer)
            .where(Customer.churn_label == 0)
        )

        # Null TotalCharges count
        null_total_charges = conn.scalar(
            select(func.count())
            .select_from(Customer)
            .where(Customer.TotalCharges.is_(None))
        )

    verification = {
        "database_total_customers": total_rows,
        "database_unique_ids": distinct_ids,
        "database_churned_customers": churned_count,
        "database_retained_customers": retained_count,
        "database_null_total_charges": null_total_charges,
        "matches_expected_benchmarks": (
            total_rows == 7043
            and distinct_ids == 7043
            and churned_count == 1869
            and retained_count == 5174
            and null_total_charges == 0
        ),
    }

    logger.info("Database verification results: %s", verification)
    return verification


def run_ingestion_pipeline() -> Dict[str, Any]:
    """Execute validation, connection, table creation, ingestion, and verification."""
    # 1. Load and validate locally
    df = load_cleaned_records()
    validation = validate_dataframe_for_ingestion(df)
    logger.info("Pre-ingestion validation passed: %d records ready.", validation["total_records"])

    # 2. Acquire database engine
    engine = get_engine()

    # 3. Ingest records
    ingest_result = ingest_records(df, engine)

    # 4. Verify in database
    db_verification = verify_database_records(engine)

    return {
        "validation": validation,
        "ingestion": ingest_result,
        "database_verification": db_verification,
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING SUPABASE POSTGRESQL INGESTION PIPELINE")
    print("=" * 60)
    try:
        results = run_ingestion_pipeline()
        for section, stats in results.items():
            print(f"\n[{section.upper()}]")
            for k, v in stats.items():
                print(f" - {k}: {v}")
    except Exception as e:
        logger.error("Ingestion halted: %s", e)
    print("=" * 60)
