"""Unit tests for database validation, schema definition, and safe connection utilities.

All tests run locally in isolation without requiring active credentials or a live database.
"""

from pathlib import Path
import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect

from src.database.connection import get_database_url, mask_database_url
from src.database.load_data import (
    EXPECTED_COLUMNS,
    load_cleaned_records,
    validate_dataframe_for_ingestion,
)
from src.database.schema import Base, Customer, create_tables

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"


# ==============================================================================
# 1. Connection & Credential Masking Tests
# ==============================================================================

def test_mask_database_url():
    """Verify that credentials are never exposed in logs."""
    raw_url = "postgresql+psycopg://postgres.project123:SecretPassword999@aws-0-region.pooler.supabase.com:5432/postgres"
    masked = mask_database_url(raw_url)

    assert "SecretPassword999" not in masked
    assert "postgres.project123:*****@" in masked
    assert "aws-0-region.pooler.supabase.com:5432/postgres" in masked


def test_mask_database_url_empty():
    """Verify handling of empty or None URLs."""
    assert mask_database_url("") == "<empty>"


def test_get_database_url_placeholder_detection(monkeypatch):
    """Verify that unconfigured template placeholders raise an informative error."""
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres.[YOUR-PROJECT-REF]:[YOUR-PASSWORD]@host:5432/db",
    )
    with pytest.raises(ValueError, match="contains unconfigured template placeholders"):
        get_database_url()


def test_get_database_url_scheme_normalization(monkeypatch):
    """Verify that postgresql:// is automatically upgraded to postgresql+psycopg://."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
    url = get_database_url()
    assert url == "postgresql+psycopg://user:pass@host:5432/db"


def test_get_database_url_invalid_scheme(monkeypatch):
    """Verify that unsupported database schemes are rejected."""
    monkeypatch.setenv("DATABASE_URL", "mysql://user:pass@host:3306/db")
    with pytest.raises(ValueError, match="Unsupported database scheme"):
        get_database_url()


# ==============================================================================
# 2. DataFrame Validation Logic Tests
# ==============================================================================

def test_validate_actual_processed_csv():
    """Validate that the real cleaned CSV meets all 100% data integrity criteria."""
    assert PROCESSED_CSV.exists(), "Processed CSV file must exist."
    df = load_cleaned_records(PROCESSED_CSV)
    result = validate_dataframe_for_ingestion(df)

    assert result["valid"] is True
    assert result["total_records"] == 7043
    assert result["unique_customer_ids"] == 7043
    assert result["churned_count"] == 1869
    assert result["retained_count"] == 5174


def test_validate_missing_required_column():
    """Verify that dropping a required column triggers a descriptive error."""
    df = pd.DataFrame({
        "customerID": ["001"],
        "gender": ["Female"],
    })
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe_for_ingestion(df)


def test_validate_duplicate_customer_ids():
    """Verify that duplicate customerIDs are caught immediately."""
    df = load_cleaned_records(PROCESSED_CSV).head(10).copy()
    # Introduce duplicate
    df.loc[1, "customerID"] = df.loc[0, "customerID"]

    with pytest.raises(ValueError, match="duplicate customerIDs detected"):
        validate_dataframe_for_ingestion(df)


def test_validate_unexpected_nulls():
    """Verify that null values in required features are caught."""
    df = load_cleaned_records(PROCESSED_CSV).head(10).copy()
    df.loc[2, "MonthlyCharges"] = None

    with pytest.raises(ValueError, match="null values found in columns"):
        validate_dataframe_for_ingestion(df)


def test_validate_invalid_churn_label():
    """Verify that non-binary churn labels (e.g. 2, -1) trigger validation errors."""
    df = load_cleaned_records(PROCESSED_CSV).head(10).copy()
    df.loc[3, "churn_label"] = 99

    with pytest.raises(ValueError, match="Invalid churn_label values"):
        validate_dataframe_for_ingestion(df)


def test_validate_negative_numeric_values():
    """Verify that negative tenure or charges are rejected."""
    df = load_cleaned_records(PROCESSED_CSV).head(10).copy()
    df.loc[0, "tenure"] = -5
    with pytest.raises(ValueError, match="negative tenure values"):
        validate_dataframe_for_ingestion(df)

    df2 = load_cleaned_records(PROCESSED_CSV).head(10).copy()
    df2.loc[0, "MonthlyCharges"] = -10.5
    with pytest.raises(ValueError, match="negative MonthlyCharges values"):
        validate_dataframe_for_ingestion(df2)


# ==============================================================================
# 3. Schema & ORM Model Tests (In-Memory Engine)
# ==============================================================================

def test_customer_model_columns():
    """Verify that Customer ORM model matches all expected feature columns."""
    column_names = {col.name for col in Customer.__table__.columns}
    assert column_names == set(EXPECTED_COLUMNS)
    assert Customer.__table__.primary_key.columns.keys() == ["customerID"]


def test_schema_creation_in_memory():
    """Verify that tables and constraints create cleanly on an in-memory SQL engine."""
    engine = create_engine("sqlite:///:memory:")
    create_tables(engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "customers" in tables

    columns = [col["name"] for col in inspector.get_columns("customers")]
    assert len(columns) == 22
    assert "customerID" in columns
    assert "churn_label" in columns
