"""Database Schema Definition for Supabase PostgreSQL.

Defines the `customers` table using SQLAlchemy 2.0 DeclarativeBase.
All 21 original Telco features plus the standardized `churn_label` binary
indicator are modeled with strongly typed columns and constraints.
"""

import logging
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    Float,
    Index,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy 2.0 declarative models."""
    pass


class Customer(Base):
    """Customer account record in the `customers` table.

    Key Schema Decisions:
    1. Primary Key: `customerID` (e.g. '7590-VHVEG') uniquely identifies every account.
    2. Types:
       - `SeniorCitizen`: Stored as Integer (0 or 1) consistent with raw data.
       - `tenure`: Non-negative integer representing active months.
       - `MonthlyCharges` and `TotalCharges`: Float/Numeric(10, 2) storing financial amounts.
       - `Churn`: Original string ('Yes' / 'No') for natural reporting and dashboards.
       - `churn_label`: Binary integer (1 for Yes, 0 for No) for fast ML and aggregation.
    3. Constraints & Indexes:
       - Primary Key on `customerID`.
       - Index on `churn_label` and `Contract` to accelerate filtering in Streamlit.
    """

    __tablename__ = "customers"

    # Primary Key Identifier
    customerID: Mapped[str] = mapped_column(
        String(32), primary_key=True, index=True, doc="Unique customer identifier"
    )

    # Demographics
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    SeniorCitizen: Mapped[int] = mapped_column(Integer, nullable=False)
    Partner: Mapped[str] = mapped_column(String(8), nullable=False)
    Dependents: Mapped[str] = mapped_column(String(8), nullable=False)

    # Account & Service Tenure
    tenure: Mapped[int] = mapped_column(Integer, nullable=False)

    # Phone Services
    PhoneService: Mapped[str] = mapped_column(String(8), nullable=False)
    MultipleLines: Mapped[str] = mapped_column(String(32), nullable=False)

    # Internet & Add-on Services
    InternetService: Mapped[str] = mapped_column(String(32), nullable=False)
    OnlineSecurity: Mapped[str] = mapped_column(String(32), nullable=False)
    OnlineBackup: Mapped[str] = mapped_column(String(32), nullable=False)
    DeviceProtection: Mapped[str] = mapped_column(String(32), nullable=False)
    TechSupport: Mapped[str] = mapped_column(String(32), nullable=False)
    StreamingTV: Mapped[str] = mapped_column(String(32), nullable=False)
    StreamingMovies: Mapped[str] = mapped_column(String(32), nullable=False)

    # Contract, Billing & Payment
    Contract: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    PaperlessBilling: Mapped[str] = mapped_column(String(8), nullable=False)
    PaymentMethod: Mapped[str] = mapped_column(String(64), nullable=False)
    MonthlyCharges: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    TotalCharges: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)

    # Target Outcomes
    Churn: Mapped[str] = mapped_column(String(8), nullable=False)
    churn_label: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    __table_args__ = (
        CheckConstraint("tenure >= 0", name="chk_tenure_non_negative"),
        CheckConstraint('"MonthlyCharges" >= 0', name="chk_monthly_charges_positive"),
        CheckConstraint('"TotalCharges" >= 0', name="chk_total_charges_positive"),
        CheckConstraint("churn_label IN (0, 1)", name="chk_churn_label_binary"),
    )

    def __repr__(self) -> str:
        return (
            f"<Customer(id='{self.customerID}', contract='{self.Contract}', "
            f"tenure={self.tenure}, churn='{self.Churn}')>"
        )


def create_tables(engine: Engine) -> None:
    """Create all declarative tables in the connected database if they do not exist."""
    logger.info("Creating database tables if not present...")
    Base.metadata.create_all(bind=engine)
    logger.info("Table creation check completed.")


def drop_tables(engine: Engine) -> None:
    """Drop all tables defined in metadata (CAUTION: destructive action)."""
    logger.warning("Dropping all database tables defined in Base metadata...")
    Base.metadata.drop_all(bind=engine)
    logger.warning("Table drop completed.")


if __name__ == "__main__":
    from src.database.connection import get_engine
    print("Compiling schema and table definitions...")
    for col in Customer.__table__.columns:
        print(f" - {col.name:20s} : {col.type}")
    print(f"\nTotal columns in 'customers' table: {len(Customer.__table__.columns)}")
