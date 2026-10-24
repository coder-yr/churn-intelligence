"""Database Connection Module for Supabase Cloud PostgreSQL.

Provides secure, production-grade SQLAlchemy 2.x engine and session utilities.
Credentials are read exclusively from environment variables (.env).
Connection URLs are sanitized in all logs and error messages to prevent credential leaks.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load .env file from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()

# Global cached engine instance
_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[sessionmaker] = None


def mask_database_url(url: str) -> str:
    """Mask password and user details in database URL for safe logging.

    Example:
        'postgresql+psycopg://postgres.abc:mypass123@host:5432/db' ->
        'postgresql+psycopg://postgres.abc:*****@host:5432/db'
    """
    if not url:
        return "<empty>"
    # Regex masks anything between : and @ in user:password@ segment
    return re.sub(r":([^@/]+)@", r":*****@", url)


def get_database_url() -> str:
    """Retrieve and validate DATABASE_URL from environment variables.

    Normalizes 'postgresql://' to 'postgresql+psycopg://' to ensure
    the modern psycopg v3 driver is utilized with SQLAlchemy 2.x.
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise ValueError(
            "DATABASE_URL environment variable is missing. "
            "Please configure your Supabase connection string in .env file. "
            "See .env.example for the required format."
        )

    # Check for unreplaced template placeholders
    if "[YOUR-PASSWORD]" in url or "[YOUR-PROJECT-REF]" in url:
        raise ValueError(
            "DATABASE_URL contains unconfigured template placeholders "
            "([YOUR-PASSWORD] or [YOUR-PROJECT-REF]). "
            "Please edit .env with your actual Supabase credentials."
        )

    # Ensure psycopg (v3) driver is specified for modern SQLAlchemy 2.x
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)

    if not url.startswith("postgresql+psycopg://"):
        raise ValueError(
            f"Unsupported database scheme: '{url.split('://')[0]}'. "
            "Expected 'postgresql+psycopg://' or 'postgresql://'."
        )

    return url


def get_engine(database_url: Optional[str] = None) -> Engine:
    """Get or create a cached SQLAlchemy Engine.

    Configured with:
    - pool_pre_ping=True: Tests connections before checkout (handles dropped/idle sockets).
    - pool_recycle=300: Recycles connections every 5 minutes to prevent stale pooler disconnects.
    - pool_size=5, max_overflow=10: Conservative sizing appropriate for Supabase session pooler.
    """
    global _ENGINE, _SESSION_FACTORY

    # If an explicit database URL is passed (e.g. for testing/mocking), create a dedicated engine
    if database_url:
        return create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=300,
        )

    if _ENGINE is None:
        url = get_database_url()
        safe_url = mask_database_url(url)
        logger.info("Initializing SQLAlchemy engine for: %s", safe_url)

        _ENGINE = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=300,
        )
        _SESSION_FACTORY = sessionmaker(bind=_ENGINE)

    return _ENGINE


def get_session(database_url: Optional[str] = None) -> Session:
    """Provide a new SQLAlchemy Session from the session factory."""
    global _SESSION_FACTORY
    if database_url:
        engine = get_engine(database_url)
        return sessionmaker(bind=engine)()

    if _SESSION_FACTORY is None:
        get_engine()
    return _SESSION_FACTORY()


def test_connection(database_url: Optional[str] = None) -> dict:
    """Test connection to PostgreSQL and return server metadata without leaking credentials.

    Returns:
        dict: Connection status, database version, and latency information.
    """
    try:
        engine = get_engine(database_url)
        with engine.connect() as conn:
            # Query PostgreSQL version and current database
            result = conn.execute(
                text("SELECT version(), current_database(), current_user;")
            ).fetchone()

            version_str = result[0] if result else "Unknown"
            db_name = result[1] if result else "Unknown"
            db_user = result[2] if result else "Unknown"

            status = {
                "connected": True,
                "database": db_name,
                "user": db_user,
                "postgres_version": version_str.split(",")[0],
                "message": "Successfully connected to PostgreSQL database.",
            }
            logger.info("Connection test PASSED: %s (User: %s)", db_name, db_user)
            return status

    except Exception as e:
        error_msg = str(e)
        # Sanitize error message to ensure no raw passwords leak
        sanitized_msg = re.sub(r":([^@/\s]+)@", r":*****@", error_msg)
        logger.error("Connection test FAILED: %s", sanitized_msg)
        return {
            "connected": False,
            "error": sanitized_msg,
            "message": (
                "Unable to connect to Supabase PostgreSQL. "
                "Please verify your credentials in .env, check that your Supabase "
                "project is active, and ensure your network can reach Supabase."
            ),
        }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SUPABASE POSTGRESQL CONNECTION TEST")
    print("=" * 60)
    result = test_connection()
    for k, v in result.items():
        print(f" - {k}: {v}")
    print("=" * 60)
