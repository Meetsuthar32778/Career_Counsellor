"""
database.py - SQLAlchemy Database Configuration
================================================
Sets up the SQLite database engine, session factory,
and provides a base class for all ORM models.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# ---------------------------------------------------------------------------
# Database URL: SQLite file stored at the project root
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'career_bot.db')}"

# ---------------------------------------------------------------------------
# SQLAlchemy Engine & Session
# ---------------------------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite
    echo=False,  # Set True for SQL query logging during development
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ---------------------------------------------------------------------------
# Base class for all ORM models
# ---------------------------------------------------------------------------
Base = declarative_base()


def get_db():
    """
    Dependency function for FastAPI.
    Yields a database session and ensures it is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Create all database tables if they do not exist.
    Called once at application startup.
    """
    Base.metadata.create_all(bind=engine)
