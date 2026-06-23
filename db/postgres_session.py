import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ======================================================
# DATABASE CONFIG
# ======================================================

load_dotenv(find_dotenv(".env.dev"))

POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
DATABASE_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# ======================================================
# SQLALCHEMY ENGINE
# ======================================================

engine = create_engine(
    DATABASE_URL,
    # Future SQLAlchemy 2.x style
    future=True,
    # Enables connection pool health checks
    pool_pre_ping=True,
    # Connection pool size
    pool_size=10,
    # Extra temporary connections
    max_overflow=20,
    # Print SQL queries (disable later)
    echo=False,
)

# ======================================================
# SESSION FACTORY
# ======================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)

__all__ = ["SessionLocal"]
