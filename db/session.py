import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ======================================================
# DATABASE CONFIG
# ======================================================

load_dotenv(".env.dev")
DATABASE_URL = os.getenv("DATABASE_URL", "")


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
