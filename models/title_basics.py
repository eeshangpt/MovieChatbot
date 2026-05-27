"""
Database model for title_basics tables.
Author: Eeshan Gupta
"""

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class TitleBasics(Base):
    __tablename__ = "title_basics"

    tconst: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
        index=True,
    )

    title_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    primary_title: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    original_title: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    is_adult: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    start_year: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    end_year: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    runtime_minutes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    genres: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
    )
