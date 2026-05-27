"""
Database model for the name_basic table.
Author: Eeshan Gupta
"""

from sqlalchemy import Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class NameBasics(Base):
    __tablename__ = "name_basics"

    nconst: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
        index=True,
    )

    primary_name: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    birth_year: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    death_year: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    primary_profession: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
    )

    known_for_titles: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
    )
