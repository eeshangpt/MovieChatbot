"""
Database model for the title_akas table.
Author: Eeshan Gupta
"""

from sqlalchemy import Boolean, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class TitleAkas(Base):
    __tablename__ = "title_akas"

    title_id: Mapped[str] = mapped_column(
        ForeignKey("title_basics.tconst"),
        primary_key=True,
    )

    ordering: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    title: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    region: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )
    country_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    language: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    types: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
    )

    attributes: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
    )

    is_original_title: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )
