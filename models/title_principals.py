"""
Database model for title_principals table
Author: Eeshan Gupta
"""

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class TitlePrincipals(Base):
    __tablename__ = "title_principals"

    tconst: Mapped[str] = mapped_column(
        ForeignKey("title_basics.tconst"),
        primary_key=True,
    )

    ordering: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    nconst: Mapped[str | None] = mapped_column(
        ForeignKey("name_basics.nconst"),
        nullable=True,
        index=True,
    )

    category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    job: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    characters: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )
