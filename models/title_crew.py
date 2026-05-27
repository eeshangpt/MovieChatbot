"""
Database model for the title_crew table.
Author: Eeshan Gupta
"""

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class TitleCrew(Base):
    __tablename__ = "title_crew"

    tconst: Mapped[str] = mapped_column(
        ForeignKey("title_basics.tconst"),
        primary_key=True,
    )

    directors: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
    )

    writers: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
    )
