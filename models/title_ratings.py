"""
Database model for title_ratings table.
Author: Eeshan Gupta
"""

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class TitleRatings(Base):
    __tablename__ = "title_ratings"

    tconst: Mapped[str] = mapped_column(
        ForeignKey("title_basics.tconst"),
        primary_key=True,
    )

    average_rating: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    num_votes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
