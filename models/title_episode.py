"""
Database model for title_episode table.
Author: Eeshan Gupta
"""

from sqlalchemy import ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class TitleEpisode(Base):
    __tablename__ = "title_episode"

    tconst: Mapped[str] = mapped_column(
        ForeignKey("title_basics.tconst"),
        primary_key=True,
    )

    parent_tconst: Mapped[str | None] = mapped_column(
        ForeignKey("title_basics.tconst"),
        nullable=True,
    )

    season_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    episode_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
