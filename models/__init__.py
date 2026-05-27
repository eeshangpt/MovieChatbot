"""
Database Model Module for IMDB Database
Author: Eeshan Gupta
"""

from .base import Base
from .name_basics import NameBasics
from .title_akas import TitleAkas
from .title_basics import TitleBasics
from .title_crew import TitleCrew
from .title_episode import TitleEpisode
from .title_principals import TitlePrincipals
from .title_ratings import TitleRatings

__all__ = [
    "Base",
    "NameBasics",
    "TitleAkas",
    "TitleBasics",
    "TitleCrew",
    "TitleEpisode",
    "TitlePrincipals",
    "TitleRatings",
]
