from db.postgres_session import engine
from models.base import Base
from models.name_basics import NameBasics
from models.title_akas import TitleAkas
from models.title_basics import TitleBasics
from models.title_crew import TitleCrew
from models.title_episode import TitleEpisode
from models.title_principals import TitlePrincipals
from models.title_ratings import TitleRatings

Base.metadata.create_all(bind=engine)

print("All tables created")
