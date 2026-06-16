SCHEMA = """
PostgreSQL database schema (IMDB data):

Table: title_basics
  tconst          VARCHAR(20)  PK   — title ID, e.g. 'tt0000001'
  title_type      VARCHAR(50)       — movie | tvSeries | short | tvEpisode | tvMovie | ...
  primary_title   TEXT              — most widely known title
  original_title  TEXT              — original-language title
  is_adult        BOOLEAN
  start_year      INTEGER           — release year (or series start year)
  end_year        INTEGER           — series end year (NULL for movies)
  runtime_minutes INTEGER
  genres          TEXT[]            — e.g. ARRAY['Drama','Comedy']

Table: title_ratings
  tconst          VARCHAR  FK → title_basics.tconst  PK
  average_rating  FLOAT                — weighted average (1–10)
  num_votes       INTEGER              — number of user votes

Table: title_akas
  title_id        VARCHAR  FK → title_basics.tconst  (composite PK with ordering)
  ordering        INTEGER              (composite PK)
  title           TEXT                 — localised title
  region          VARCHAR(10)          — e.g. 'US', 'GB'
  country_name    VARCHAR(255)         — full country name
  language        VARCHAR(20)          — language code
  types           TEXT[]
  attributes      TEXT[]
  is_original_title BOOLEAN

Table: title_crew
  tconst          VARCHAR  FK → title_basics.tconst  PK
  directors       TEXT[]               — array of nconst IDs
  writers         TEXT[]               — array of nconst IDs

Table: title_episode
  tconst          VARCHAR  FK → title_basics.tconst  PK
  parent_tconst   VARCHAR  FK → title_basics.tconst  — parent TV series
  season_number   INTEGER
  episode_number  INTEGER

Table: title_principals
  tconst          VARCHAR  FK → title_basics.tconst  (composite PK with ordering)
  ordering        INTEGER              (composite PK)
  nconst          VARCHAR  FK → name_basics.nconst
  category        VARCHAR(100)         — actor | director | producer | writer | ...
  job             TEXT                 — specific job title
  characters      TEXT                 — JSON string of character names

Table: name_basics
  nconst          VARCHAR(20)  PK      — person ID, e.g. 'nm0000001'
  primary_name    TEXT
  birth_year      INTEGER
  death_year      INTEGER
  primary_profession TEXT[]            — e.g. ARRAY['actor','producer']
  known_for_titles   TEXT[]            — array of tconst IDs

Query tips:
  - Search arrays with: 'Drama' = ANY(genres)
  - tconst IDs look like 'tt1234567'; nconst IDs like 'nm1234567'
  - Ratings live in title_ratings — always JOIN it when filtering/sorting by rating
  - People ↔ titles via title_principals; person details in name_basics
  - Always add LIMIT (default 20) unless the user asks for more
"""
