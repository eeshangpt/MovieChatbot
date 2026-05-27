# IMDB Data Analysis
AUTHOR: [Eeshan Gupta](eeshan.gupta@hotmail.com)

## Phase 1: Data Ingestion
1. Ingesting data from `.tsv` files and storing them as raw `.parquet`

## Local setup
1. Start Postgres:
   `docker compose up -d`
2. Copy environment template:
   `cp .env.example .env.dev`
3. Create DB tables:
   `python create_tables.py`
