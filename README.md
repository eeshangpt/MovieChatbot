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

## Pipeline Commands
1. Clean raw TSV to parquet only:
   `.venv/bin/python LoadAndStore.py --raw-dir ./data/raw --clean-dir ./data/cleaned --chunk-size 100000`
2. Full pipeline (clean + load to Postgres):
   `.venv/bin/python pipeline.py --mode full-run --raw-dir ./data/raw --clean-dir ./data/cleaned --chunk-size 100000`
3. Load cleaned parquet to Postgres only:
   `.venv/bin/python pipeline.py --mode load-only --clean-dir ./data/cleaned`

## Database Access
To access the database shell directly via Docker:
```bash
docker exec -it <container_name> psql -U <user_from_env> -d <db_from_env>
```
Credentials can be found in your local `.env.dev` file.
