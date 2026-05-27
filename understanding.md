# IMDB Ingestion Pipeline: Current Understanding

## 1) What this codebase currently is

This repository is an **early-stage IMDB ingestion project** with three main parts:

1. **Raw-to-cleaned file processing** (`LoadAndStore.py` / notebook form)  
   - Reads IMDB TSV files from `data/raw/` in chunks.
   - Replaces `\N` with nulls.
   - Applies limited dataset-specific preprocessing:
     - `title.akas`: derives `is_original_title` and `country_name`
     - `title.basics`: converts adult flag to boolean
     - `title.episode`: downcasts season/episode number
   - Writes chunked parquet files to `data/cleaned/` (snappy compressed).

2. **Relational schema setup** (`models/*`, `create_tables.py`, `db/session.py`)  
   - SQLAlchemy ORM models exist for all key IMDB datasets (`title_basics`, `title_akas`, `title_crew`, `title_episode`, `title_principals`, `title_ratings`, `name_basics`).
   - `create_tables.py` creates tables in Postgres.
   - Postgres can be run via `docker-compose.yml`.

3. **Unrelated search utility** (`search_engine/*`, `search.py`)  
   - A DuckDuckGo news search wrapper exists.
   - This is separate from IMDB ingestion and not on the ingestion critical path.

## 2) Observed pipeline flow

Current practical data flow appears to be:

`data/.RAW/*.tsv.gz` -> (manually/externally unzipped) -> `data/raw/*.tsv` -> `LoadAndStore.py` chunk cleaning -> `data/cleaned/*.parquet`

There is **no implemented production-grade load step from cleaned parquet into Postgres** yet (despite models and DB setup existing).

## 3) Key issues and risks identified

1. **Critical security issue**  
   - `.env.dev` contains what looks like a real `OPENAI_API_KEY` in plaintext.
   - This should be treated as compromised and rotated immediately.

2. **Ingestion pipeline is incomplete**  
   - Core ELT “Load into DB” step is missing.
   - `LoadAndStore.py` ends in exploratory code and does not perform durable orchestration.

3. **Notebook/script hybrid structure**  
   - `LoadAndStore.py` is jupytext-exported notebook style; not yet a clean CLI/job module.
   - Harder to run in CI/scheduler as-is.

4. **Schema/data mismatch risk**  
   - ORM uses snake_case names (e.g., `title_id`, `is_adult`) while raw data columns are camelCase/raw IMDB names.
   - Explicit mapping/renaming layer is not implemented.
   - `TitleAkas` currently contains fields (`name`, `email`, `age`, `created_at`) that do not belong to IMDB `title.akas` and likely came from scaffolding.

5. **Database naming consistency risk**  
   - ORM tablenames are uppercase (e.g., `TITLE_BASICS`) while some foreign keys reference lowercase names (e.g., `title_basics.tconst`), which is fragile in Postgres quoting behavior.

6. **No orchestration, observability, or data quality checks**  
   - No run metadata, row counts, rejects/dead-letter handling, constraints validation, or load audit.

7. **Environment assumptions are hardcoded**  
   - DB URL is hardcoded in code.
   - Runtime and deployment configs are not parameterized for environments (dev/stage/prod).

8. **Versioning/runtime concerns**  
   - `requires-python >=3.14` may reduce portability depending on deployment target.

## 4) What is working today

- Chunked cleaning and parquet output is working and appears to have already produced a large set of cleaned parquet files.
- SQLAlchemy model set and table creation entrypoint exist.
- Local Postgres bootstrap via Docker Compose exists.

## 5) Recommended next steps (priority order)

## Phase A: Stabilize immediately

1. **Rotate exposed API key now** and remove secrets from tracked files.
2. Add `.env.example` and load secrets/config only from environment.
3. Confirm/standardize table naming + FK references (all lowercase unquoted is the safest route in Postgres).

## Phase B: Make ingestion production-capable

1. Refactor `LoadAndStore.py` into a real module/CLI, for example:
   - `extract_clean.py` (TSV -> cleaned parquet)
   - `load_postgres.py` (parquet -> staging tables -> final tables)
   - `main.py` or task runner entrypoint
2. Implement deterministic column mapping per dataset (raw IMDB columns -> ORM columns).
3. Implement explicit type casting and list parsing (ARRAY fields like genres/professions/directors/writers).
4. Add idempotent load strategy:
   - batch-level metadata table
   - upsert/merge semantics on PKs
   - safe restart behavior.

## Phase C: Add data quality and reliability

1. Add per-dataset validation checks:
   - row counts raw vs cleaned vs loaded
   - null/PK checks
   - FK consistency checks
2. Add structured logging + run summary artifacts.
3. Add tests:
   - unit tests for preprocessing functions
   - integration tests for DB load path.

## Phase D: Operationalization

1. Add Makefile or task commands for standard workflow.
2. Add CI checks (`ruff`, tests, basic smoke run).
3. Optionally add workflow orchestration (Airflow/Prefect/Dagster) once local pipeline is stable.

## 6) Proposed target architecture

1. **Bronze**: raw TSV/TSV.GZ (immutable)
2. **Silver**: cleaned, typed parquet partitions
3. **Gold/Serving**: normalized Postgres tables with constraints and indexes
4. **Metadata**: run table + quality metrics + load audit logs

## 7) Practical implementation order for the next sprint

1. Security cleanup (keys, env handling)  
2. Fix ORM/schema correctness (`TitleAkas`, table naming/FKs)  
3. Build `parquet -> Postgres` loader with idempotency  
4. Add row-count + PK/FK data quality checks  
5. Add tests and CI

---

In short: the project has a good start on **cleaning and schema definition**, but it still needs the **core reliable load/orchestration/data-quality layer** to become a complete ingestion pipeline.
