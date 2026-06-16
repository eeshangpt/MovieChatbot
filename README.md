# IMDB Data Analysis
AUTHOR: [Eeshan Gupta](mailto:eeshangpt@gmail.com)

---

## Overview

End-to-end IMDB data platform:

| Layer | Technology | Purpose |
| --- | --- | --- |
| Ingestion | pandas + pyarrow | TSV dumps → cleaned parquet → Postgres |
| Storage | PostgreSQL 16 | Relational store for all IMDB datasets |
| Graph DB | Neo4j 5.20 | Graph layer (future use) |
| Chatbot API | FastAPI + LangGraph | Natural-language queries over IMDB data |
| Chatbot UI | Streamlit | Browser chat interface |

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python ≥ 3.14

---

## Environment Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd IMDB

# 2. Copy the env template and fill in all values
cp .env.example .env.dev

# 3. Install dependencies
uv sync

# 4. Start Postgres + Neo4j
docker compose up -d

# 5. Create Postgres tables (one-time)
.venv/bin/python create_tables.py
```

### Required `.env.dev` variables

| Variable | Example                                                          | Used by |
| --- |------------------------------------------------------------------| --- |
| `DATABASE_URL` | `postgresql://admin:admin_pass@localhost:5432/ingestion_db_name` | Pipeline, Chatbot API |
| `POSTGRES_USER` | `admin`                                                          | docker-compose |
| `POSTGRES_PASSWORD` | `admin_pass`                                                     | docker-compose |
| `POSTGRES_DB` | `ingestion_db_name`                                              | docker-compose |
| `POSTGRES_PORT` | `5432`                                                           | docker-compose |
| `NEO4J_HOST` | `localhost`                                                      | Chatbot API |
| `NEO4J_PORT` | `7687`                                                           | docker-compose, Chatbot API |
| `NEO4J_BROWSER_PORT` | `7474`                                                           | docker-compose |
| `NEO4J_USER` | `neo4j`                                                          | docker-compose, Chatbot API |
| `NEO4J_PASSWORD` | `somepassword`                                                   | docker-compose, Chatbot API |
| `OPENAI_API_KEY` | `sk-...`                                                         | Chatbot API |
| `ANTHROPIC_API_KEY` | `sk-ant-...`                                                     | Chatbot API |
| `LLM_PROVIDER` | `openai`                                                         | Chatbot API |
| `LLM_MODEL` | `gpt-4o`                                                         | Chatbot API |

---

## Phase 1 — Data Ingestion Pipeline

Streams each IMDB TSV dump through a two-stage ELT:

1. **Clean** — replaces `\N` sentinels, normalises types, writes snappy parquet chunks.
2. **Load** — idempotent `INSERT … ON CONFLICT DO UPDATE` into Postgres.

```bash
# Clean raw TSV → parquet only
.venv/bin/python LoadAndStore.py \
  --raw-dir ./data/raw \
  --clean-dir ./data/cleaned \
  --chunk-size 100000

# Full pipeline (clean + load)
.venv/bin/python pipeline.py \
  --mode full-run \
  --raw-dir ./data/raw \
  --clean-dir ./data/cleaned \
  --chunk-size 100000

# Load already-cleaned parquet into Postgres
.venv/bin/python pipeline.py \
  --mode load-only \
  --clean-dir ./data/cleaned
```

---

## Phase 2 — IMDB Chatbot

Ask natural-language questions about the IMDB dataset. The chatbot generates SQL, runs it against Postgres, and streams a plain-English answer.

```
User question
    → [LangGraph: generate_sql]   — LLM writes a PostgreSQL query
    → [LangGraph: run_sql]        — query executed against Postgres
    → [LangGraph: generate_answer]— LLM streams a grounded answer
```

Supports **OpenAI** (GPT-4o, GPT-4o-mini) and **Anthropic** (Claude Sonnet/Opus/Haiku) — switchable from the UI sidebar.

### Run locally

```bash
# Terminal 1 — API (port 8000)
.venv/bin/python -m uvicorn chatbot.api.main:app --reload --port 8000

# Terminal 2 — UI (port 8501)
.venv/bin/streamlit run chatbot/ui/app.py
```

Open **http://localhost:8501** in your browser.

### Run via Docker Compose (WIP)

```bash
docker compose up --build chatbot-api chatbot-ui
```

| Service | URL |
| --- | --- |
| Chatbot UI | http://localhost:8501 |
| Chatbot API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Neo4j Browser | http://localhost:7474 |

### Logging

All chatbot logs go to **`logs/chatbot.log`** (rotated at 10 MB) and to the console.
Set `LOG_LEVEL=DEBUG` in `.env.dev` for full SQL query and token-level detail.

---

## Database Access

```bash
# Postgres shell
docker exec -it imdb_postgres psql -U $POSTGRES_USER -d $POSTGRES_DB

# Lint + format before committing
./run-ruff.sh
```