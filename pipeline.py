"""
Pipeline Orchestrator:
1) clean raw TSV to parquet
2) load parquet to Postgres with idempotent upserts
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.postgres_session import SessionLocal, engine
from LoadAndStore import clean_raw_files
from models import (
    NameBasics,
    TitleAkas,
    TitleBasics,
    TitleCrew,
    TitleEpisode,
    TitlePrincipals,
    TitleRatings,
)

DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "title.basics": {
        "model": TitleBasics,
        "rename": {
            "titleType": "title_type",
            "primaryTitle": "primary_title",
            "originalTitle": "original_title",
            "isAdult": "is_adult",
            "startYear": "start_year",
            "endYear": "end_year",
            "runtimeMinutes": "runtime_minutes",
            "genres": "genres",
        },
        "arrays": ["genres"],
        "ints": ["start_year", "end_year", "runtime_minutes"],
        "bools": ["is_adult"],
    },
    "title.akas": {
        "model": TitleAkas,
        "rename": {
            "titleId": "title_id",
            "title_id": "title_id",
            "ordering": "ordering",
            "title": "title",
            "region": "region",
            "country_name": "country_name",
            "language": "language",
            "types": "types",
            "attributes": "attributes",
            "isOriginalTitle": "is_original_title",
            "is_original_title": "is_original_title",
        },
        "arrays": ["types", "attributes"],
        "ints": ["ordering"],
        "bools": ["is_original_title"],
        "fks": [
            {"column": "title_id", "ref_model": TitleBasics, "ref_column": "tconst"}
        ],
    },
    "title.crew": {
        "model": TitleCrew,
        "rename": {"tconst": "tconst", "directors": "directors", "writers": "writers"},
        "arrays": ["directors", "writers"],
        "fks": [{"column": "tconst", "ref_model": TitleBasics, "ref_column": "tconst"}],
    },
    "title.episode": {
        "model": TitleEpisode,
        "rename": {
            "tconst": "tconst",
            "parentTconst": "parent_tconst",
            "seasonNumber": "season_number",
            "episodeNumber": "episode_number",
        },
        "ints": ["season_number", "episode_number"],
        "fks": [
            {"column": "tconst", "ref_model": TitleBasics, "ref_column": "tconst"},
            {
                "column": "parent_tconst",
                "ref_model": TitleBasics,
                "ref_column": "tconst",
            },
        ],
    },
    "title.principals": {
        "model": TitlePrincipals,
        "rename": {
            "tconst": "tconst",
            "ordering": "ordering",
            "nconst": "nconst",
            "category": "category",
            "job": "job",
            "characters": "characters",
        },
        "ints": ["ordering"],
        "fks": [
            {"column": "tconst", "ref_model": TitleBasics, "ref_column": "tconst"},
            {"column": "nconst", "ref_model": NameBasics, "ref_column": "nconst"},
        ],
    },
    "title.ratings": {
        "model": TitleRatings,
        "rename": {
            "tconst": "tconst",
            "averageRating": "average_rating",
            "numVotes": "num_votes",
        },
        "ints": ["num_votes"],
        "floats": ["average_rating"],
        "fks": [{"column": "tconst", "ref_model": TitleBasics, "ref_column": "tconst"}],
    },
    "name.basics": {
        "model": NameBasics,
        "rename": {
            "nconst": "nconst",
            "primaryName": "primary_name",
            "birthYear": "birth_year",
            "deathYear": "death_year",
            "primaryProfession": "primary_profession",
            "knownForTitles": "known_for_titles",
        },
        "arrays": ["primary_profession", "known_for_titles"],
        "ints": ["birth_year", "death_year"],
    },
}

LOAD_ORDER = [
    "name.basics",
    "title.basics",
    "title.akas",
    "title.ratings",
    "title.crew",
    "title.episode",
    "title.principals",
]

DEFAULT_LOAD_BATCH_SIZE = 1000


def ensure_run_log_table() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS pipeline_run_log (
      id BIGSERIAL PRIMARY KEY,
      run_id TEXT NOT NULL,
      stage TEXT NOT NULL,
      dataset TEXT NOT NULL,
      file_name TEXT,
      row_count INTEGER DEFAULT 0,
      status TEXT NOT NULL,
      error_message TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def log_run(
    session: Session,
    run_id: str,
    stage: str,
    dataset: str,
    status: str,
    file_name: str | None = None,
    row_count: int = 0,
    error_message: str | None = None,
) -> None:
    stmt = text(
        """
        INSERT INTO pipeline_run_log
        (run_id, stage, dataset, file_name, row_count, status, error_message, created_at)
        VALUES (:run_id, :stage, :dataset, :file_name, :row_count, :status, :error_message, :created_at)
        """
    )
    session.execute(
        stmt,
        {
            "run_id": run_id,
            "stage": stage,
            "dataset": dataset,
            "file_name": file_name,
            "row_count": row_count,
            "status": status,
            "error_message": error_message,
            "created_at": datetime.now(timezone.utc),
        },
    )


def parse_dataset_from_parquet(path: Path) -> str:
    parts = path.name.split(".")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer dataset from filename: {path.name}")
    return ".".join(parts[:2])


def _is_array_like(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray, pd.Series))


def as_nullable(value: Any) -> Any:
    if _is_array_like(value):
        return value
    if pd.isna(value):
        return None
    return value


def as_list(value: Any) -> list[str] | None:
    if _is_array_like(value):
        items = [str(v).strip() for v in value if not pd.isna(v) and str(v).strip()]
        return items or None
    value = as_nullable(value)
    if value is None:
        return None
    items = [v.strip() for v in str(value).split(",") if v.strip()]
    return items or None


def normalize_scalar(value: Any) -> Any:
    if _is_array_like(value):
        return [normalize_scalar(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _existing_parent_keys(model: Any, column: str, candidates: set[str]) -> set[str]:
    if not candidates:
        return set()
    table_name = model.__tablename__
    stmt = text(f"SELECT {column} FROM {table_name} WHERE {column} = ANY(:keys)")
    with engine.connect() as conn:
        result = conn.execute(stmt, {"keys": list(candidates)})
        return {row[0] for row in result}


def filter_fk_orphans(
    df: pd.DataFrame, config: dict[str, Any], dataset: str, file_name: str
) -> pd.DataFrame:
    for fk in config.get("fks", []):
        col = fk["column"]
        if col not in df.columns:
            continue
        candidates = set(df[col].dropna().astype(str).unique())
        existing = _existing_parent_keys(fk["ref_model"], fk["ref_column"], candidates)
        before = len(df)
        mask = df[col].isna() | df[col].isin(existing)
        df = df.loc[mask]
        dropped = before - len(df)
        if dropped:
            print(
                f"[load][fk-drop] {file_name} {dataset}.{col} -> "
                f"{fk['ref_model'].__tablename__}.{fk['ref_column']}: {dropped} orphans"
            )
    return df


def validate_required_columns(
    df: pd.DataFrame, config: dict[str, Any], dataset: str
) -> pd.DataFrame:
    pk_columns = [c.name for c in config["model"].__table__.primary_key.columns]
    missing_pk = [col for col in pk_columns if col not in df.columns]
    if missing_pk:
        raise ValueError(
            f"{dataset}: cleaned parquet is missing required primary key columns: {missing_pk}"
        )
    return df.dropna(subset=pk_columns)


def normalize_frame(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    df = df.rename(columns=config.get("rename", {}))
    df = df.loc[:, ~df.columns.duplicated(keep="last")]
    if "ints" in config:
        for col in config["ints"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "floats" in config:
        for col in config["floats"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    if "bools" in config:
        for col in config["bools"]:
            if col in df.columns:
                df[col] = df[col].astype("boolean")
    if "arrays" in config:
        for col in config["arrays"]:
            if col in df.columns:
                df[col] = df[col].map(as_list)
    return df


def upsert_dataframe(session: Session, model: Any, df: pd.DataFrame) -> int:
    return upsert_dataframe_batched(session, model, df, DEFAULT_LOAD_BATCH_SIZE)


def upsert_dataframe_batched(
    session: Session, model: Any, df: pd.DataFrame, batch_size: int
) -> int:
    table = model.__table__
    db_columns = [c.name for c in table.columns]
    pk_columns = [c.name for c in table.primary_key.columns]
    usable_columns = [c for c in db_columns if c in df.columns]
    if not usable_columns:
        return 0

    total_rows = 0
    records = []
    for row in df[usable_columns].to_dict(orient="records"):
        records.append({key: normalize_scalar(value) for key, value in row.items()})

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        if not batch:
            continue

        stmt = insert(table).values(batch)
        update_cols = {
            c: getattr(stmt.excluded, c) for c in usable_columns if c not in pk_columns
        }
        if update_cols:
            stmt = stmt.on_conflict_do_update(
                index_elements=pk_columns,
                set_=update_cols,
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=pk_columns)
        session.execute(stmt)
        total_rows += len(batch)

    return total_rows


def load_cleaned_to_db(
    clean_dir: Path, run_id: str, batch_size: int = DEFAULT_LOAD_BATCH_SIZE
) -> None:
    parquet_files = sorted(clean_dir.glob("*.parquet"))
    files_by_dataset: dict[str, list[Path]] = {}
    for path in parquet_files:
        dataset = parse_dataset_from_parquet(path)
        if dataset not in DATASET_CONFIG:
            continue
        files_by_dataset.setdefault(dataset, []).append(path)

    with SessionLocal() as session:
        for dataset in LOAD_ORDER:
            for path in files_by_dataset.get(dataset, []):
                cfg = DATASET_CONFIG[dataset]
                try:
                    df = pd.read_parquet(path)
                    df = normalize_frame(df, cfg)
                    df = validate_required_columns(df, cfg, dataset)
                    df = filter_fk_orphans(df, cfg, dataset, path.name)
                    loaded_count = upsert_dataframe_batched(
                        session, cfg["model"], df, batch_size
                    )
                    session.commit()
                    log_run(
                        session=session,
                        run_id=run_id,
                        stage="load",
                        dataset=dataset,
                        status="success",
                        file_name=path.name,
                        row_count=loaded_count,
                    )
                    session.commit()
                    print(f"[load] {path.name}: {loaded_count} rows")
                except Exception as exc:
                    session.rollback()
                    log_run(
                        session=session,
                        run_id=run_id,
                        stage="load",
                        dataset=dataset,
                        status="failed",
                        file_name=path.name,
                        error_message=str(exc),
                    )
                    session.commit()
                    print(f"[load][failed] {path.name}: {exc}")


def run_clean_stage(
    raw_dir: Path, clean_dir: Path, chunk_size: int, run_id: str
) -> None:
    stats = clean_raw_files(raw_dir, clean_dir, chunk_size)
    with SessionLocal() as session:
        for item in stats:
            log_run(
                session=session,
                run_id=run_id,
                stage="clean",
                dataset=str(item["dataset"]),
                status="success",
                row_count=int(item["rows"]),
            )
        session.commit()
    for item in stats:
        print(f"[clean] {item['dataset']}: rows={item['rows']} chunks={item['chunks']}")


def main() -> None:
    load_dotenv(".env.dev")
    parser = argparse.ArgumentParser(description="IMDB ingestion pipeline")
    parser.add_argument(
        "--mode",
        choices=["clean-only", "load-only", "full-run"],
        default="full-run",
    )
    parser.add_argument("--raw-dir", default="./data/raw")
    parser.add_argument("--clean-dir", default="./data/cleaned")
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--load-batch-size", type=int, default=DEFAULT_LOAD_BATCH_SIZE)
    args = parser.parse_args()

    ensure_run_log_table()
    run_id = str(uuid.uuid4())
    raw_dir = Path(args.raw_dir)
    clean_dir = Path(args.clean_dir)

    if args.mode in {"clean-only", "full-run"}:
        run_clean_stage(raw_dir, clean_dir, args.chunk_size, run_id)
    if args.mode in {"load-only", "full-run"}:
        load_cleaned_to_db(clean_dir, run_id, args.load_batch_size)
    print(f"Pipeline completed. run_id={run_id}")


if __name__ == "__main__":
    main()
