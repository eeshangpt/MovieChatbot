from sqlalchemy import text

from db.postgres_session import engine
from logger import get_logger

logger = get_logger("chatbot.api.tools.sql_tool")

_MAX_ROWS = 100


def execute_sql(query: str) -> str:
    logger.debug("Executing SQL: %s", query)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchmany(_MAX_ROWS)
            if not rows:
                logger.warning("SQL returned no rows")
                return "Query returned no results."
            cols = list(result.keys())
            header = " | ".join(cols)
            lines = [header, "-" * len(header)]
            for row in rows:
                lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
            if len(rows) == _MAX_ROWS:
                lines.append(f"... (results capped at {_MAX_ROWS} rows)")
            logger.info("SQL returned %d row(s)", len(rows))
            return "\n".join(lines)
    except Exception as exc:
        logger.error("SQL execution failed: %s", exc)
        return f"SQL Error: {exc}"
