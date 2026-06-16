import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from chatbot.api.graph.schema import SCHEMA
from chatbot.api.graph.state import ChatState
from chatbot.api.llm.factory import get_llm
from chatbot.api.tools.sql_tool import execute_sql
from logger import get_logger

logger = get_logger("chatbot.api.graph.nodes")

_SQL_SYSTEM = f"""You are a PostgreSQL expert. Given the conversation below, write a single \
valid SQL query that answers the user's latest question.

Schema:
{SCHEMA}

Rules:
- Output ONLY the raw SQL query — no markdown fences, no explanation.
- Use lowercase identifiers.
- Default LIMIT 20 unless the user asks for more.
- Use ANY() for array columns, e.g. 'Drama' = ANY(genres).
"""

_ANSWER_SYSTEM = """You are a helpful assistant that answers questions about IMDB data.
A SQL query has already been executed and the results are provided below.
Use those results to give a concise, clear answer.
If the results are empty or contain an error, say so honestly — do not invent data."""


def _clean_sql(raw: str) -> str:
    sql = raw.strip()
    for fence in (
        "```sql\n",
        "```postgresql\n",
        "```\n",
        "```sql",
        "```postgresql",
        "```",
    ):
        if sql.startswith(fence):
            sql = sql[len(fence) :]
            break
    return sql.rstrip("`").strip()


def _thread(config: RunnableConfig) -> str:
    return config["configurable"].get("thread_id", "unknown")


async def generate_sql(state: ChatState, config: RunnableConfig) -> dict:
    tid = _thread(config)
    cfg = config["configurable"]
    logger.info(
        "[thread:%s] generate_sql — provider=%s model=%s history_len=%d",
        tid,
        cfg["provider"],
        cfg["model"],
        len(state["messages"]),
    )
    llm = get_llm(cfg["provider"], cfg["model"])
    messages = [SystemMessage(content=_SQL_SYSTEM), *state["messages"]]
    response = await llm.ainvoke(messages)
    sql = _clean_sql(response.content)
    logger.debug("[thread:%s] Generated SQL: %s", tid, sql)
    return {"sql_query": sql}


async def run_sql(state: ChatState, config: RunnableConfig) -> dict:
    tid = _thread(config)
    logger.info("[thread:%s] run_sql — executing query", tid)
    logger.debug("[thread:%s] SQL: %s", tid, state["sql_query"])
    results = await asyncio.to_thread(execute_sql, state["sql_query"])
    row_info = results.splitlines()[0] if results else "(empty)"
    logger.info("[thread:%s] run_sql — result header: %s", tid, row_info)
    return {"sql_results": results}


async def generate_answer(state: ChatState, config: RunnableConfig) -> dict:
    tid = _thread(config)
    cfg = config["configurable"]
    logger.info(
        "[thread:%s] generate_answer — provider=%s model=%s",
        tid,
        cfg["provider"],
        cfg["model"],
    )
    llm = get_llm(cfg["provider"], cfg["model"])
    context = (
        f"SQL query executed:\n{state['sql_query']}\n\nResults:\n{state['sql_results']}"
    )
    messages = [
        SystemMessage(content=_ANSWER_SYSTEM),
        *state["messages"],
        HumanMessage(content=f"[Database context]\n{context}"),
    ]
    response = await llm.ainvoke(messages)
    logger.info(
        "[thread:%s] generate_answer — answer length=%d chars",
        tid,
        len(response.content),
    )
    return {"messages": [AIMessage(content=response.content)]}
