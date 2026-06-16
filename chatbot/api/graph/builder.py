from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from chatbot.api.graph.nodes import generate_answer, generate_sql, run_sql
from chatbot.api.graph.state import ChatState

_checkpointer = MemorySaver()


def _build():
    g = StateGraph(ChatState)
    g.add_node("generate_sql", generate_sql)
    g.add_node("run_sql", run_sql)
    g.add_node("generate_answer", generate_answer)
    g.set_entry_point("generate_sql")
    g.add_edge("generate_sql", "run_sql")
    g.add_edge("run_sql", "generate_answer")
    g.add_edge("generate_answer", END)
    return g.compile(checkpointer=_checkpointer)


graph = _build()
