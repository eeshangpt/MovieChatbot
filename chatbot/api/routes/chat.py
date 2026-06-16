import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from chatbot.api.config import DEFAULT_MODEL, DEFAULT_PROVIDER
from chatbot.api.graph.builder import graph
from logger import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger("chatbot.api.routes.chat")


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    preview = request.message[:80] + ("…" if len(request.message) > 80 else "")
    logger.info(
        "[thread:%s] Request received — provider=%s model=%s message=%r",
        request.thread_id,
        request.provider,
        request.model,
        preview,
    )

    async def event_generator():
        config = {
            "configurable": {
                "thread_id": request.thread_id,
                "provider": request.provider,
                "model": request.model,
            }
        }
        input_state = {"messages": [HumanMessage(content=request.message)]}
        token_count = 0

        try:
            logger.debug("[thread:%s] Starting graph stream", request.thread_id)
            async for event in graph.astream_events(
                input_state, config=config, version="v2"
            ):
                if (
                    event["event"] == "on_chat_model_stream"
                    and event["metadata"].get("langgraph_node") == "generate_answer"
                ):
                    chunk = event["data"]["chunk"]
                    content = chunk.content
                    if isinstance(content, list):
                        content = "".join(
                            c.get("text", "") for c in content if isinstance(c, dict)
                        )
                    if content:
                        token_count += 1
                        yield f"data: {json.dumps({'token': content})}\n\n"

            logger.info(
                "[thread:%s] Stream complete — %d token chunks sent",
                request.thread_id,
                token_count,
            )
        except Exception:
            logger.exception("[thread:%s] Error during graph stream", request.thread_id)
            raise

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
