import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from chatbot.api.routes.chat import router as chat_router
from logger import get_logger

logger = get_logger("chatbot.api.main")

app = FastAPI(title="IMDB Chatbot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.on_event("startup")
async def on_startup():
    logger.info("IMDB Chatbot API started")


@app.get("/health")
def health():
    return {"status": "ok"}
