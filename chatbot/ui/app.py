import json
import os
import sys
import uuid
from pathlib import Path

# Ensure the repo root is on sys.path so top-level modules (logger, db, …) are importable
# when Streamlit launches this script directly from chatbot/ui/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import streamlit as st
from dotenv import load_dotenv

from logger import get_logger

load_dotenv(".env.dev")

logger = get_logger("chatbot.ui.app")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

PROVIDERS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "anthropic": ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001"],
}

st.set_page_config(page_title="IMDB Chatbot", page_icon="🎬", layout="centered")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    provider = st.selectbox("Provider", list(PROVIDERS.keys()))
    model = st.selectbox("Model", PROVIDERS[provider])
    st.divider()
    if st.button("🗑️ New conversation", use_container_width=True):
        logger.info("New conversation started by user")
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.caption(f"API: `{API_BASE_URL}`")

# ── Session state ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    logger.info("Session initialised — thread_id=%s", st.session_state.thread_id)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎬 IMDB Chatbot")
st.caption("Ask anything about movies, ratings, directors, actors…")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("e.g. Top 10 highest-rated movies of the 90s"):
    logger.info(
        "[thread:%s] User message — provider=%s model=%s message=%r",
        st.session_state.thread_id,
        provider,
        model,
        prompt[:80],
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            logger.debug(
                "[thread:%s] Calling POST %s/chat/stream",
                st.session_state.thread_id,
                API_BASE_URL,
            )
            with httpx.Client(timeout=120) as client:
                with client.stream(
                    "POST",
                    f"{API_BASE_URL}/chat/stream",
                    json={
                        "message": prompt,
                        "thread_id": st.session_state.thread_id,
                        "provider": provider,
                        "model": model,
                    },
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        token = json.loads(payload).get("token", "")
                        full_response += token
                        placeholder.markdown(full_response + "▌")

            logger.info(
                "[thread:%s] Response complete — %d chars",
                st.session_state.thread_id,
                len(full_response),
            )
        except Exception as exc:
            logger.error(
                "[thread:%s] API call failed: %s",
                st.session_state.thread_id,
                exc,
            )
            full_response = f"⚠️ Error contacting API: {exc}"

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
