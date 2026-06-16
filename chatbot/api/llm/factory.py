from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from chatbot.api.config import ANTHROPIC_API_KEY, OPENAI_API_KEY


def get_llm(provider: str, model: str) -> BaseChatModel:
    if provider == "openai":
        return ChatOpenAI(model=model, api_key=OPENAI_API_KEY, streaming=True)
    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=ANTHROPIC_API_KEY, streaming=True)
    raise ValueError(
        f"Unsupported provider: {provider!r}. Choose 'openai' or 'anthropic'."
    )
