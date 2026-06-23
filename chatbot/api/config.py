import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(".env.dev"))

DATABASE_URL: str = os.getenv("DATABASE_URL", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
