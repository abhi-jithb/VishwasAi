"""Configuration module for Vishwas AI.

Loads environment variables using python-dotenv and exposes configuration constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


def _get_env_or_default(var_name: str, default: str = "") -> str:
    """Retrieve an environment variable or return default."""
    value = os.getenv(var_name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _get_required_env(var_name: str) -> str:
    """Retrieve an environment variable or raise an error naming the missing variable."""
    value = os.getenv(var_name)
    if value is None or not value.strip():
        raise ValueError(
            f"Missing required environment variable: '{var_name}'. "
            "Please ensure it is set in your .env file."
        )
    return value.strip()


LLM_PROVIDER: str = _get_env_or_default("LLM_PROVIDER", "ollama")
BENCHMARK_MODEL: str = _get_env_or_default("BENCHMARK_MODEL", "")
LLM_MODEL: str = BENCHMARK_MODEL or _get_env_or_default(
    "LLM_MODEL",
    "qwen2.5:7b" if LLM_PROVIDER.lower() == "ollama" else "llama-3.3-70b-versatile",
)
LLM_API_KEY: str = _get_env_or_default("LLM_API_KEY", "")
if LLM_PROVIDER.lower() != "ollama" and not LLM_API_KEY:
    LLM_API_KEY = _get_required_env("LLM_API_KEY")

SEARCH_API_KEY: str = _get_env_or_default("SEARCH_API_KEY", "")
OLLAMA_BASE_URL: str = _get_env_or_default("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
CACHE_DB_PATH: Path = _PROJECT_ROOT / "data" / "cache.db"


def get_active_provider() -> str:
    """Return currently active provider considering runtime environment changes."""
    return os.getenv("LLM_PROVIDER", "").strip().lower() or LLM_PROVIDER.strip().lower()


def get_active_model() -> str:
    """Return currently active model considering BENCHMARK_MODEL or runtime env."""
    bm = os.getenv("BENCHMARK_MODEL", "").strip()
    if bm:
        return bm
    return os.getenv("LLM_MODEL", "").strip() or LLM_MODEL
