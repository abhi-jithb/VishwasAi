"""Configuration module for Vishwas AI.

Loads environment variables using python-dotenv and exposes configuration constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


def _get_required_env(var_name: str) -> str:
    """Retrieve an environment variable or raise an error naming the missing variable."""
    value = os.getenv(var_name)
    if value is None or not value.strip():
        raise ValueError(
            f"Missing required environment variable: '{var_name}'. "
            "Please ensure it is set in your .env file."
        )
    return value.strip()


LLM_PROVIDER: str = _get_required_env("LLM_PROVIDER")
LLM_MODEL: str = _get_required_env("LLM_MODEL")
LLM_API_KEY: str = _get_required_env("LLM_API_KEY")
SEARCH_API_KEY: str = _get_required_env("SEARCH_API_KEY")
