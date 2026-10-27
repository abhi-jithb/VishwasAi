"""Disk cache implementation for LLM responses and search results using SQLite."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any

from backend.config import CACHE_DB_PATH

from contextlib import contextmanager

_LOCK = threading.Lock()


@contextmanager
def _db_session():
    """Create, configure, and safely close a thread-safe SQLite connection."""
    db_path = Path(CACHE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
    finally:
        conn.close()


def init_cache_db() -> None:
    """Initialize cache tables if they do not exist."""
    with _LOCK:
        with _db_session() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    user_prompt TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    format_type TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    max_results INTEGER NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()


# Auto-initialize tables upon import
try:
    init_cache_db()
except Exception:
    pass


def _is_cache_disabled() -> bool:
    """Check if cache is explicitly disabled via environment."""
    return os.getenv("DISABLE_CACHE", "0").lower() in ("1", "true", "yes")


def _compute_llm_key(
    provider: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    format_type: str,
) -> str:
    """Compute deterministic SHA-256 hash for LLM request parameters."""
    raw = f"{provider.strip().lower()}||{model.strip().lower()}||{system.strip()}||{user.strip()}||{temperature:.4f}||{format_type.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compute_search_key(query: str, max_results: int) -> str:
    """Compute deterministic SHA-256 hash for search query."""
    raw = f"{query.strip().lower()}||{max_results}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_llm_cache(
    provider: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    format_type: str = "",
) -> str | None:
    """Retrieve cached LLM response if present."""
    if _is_cache_disabled():
        return None

    key = _compute_llm_key(provider, model, system, user, temperature, format_type)
    try:
        with _LOCK:
            with _db_session() as conn:
                cursor = conn.execute(
                    "SELECT response FROM llm_cache WHERE cache_key = ?", (key,)
                )
                row = cursor.fetchone()
                if row:
                    return str(row[0])
    except Exception:
        return None
    return None


def set_llm_cache(
    provider: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    response: str,
    format_type: str = "",
) -> None:
    """Store LLM response into disk cache."""
    if _is_cache_disabled() or not response:
        return

    key = _compute_llm_key(provider, model, system, user, temperature, format_type)
    try:
        with _LOCK:
            with _db_session() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO llm_cache 
                    (cache_key, provider, model, system_prompt, user_prompt, temperature, format_type, response)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (key, provider, model, system, user, temperature, format_type, response),
                )
                conn.commit()
    except Exception:
        pass


def get_search_cache(query: str, max_results: int) -> list[dict[str, Any]] | None:
    """Retrieve cached search results if present."""
    if _is_cache_disabled():
        return None

    key = _compute_search_key(query, max_results)
    try:
        with _LOCK:
            with _db_session() as conn:
                cursor = conn.execute(
                    "SELECT results_json FROM search_cache WHERE cache_key = ?", (key,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
    except Exception:
        return None
    return None


def set_search_cache(query: str, max_results: int, results: list[dict[str, Any]]) -> None:
    """Store search results into disk cache."""
    if _is_cache_disabled() or results is None:
        return

    key = _compute_search_key(query, max_results)
    try:
        with _LOCK:
            with _db_session() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO search_cache 
                    (cache_key, query, max_results, results_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (key, query, max_results, json.dumps(results)),
                )
                conn.commit()
    except Exception:
        pass


def get_cache_stats() -> dict[str, int]:
    """Return count of cached LLM and Search entries."""
    stats = {"llm_cached": 0, "search_cached": 0}
    try:
        with _LOCK:
            with _db_session() as conn:
                llm_cur = conn.execute("SELECT COUNT(*) FROM llm_cache")
                stats["llm_cached"] = llm_cur.fetchone()[0]
                search_cur = conn.execute("SELECT COUNT(*) FROM search_cache")
                stats["search_cached"] = search_cur.fetchone()[0]
    except Exception:
        pass
    return stats
