"""LLM abstraction module for Vishwas AI.

Exposes a unified call_llm function supporting official SDKs for OpenAI, Anthropic, Groq, and local Ollama.
"""

import json
import os
import urllib.request
import urllib.error

from backend.config import (
    LLM_API_KEY,
    OLLAMA_BASE_URL,
    get_active_model,
    get_active_provider,
)
from backend.cache import get_llm_cache, set_llm_cache


def call_llm(
    system: str,
    user: str,
    temperature: float = 0.0,
    json_mode: bool | None = None,
) -> str:
    """Call the configured LLM provider and return the response string.

    Args:
        system: System prompt providing instructions and context.
        user: User message or query to process.
        temperature: Sampling temperature (default: 0.0).
        json_mode: Optional boolean to explicitly force JSON formatting.

    Returns:
        The generated text response from the model.

    Raises:
        ValueError: If LLM_PROVIDER is unsupported.
    """
    provider = get_active_provider()
    model = get_active_model()
    benchmark_model = os.getenv("BENCHMARK_MODEL", "").strip()

    # Determine if JSON structured format is required
    is_json = json_mode is True or (
        json_mode is None and ("json" in system.lower() or "json" in user.lower())
    )
    format_type = "json" if is_json else ""

    # 1. Check disk cache first
    cached_response = get_llm_cache(
        provider=provider,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        format_type=format_type,
    )
    if cached_response is not None:
        return cached_response

    max_retries = 5
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            # -------------------------------------------------------------
            # Local Ollama Provider (REST API)
            # -------------------------------------------------------------
            if provider == "ollama":
                url = f"{OLLAMA_BASE_URL}/api/chat"
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                    },
                }
                if is_json:
                    payload["format"] = "json"

                req_body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=req_body,
                    headers={"Content-Type": "application/json"},
                )

                try:
                    with urllib.request.urlopen(req, timeout=180.0) as resp:
                        resp_data = json.loads(resp.read().decode("utf-8"))
                        response_text = resp_data.get("message", {}).get("content", "")
                except urllib.error.URLError as err:
                    raise RuntimeError(
                        f"Failed to connect to local Ollama server at {url}. "
                        "Ensure Ollama is running ('ollama serve'). "
                        f"Details: {err}"
                    ) from err

                set_llm_cache(
                    provider=provider,
                    model=model,
                    system=system,
                    user=user,
                    temperature=temperature,
                    response=response_text,
                    format_type=format_type,
                )
                return response_text

            # -------------------------------------------------------------
            # Hosted Groq Provider
            # -------------------------------------------------------------
            if provider == "groq":
                from groq import Groq

                client = Groq(api_key=LLM_API_KEY)
                models_to_try = [model]
                # If BENCHMARK_MODEL is set, strict benchmark execution: no fallback
                if not benchmark_model:
                    for candidate in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]:
                        if candidate not in models_to_try:
                            models_to_try.append(candidate)

                model_err = None
                for m in models_to_try:
                    try:
                        response = client.chat.completions.create(
                            model=m,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            temperature=temperature,
                        )
                        result_text = response.choices[0].message.content or ""
                        set_llm_cache(
                            provider=provider,
                            model=m,
                            system=system,
                            user=user,
                            temperature=temperature,
                            response=result_text,
                            format_type=format_type,
                        )
                        return result_text
                    except Exception as exc:
                        model_err = exc
                        if not benchmark_model and ("429" in str(exc) or "rate_limit" in str(exc)):
                            continue
                        raise exc
                if model_err:
                    raise model_err

            # -------------------------------------------------------------
            # OpenAI Provider
            # -------------------------------------------------------------
            if provider == "openai":
                from openai import OpenAI

                client = OpenAI(api_key=LLM_API_KEY)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                )
                result_text = response.choices[0].message.content or ""
                set_llm_cache(
                    provider=provider,
                    model=model,
                    system=system,
                    user=user,
                    temperature=temperature,
                    response=result_text,
                    format_type=format_type,
                )
                return result_text

            # -------------------------------------------------------------
            # Anthropic Provider
            # -------------------------------------------------------------
            if provider == "anthropic":
                from anthropic import Anthropic

                client = Anthropic(api_key=LLM_API_KEY)
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=system,
                    messages=[
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                )
                text_blocks = [
                    block.text for block in response.content if getattr(block, "text", None)
                ]
                result_text = "".join(text_blocks)
                set_llm_cache(
                    provider=provider,
                    model=model,
                    system=system,
                    user=user,
                    temperature=temperature,
                    response=result_text,
                    format_type=format_type,
                )
                return result_text

            raise ValueError(
                f"Unsupported LLM_PROVIDER: '{provider}'. "
                "Supported providers are 'ollama', 'groq', 'openai', and 'anthropic'."
            )
        except ValueError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                import random
                import re
                import time

                err_str = str(exc)
                wait_sec = 2.0 * (attempt + 1)
                match = re.search(r"try again in ([\d\.]+)s", err_str)
                if match:
                    wait_sec = float(match.group(1)) + 1.0
                elif "429" in err_str or "rate_limit" in err_str:
                    wait_sec = max(wait_sec, 6.0)
                # Add jitter to avoid thundering herd across concurrent threads
                wait_sec += random.uniform(1.0, 3.5)
                time.sleep(wait_sec)
                continue
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("LLM request failed with unknown error.")
