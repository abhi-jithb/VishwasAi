"""LLM abstraction module for Vishwas AI.

Exposes a unified call_llm function supporting official SDKs for OpenAI, Anthropic, and Groq.
"""

from backend.config import LLM_API_KEY, LLM_MODEL, LLM_PROVIDER


def call_llm(system: str, user: str, temperature: float = 0.0) -> str:
    """Call the configured LLM provider and return the response string.

    Args:
        system: System prompt providing instructions and context.
        user: User message or query to process.
        temperature: Sampling temperature (default: 0.0).

    Returns:
        The generated text response from the model.

    Raises:
        ValueError: If LLM_PROVIDER is unsupported.
    """
    provider: str = LLM_PROVIDER.strip().lower()

    max_retries = 5
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            if provider == "groq":
                from groq import Groq

                client = Groq(api_key=LLM_API_KEY)
                models_to_try = [LLM_MODEL]
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
                        return response.choices[0].message.content or ""
                    except Exception as exc:
                        model_err = exc
                        if "429" in str(exc) or "rate_limit" in str(exc):
                            continue
                        raise exc
                if model_err:
                    raise model_err

            if provider == "openai":
                from openai import OpenAI

                client = OpenAI(api_key=LLM_API_KEY)
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""

            if provider == "anthropic":
                from anthropic import Anthropic

                client = Anthropic(api_key=LLM_API_KEY)
                response = client.messages.create(
                    model=LLM_MODEL,
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
                return "".join(text_blocks)

            raise ValueError(
                f"Unsupported LLM_PROVIDER: '{LLM_PROVIDER}'. "
                "Supported providers are 'openai', 'anthropic', and 'groq'."
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
            raise last_exc
