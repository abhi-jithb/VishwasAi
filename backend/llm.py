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

    max_retries = 3
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            if provider == "groq":
                from groq import Groq

                client = Groq(api_key=LLM_API_KEY)
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""

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
                import time
                time.sleep(1.0 * (attempt + 1))
                continue
            raise last_exc
