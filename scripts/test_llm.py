"""Connectivity test script for the LLM abstraction."""

import sys


def main() -> None:
    """Test LLM invocation and print the result or an informative error."""
    try:
        from backend.config import LLM_MODEL, LLM_PROVIDER
        from backend.llm import call_llm

        print(f"Testing LLM connectivity [Provider: '{LLM_PROVIDER}', Model: '{LLM_MODEL}']...")
        system_prompt = "You are a concise AI verification assistant."
        user_prompt = "Say 'Hello from Vishwas AI!' and briefly state your model name."

        response = call_llm(
            system=system_prompt,
            user=user_prompt,
            temperature=0.0,
        )

        print("\n--- LLM Response Received ---")
        print(response)
        print("-----------------------------")
        print("LLM connectivity test completed successfully.")
    except Exception as exc:
        print(f"\n[ERROR] LLM connectivity test failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
