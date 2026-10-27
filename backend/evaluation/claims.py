"""Claim extraction module for Vishwas AI."""

import json
import re
from typing import Any

from backend.llm import call_llm
from backend.evaluation.prompts import (
    EXTRACT_CLAIMS_RETRY_PROMPT_TEMPLATE,
    EXTRACT_CLAIMS_SYSTEM_PROMPT,
    EXTRACT_CLAIMS_USER_PROMPT_TEMPLATE,
)


def _clean_and_parse_claims_json(raw_text: str) -> list[str]:
    """Strip code fences, clean LLM response, and parse a list of claim strings."""
    cleaned = raw_text.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: attempt to find a JSON array within the text
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            raise ValueError(f"Could not locate JSON array in response: {cleaned[:120]}...")

    if isinstance(data, dict):
        for candidate_key in ["claims", "extracted_claims", "atomic_claims", "assertions", "items", "data"]:
            if candidate_key in data and isinstance(data[candidate_key], list):
                data = data[candidate_key]
                break
        else:
            for val in data.values():
                if isinstance(val, list):
                    data = val
                    break
            else:
                # Local LLMs in JSON object mode may return {"claim_1": "text...", "claim_2": "text..."}
                claims_from_dict: list[str] = []
                for k, v in data.items():
                    if isinstance(v, str) and len(v.strip()) > 3:
                        claims_from_dict.append(v.strip())
                    elif isinstance(k, str) and len(k.strip()) > 3 and not k.startswith("claim"):
                        claims_from_dict.append(k.strip())
                if claims_from_dict:
                    data = claims_from_dict

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    claims: list[str] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            claims.append(item.strip())
        elif isinstance(item, dict) and "claim" in item:
            # Handle potential dict element fallback gracefully
            claims.append(str(item["claim"]).strip())
        elif isinstance(item, dict) and "text" in item:
            claims.append(str(item["text"]).strip())

    return claims


def extract_claims(
    question: str,
    answer: str,
    cost_tracker: dict[str, int] | None = None,
    trace: dict[str, Any] | None = None,
) -> list[str]:
    """Decompose an answer into atomic, self-contained factual claims.

    Args:
        question: The source question or context prompt.
        answer: The text to audit.
        cost_tracker: Optional dictionary to increment 'llm_calls'.
        trace: Optional dictionary to capture intermediate inspection data.

    Returns:
        List of atomic claim strings.

    Raises:
        RuntimeError: If LLM communication fails.
        ValueError: If response cannot be parsed into a JSON array after retry.
    """
    if not answer or not answer.strip():
        return []

    user_prompt = EXTRACT_CLAIMS_USER_PROMPT_TEMPLATE.format(
        question=question.strip(),
        answer=answer.strip(),
    )

    if cost_tracker is not None:
        cost_tracker["llm_calls"] = cost_tracker.get("llm_calls", 0) + 1

    try:
        raw_response = call_llm(
            system=EXTRACT_CLAIMS_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.0,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM communication error during claim extraction: {exc}") from exc

    if trace is not None:
        trace["system_prompt"] = EXTRACT_CLAIMS_SYSTEM_PROMPT
        trace["user_prompt"] = user_prompt
        trace["raw_response"] = raw_response

    try:
        claims = _clean_and_parse_claims_json(raw_response)
        if trace is not None:
            trace["claims"] = claims
        return claims
    except Exception as first_error:
        # Retry once with stricter prompt
        retry_prompt = EXTRACT_CLAIMS_RETRY_PROMPT_TEMPLATE.format(
            raw_response=raw_response
        )

        if cost_tracker is not None:
            cost_tracker["llm_calls"] = cost_tracker.get("llm_calls", 0) + 1

        try:
            retry_response = call_llm(
                system=EXTRACT_CLAIMS_SYSTEM_PROMPT,
                user=retry_prompt,
                temperature=0.0,
            )
            return _clean_and_parse_claims_json(retry_response)
        except Exception as retry_error:
            raise ValueError(
                f"Failed to parse LLM response into JSON claims array after retry. "
                f"Initial error: {first_error}; Retry error: {retry_error}"
            ) from retry_error
