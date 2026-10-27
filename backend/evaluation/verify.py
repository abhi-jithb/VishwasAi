"""Claim verification module for Vishwas AI."""

import json
import re
from typing import Any

from backend.llm import call_llm
from backend.evaluation.prompts import (
    VERIFY_CLAIM_NO_RETRIEVAL_SYSTEM_PROMPT,
    VERIFY_CLAIM_NO_RETRIEVAL_USER_PROMPT_TEMPLATE,
    VERIFY_CLAIM_RETRY_PROMPT_TEMPLATE,
    VERIFY_CLAIM_SYSTEM_PROMPT,
    VERIFY_CLAIM_USER_PROMPT_TEMPLATE,
)


def _format_evidence_snippets(evidence: list[dict[str, str]]) -> str:
    """Format evidence dictionaries into numbered text blocks for prompting."""
    blocks: list[str] = []
    for idx, item in enumerate(evidence, 1):
        title = item.get("title", "").strip() or "Untitled"
        url = item.get("url", "").strip() or "No URL"
        snippet = item.get("snippet", "").strip() or "No content available."
        blocks.append(f"[{idx}] Title: {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n".join(blocks)


def _clean_and_parse_verification_json(
    raw_text: str,
    available_urls: set[str],
) -> dict[str, Any]:
    """Parse and normalize LLM JSON verification output."""
    cleaned = raw_text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback regex search for JSON object {...}
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            raise ValueError(f"Could not locate JSON object in response: {cleaned[:120]}...")

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    raw_status = str(data.get("status", "")).strip().lower()
    if "unsupported" in raw_status:
        status = "unsupported"
    elif "contradict" in raw_status:
        status = "contradicted"
    elif "support" in raw_status:
        status = "supported"
    else:
        status = "unsupported"

    reasoning = str(data.get("reasoning", "")).strip()
    if not reasoning:
        reasoning = f"Claim evaluated as {status} based on provided evidence."

    raw_evidence_used = data.get("evidence_used", [])
    evidence_used: list[str] = []
    if isinstance(raw_evidence_used, list):
        for u in raw_evidence_used:
            if isinstance(u, str) and u.strip():
                evidence_used.append(u.strip())
    elif isinstance(raw_evidence_used, str) and raw_evidence_used.strip():
        evidence_used.append(raw_evidence_used.strip())

    # If status is unsupported, ensure evidence_used is empty per instructions
    if status == "unsupported":
        evidence_used = []
    elif available_urls:
        # Filter to only URLs that were provided in evidence snippets
        matched_urls = [u for u in evidence_used if u in available_urls]
        if matched_urls:
            evidence_used = matched_urls

    return {
        "status": status,
        "reasoning": reasoning,
        "evidence_used": evidence_used,
    }


def verify_claim(
    claim: str,
    evidence: list[dict[str, str]],
    cost_tracker: dict[str, int] | None = None,
    retrieval: bool = True,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge a claim against provided evidence snippets (or internal memory if retrieval=False).

    Args:
        claim: The atomic factual claim to verify.
        evidence: List of evidence items with 'title', 'url', and 'snippet'.
        cost_tracker: Optional dictionary to increment 'llm_calls'.
        retrieval: If True, judge strictly using evidence. If False, judge from model memory.
        trace: Optional dictionary to capture intermediate inspection data.

    Returns:
        Dict with keys:
            - 'status': "supported" | "contradicted" | "unsupported"
            - 'reasoning': str
            - 'evidence_used': list[str]

    Raises:
        RuntimeError: If LLM call fails.
        ValueError: If response cannot be parsed into required JSON after retry.
    """
    if retrieval:
        # If retrieval is enabled and evidence is empty, return unsupported without calling the LLM
        if not evidence:
            result = {
                "status": "unsupported",
                "reasoning": "No relevant search evidence was found for this claim.",
                "evidence_used": [],
            }
            if trace is not None:
                trace["system_prompt"] = VERIFY_CLAIM_SYSTEM_PROMPT
                trace["user_prompt"] = "(Search returned 0 results; verification skipped)"
                trace["raw_response"] = json.dumps(result)
            return result

        evidence_text = _format_evidence_snippets(evidence)
        available_urls = {item.get("url", "").strip() for item in evidence if item.get("url")}
        system_prompt = VERIFY_CLAIM_SYSTEM_PROMPT
        user_prompt = VERIFY_CLAIM_USER_PROMPT_TEMPLATE.format(
            claim=claim.strip(),
            evidence_text=evidence_text,
        )
    else:
        # No-retrieval baseline: judge strictly using internal parametric memory
        available_urls = set()
        system_prompt = VERIFY_CLAIM_NO_RETRIEVAL_SYSTEM_PROMPT
        user_prompt = VERIFY_CLAIM_NO_RETRIEVAL_USER_PROMPT_TEMPLATE.format(
            claim=claim.strip(),
        )

    if cost_tracker is not None:
        cost_tracker["llm_calls"] = cost_tracker.get("llm_calls", 0) + 1

    try:
        raw_response = call_llm(
            system=system_prompt,
            user=user_prompt,
            temperature=0.0,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM communication error during claim verification: {exc}") from exc

    if trace is not None:
        trace["system_prompt"] = system_prompt
        trace["user_prompt"] = user_prompt
        trace["raw_response"] = raw_response

    try:
        return _clean_and_parse_verification_json(raw_response, available_urls)
    except Exception as first_error:
        # Retry once with stricter prompt
        retry_prompt = VERIFY_CLAIM_RETRY_PROMPT_TEMPLATE.format(
            raw_response=raw_response
        )

        if cost_tracker is not None:
            cost_tracker["llm_calls"] = cost_tracker.get("llm_calls", 0) + 1

        try:
            retry_response = call_llm(
                system=system_prompt,
                user=retry_prompt,
                temperature=0.0,
            )
            if trace is not None:
                trace["retry_prompt"] = retry_prompt
                trace["raw_response"] = retry_response
            return _clean_and_parse_verification_json(retry_response, available_urls)
        except Exception as retry_error:
            raise ValueError(
                f"Failed to parse LLM response into verification JSON schema after retry. "
                f"Initial error: {first_error}; Retry error: {retry_error}"
            ) from retry_error
