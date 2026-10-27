"""Pipeline orchestrator for Vishwas AI answer evaluation."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from backend.evaluation.claims import extract_claims
from backend.evaluation.evidence import gather_evidence
from backend.evaluation.scoring import compute_factuality
from backend.evaluation.verify import verify_claim


def _process_single_claim(
    claim_text: str,
    retrieval: bool = True,
    inspect: bool = False,
) -> tuple[dict[str, Any], int, int, dict[str, Any] | None]:
    """Gather evidence (if retrieval=True) and verify a single claim."""
    local_cost: dict[str, int] = {"llm_calls": 0, "search_calls": 0}
    trace_info: dict[str, Any] | None = {} if inspect else None

    if retrieval:
        evidence_trace: dict[str, Any] | None = {} if inspect else None
        evidence = gather_evidence(claim=claim_text, max_results=3, cost_tracker=local_cost, trace=evidence_trace)
        if inspect and evidence_trace is not None and trace_info is not None:
            trace_info["search_query"] = evidence_trace.get("search_query", "")
            trace_info["snippets"] = evidence_trace.get("results", [])
    else:
        evidence = []
        if inspect and trace_info is not None:
            trace_info["search_query"] = "(No retrieval - Evaluated from model memory)"
            trace_info["snippets"] = []

    verify_trace: dict[str, Any] | None = {} if inspect else None
    verification = verify_claim(
        claim=claim_text,
        evidence=evidence,
        cost_tracker=local_cost,
        retrieval=retrieval,
        trace=verify_trace,
    )
    if inspect and verify_trace is not None and trace_info is not None:
        trace_info["judge_system_prompt"] = verify_trace.get("system_prompt", "")
        trace_info["judge_user_prompt"] = verify_trace.get("user_prompt", "")
        trace_info["judge_raw_response"] = verify_trace.get("raw_response", "")

    record = {
        "text": claim_text,
        "status": verification.get("status", "unsupported"),
        "reasoning": verification.get("reasoning", ""),
        "sources": verification.get("evidence_used", []),
    }
    return record, local_cost["llm_calls"], local_cost["search_calls"], trace_info


def evaluate(
    question: str,
    answer: str,
    retrieval: bool = True,
    inspect: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end factuality evaluation on a (question, answer) pair.

    Orchestrates:
    1. Extracting atomic factual claims from the answer.
    2. Gathering web search evidence for each claim concurrently (if retrieval=True).
    3. Verifying each claim strictly against its evidence (or internal memory if retrieval=False).
    4. Computing overall summary scores and logging API costs.
    5. Optionally collecting full intermediate inspection traces.

    Args:
        question: The user question or reference prompt.
        answer: The generated answer to audit.
        retrieval: Whether to use external search evidence (True) or model memory (False).
        inspect: Whether to attach full intermediate execution trace for debugging / demo.

    Returns:
        Structured evaluation dict.
    """
    cost_log: dict[str, int] = {"llm_calls": 0, "search_calls": 0}
    extract_trace: dict[str, Any] | None = {} if inspect else None

    # Step 1: Extract atomic factual claims
    raw_claims = extract_claims(
        question=question,
        answer=answer,
        cost_tracker=cost_log,
        trace=extract_trace,
    )

    # Step 2 & 3: Gather evidence and verify each claim concurrently
    claim_records: list[dict[str, Any]] = []
    inspection_claims: list[dict[str, Any]] = []

    if raw_claims:
        worker_fn = lambda c: _process_single_claim(c, retrieval=retrieval, inspect=inspect)
        with ThreadPoolExecutor(max_workers=min(len(raw_claims), 2)) as executor:
            results = list(executor.map(worker_fn, raw_claims))

        for record, llm_count, search_count, trace_info in results:
            claim_records.append(record)
            cost_log["llm_calls"] += llm_count
            cost_log["search_calls"] += search_count
            if inspect and trace_info is not None:
                inspection_claims.append(
                    {
                        "claim": record["text"],
                        "search_query": trace_info.get("search_query", ""),
                        "snippets": trace_info.get("snippets", []),
                        "judge_system_prompt": trace_info.get("judge_system_prompt", ""),
                        "judge_user_prompt": trace_info.get("judge_user_prompt", ""),
                        "judge_raw_output": trace_info.get("judge_raw_response", ""),
                        "status": record["status"],
                        "reasoning": record["reasoning"],
                        "sources": record["sources"],
                    }
                )

    # Step 4: Compute summary metrics
    summary = compute_factuality(claim_records)

    payload: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "claims": claim_records,
        "summary": summary,
        "cost_log": cost_log,
    }

    if inspect:
        payload["inspection"] = {
            "question": question,
            "answer": answer,
            "retrieval": retrieval,
            "extraction": {
                "system_prompt": extract_trace.get("system_prompt", "") if extract_trace else "",
                "user_prompt": extract_trace.get("user_prompt", "") if extract_trace else "",
                "raw_response": extract_trace.get("raw_response", "") if extract_trace else "",
                "extracted_claims": raw_claims,
            },
            "claims": inspection_claims,
            "summary": summary,
        }

    return payload
