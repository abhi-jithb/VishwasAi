"""Pipeline orchestrator for Vishwas AI answer evaluation."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from backend.evaluation.claims import extract_claims
from backend.evaluation.evidence import gather_evidence
from backend.evaluation.scoring import compute_factuality
from backend.evaluation.verify import verify_claim


def _process_single_claim(claim_text: str) -> tuple[dict[str, Any], int, int]:
    """Gather evidence and verify a single claim, returning record and call counts."""
    local_cost: dict[str, int] = {"llm_calls": 0, "search_calls": 0}
    evidence = gather_evidence(claim=claim_text, max_results=3, cost_tracker=local_cost)
    verification = verify_claim(claim=claim_text, evidence=evidence, cost_tracker=local_cost)

    record = {
        "text": claim_text,
        "status": verification.get("status", "unsupported"),
        "reasoning": verification.get("reasoning", ""),
        "sources": verification.get("evidence_used", []),
    }
    return record, local_cost["llm_calls"], local_cost["search_calls"]


def evaluate(question: str, answer: str) -> dict[str, Any]:
    """Execute end-to-end factuality evaluation on a (question, answer) pair.

    Orchestrates:
    1. Extracting atomic factual claims from the answer.
    2. Gathering web search evidence for each claim concurrently.
    3. Verifying each claim strictly against its evidence.
    4. Computing overall summary scores and logging API costs.

    Args:
        question: The user question or reference prompt.
        answer: The generated answer to audit.

    Returns:
        Structured evaluation dict:
        {
            "question": str,
            "answer": str,
            "claims": [
                {
                    "text": str,
                    "status": "supported" | "contradicted" | "unsupported",
                    "reasoning": str,
                    "sources": list[str]
                }
            ],
            "summary": {
                "supported": int,
                "contradicted": int,
                "unsupported": int,
                "total": int,
                "factuality_score": float,
                "verifiable_precision": float | None,
                "unverifiable_rate": float,
                "interpretation": str
            },
            "cost_log": {
                "llm_calls": int,
                "search_calls": int
            }
        }
    """
    cost_log: dict[str, int] = {"llm_calls": 0, "search_calls": 0}

    # Step 1: Extract atomic factual claims
    raw_claims = extract_claims(question=question, answer=answer, cost_tracker=cost_log)

    # Step 2 & 3: Gather evidence and verify each claim concurrently
    claim_records: list[dict[str, Any]] = []

    if raw_claims:
        with ThreadPoolExecutor(max_workers=min(len(raw_claims), 2)) as executor:
            results = list(executor.map(_process_single_claim, raw_claims))

        for record, llm_count, search_count in results:
            claim_records.append(record)
            cost_log["llm_calls"] += llm_count
            cost_log["search_calls"] += search_count

    # Step 4: Compute summary metrics
    summary = compute_factuality(claim_records)

    return {
        "question": question,
        "answer": answer,
        "claims": claim_records,
        "summary": summary,
        "cost_log": cost_log,
    }
