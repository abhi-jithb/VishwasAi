"""Pipeline orchestrator for Vishwas AI answer evaluation."""

from typing import Any

from backend.evaluation.claims import extract_claims
from backend.evaluation.evidence import gather_evidence
from backend.evaluation.scoring import compute_factuality
from backend.evaluation.verify import verify_claim


def evaluate(question: str, answer: str) -> dict[str, Any]:
    """Execute end-to-end factuality evaluation on a (question, answer) pair.

    Orchestrates:
    1. Extracting atomic factual claims from the answer.
    2. Gathering web search evidence for each claim.
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
                "factuality_score": float
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

    # Step 2 & 3: Gather evidence and verify each claim
    claim_records: list[dict[str, Any]] = []
    for claim_text in raw_claims:
        evidence = gather_evidence(claim=claim_text, max_results=5, cost_tracker=cost_log)
        verification = verify_claim(claim=claim_text, evidence=evidence, cost_tracker=cost_log)

        claim_records.append(
            {
                "text": claim_text,
                "status": verification.get("status", "unsupported"),
                "reasoning": verification.get("reasoning", ""),
                "sources": verification.get("evidence_used", []),
            }
        )

    # Step 4: Compute summary metrics
    summary = compute_factuality(claim_records)

    return {
        "question": question,
        "answer": answer,
        "claims": claim_records,
        "summary": summary,
        "cost_log": cost_log,
    }
