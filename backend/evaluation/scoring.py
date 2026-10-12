"""Scoring module for Vishwas AI evaluation metrics."""

from typing import Any


def compute_factuality(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall factuality summary statistics from verified claims.

    Args:
        results: List of claim result dicts, each containing a 'status' field.

    Returns:
        Dict with keys:
            - 'supported': int
            - 'contradicted': int
            - 'unsupported': int
            - 'total': int
            - 'factuality_score': float (supported / total, or 0.0 if total == 0)
    """
    supported: int = 0
    contradicted: int = 0
    unsupported: int = 0

    for item in results:
        status = str(item.get("status", "")).strip().lower()
        if status == "supported":
            supported += 1
        elif status == "contradicted":
            contradicted += 1
        else:
            unsupported += 1

    total = len(results)
    score = (supported / total) if total > 0 else 0.0

    return {
        "supported": supported,
        "contradicted": contradicted,
        "unsupported": unsupported,
        "total": total,
        "factuality_score": round(score, 4),
    }
