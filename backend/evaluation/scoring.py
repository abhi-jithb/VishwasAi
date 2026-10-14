"""Scoring module for Vishwas AI evaluation metrics."""

from typing import Any


def get_evaluation_interpretation(summary: dict[str, Any]) -> str:
    """Generate a concise one-line human interpretation of the evaluation summary."""
    total: int = summary.get("total", 0)
    supported: int = summary.get("supported", 0)
    contradicted: int = summary.get("contradicted", 0)
    unsupported: int = summary.get("unsupported", 0)

    if total == 0:
        return "No factual claims could be extracted from the answer."

    verifiable = supported + contradicted
    if verifiable == 0:
        return "0 claims could be verified against public evidence (all claims unsupported)."

    if contradicted == 0 and unsupported == 0:
        return "All claims are fully verified and supported by evidence."

    if contradicted == 0:
        return (
            f"All {supported} verifiable claim(s) are supported, "
            f"though {unsupported} claim(s) lack public evidence."
        )

    if supported == 0:
        return f"All {contradicted} verifiable claim(s) are contradicted by evidence."

    return (
        f"Mixed accuracy: {supported} supported, "
        f"{contradicted} contradicted, and {unsupported} unsupported claim(s)."
    )


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
            - 'verifiable_precision': float | None (supported / (supported + contradicted), or None)
            - 'unverifiable_rate': float (unsupported / total, or 0.0 if total == 0)
            - 'interpretation': str
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

    verifiable_total = supported + contradicted
    if verifiable_total > 0:
        verifiable_precision: float | None = round(supported / verifiable_total, 4)
    else:
        verifiable_precision = None

    unverifiable_rate: float = round(unsupported / total, 4) if total > 0 else 0.0

    summary: dict[str, Any] = {
        "supported": supported,
        "contradicted": contradicted,
        "unsupported": unsupported,
        "total": total,
        "factuality_score": round(score, 4),
        "verifiable_precision": verifiable_precision,
        "unverifiable_rate": unverifiable_rate,
    }

    summary["interpretation"] = get_evaluation_interpretation(summary)
    return summary
