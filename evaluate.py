"""CLI entrypoint for Vishwas AI answer evaluation."""

import argparse
import json
import sys
from typing import Any

from backend.evaluation.pipeline import evaluate

STATUS_SYMBOLS = {
    "supported": "[✓] supported",
    "contradicted": "[✗] contradicted",
    "unsupported": "[?] unsupported",
}


def _format_terminal_output(result: dict[str, Any]) -> str:
    """Format evaluation result dictionary into a human-readable terminal report."""
    lines: list[str] = [
        "=" * 64,
        "                   VISHWAS AI TRUST REPORT",
        "=" * 64,
        f"Question: {result.get('question', '')}",
        f"Answer:   {result.get('answer', '')}",
        "-" * 64,
        f"CLAIMS EVALUATED ({len(result.get('claims', []))} found):",
        "",
    ]

    claims = result.get("claims", [])
    if not claims:
        lines.append("  (No factual claims extracted from the answer)")
        lines.append("")
    else:
        for idx, item in enumerate(claims, 1):
            status = item.get("status", "unsupported")
            badge = STATUS_SYMBOLS.get(status, f"[{status}]")
            text = item.get("text", "")
            reasoning = item.get("reasoning", "")
            sources = item.get("sources", [])

            lines.append(f"{idx}. {badge}: {text}")
            lines.append(f"   Reasoning: {reasoning}")
            if sources:
                lines.append("   Sources:")
                for src in sources:
                    lines.append(f"     - {src}")
            else:
                lines.append("   Sources: (None)")
            lines.append("")

    summary = result.get("summary", {})
    cost_log = result.get("cost_log", {})
    score = summary.get("factuality_score", 0.0)
    vp = summary.get("verifiable_precision")
    vp_str = f"{vp * 100:.1f}%" if vp is not None else "N/A (0 verifiable claims)"
    ur = summary.get("unverifiable_rate", 0.0)
    ur_str = f"{ur * 100:.1f}%"
    interpretation = summary.get("interpretation", "")

    lines.extend(
        [
            "-" * 64,
            "SUMMARY:",
            f"  Total Claims:         {summary.get('total', 0)}",
            f"  [✓] Supported:        {summary.get('supported', 0)}",
            f"  [✗] Contradicted:     {summary.get('contradicted', 0)}",
            f"  [?] Unsupported:      {summary.get('unsupported', 0)}",
            f"  Factuality Score:     {score * 100:.1f}%",
            f"  Verifiable Precision: {vp_str}",
            f"  Unverifiable Rate:    {ur_str}",
            f"  Interpretation:       {interpretation}",
            "",
            "COST LOG:",
            f"  LLM Calls:            {cost_log.get('llm_calls', 0)}",
            f"  Search Calls:         {cost_log.get('search_calls', 0)}",
            "=" * 64,
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Parse CLI arguments and run evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Vishwas AI - Factuality and Trust Evaluation for LLM Answers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        required=True,
        help="The question or prompt given to the LLM.",
    )
    parser.add_argument(
        "--answer",
        "-a",
        type=str,
        required=True,
        help="The answer text to audit and verify.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output evaluation results as raw JSON.",
    )

    args = parser.parse_args()

    try:
        result = evaluate(question=args.question, answer=args.answer)
    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_terminal_output(result))


if __name__ == "__main__":
    main()
