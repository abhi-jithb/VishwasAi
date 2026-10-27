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


def _format_inspector_output(result: dict[str, Any]) -> str:
    """Format full intermediate pipeline inspection data into an interactive terminal view."""
    insp = result.get("inspection", {})
    if not insp:
        return _format_terminal_output(result)

    lines: list[str] = [
        "=" * 78,
        "                    VISHWAS AI PIPELINE INSPECTOR TRACE",
        "=" * 78,
        "[STAGE 1: RAW INPUTS]",
        f"  Question:  {insp.get('question', '')}",
        f"  Answer:    {insp.get('answer', '')}",
        f"  Retrieval: {'ON (Web Search Evidence)' if insp.get('retrieval') else 'OFF (Model Parametric Memory)'}",
        "-" * 78,
        "[STAGE 2: ATOMIC CLAIM EXTRACTION]",
        f"  System Prompt:  {insp.get('extraction', {}).get('system_prompt', '')[:120]}...",
        f"  User Prompt:    {insp.get('extraction', {}).get('user_prompt', '').strip()}",
        f"  Raw LLM Output: {insp.get('extraction', {}).get('raw_response', '').strip()}",
        f"  Parsed Claims ({len(insp.get('extraction', {}).get('extracted_claims', []))}):",
    ]

    for idx, cl in enumerate(insp.get("extraction", {}).get("extracted_claims", []), 1):
        lines.append(f"    {idx}. {cl}")

    lines.extend([
        "-" * 78,
        "[STAGE 3: RETRIEVAL & VERIFICATION PIPELINE TRACE]",
    ])

    for idx, ctrace in enumerate(insp.get("claims", []), 1):
        claim_text = ctrace.get("claim", "")
        status = ctrace.get("status", "unsupported")
        badge = STATUS_SYMBOLS.get(status, f"[{status}]")
        query = ctrace.get("search_query", "(none)")
        snippets = ctrace.get("snippets", [])
        judge_sys = ctrace.get("judge_system_prompt", "")
        judge_user = ctrace.get("judge_user_prompt", "")
        judge_out = ctrace.get("judge_raw_output", "")
        reasoning = ctrace.get("reasoning", "")
        sources = ctrace.get("sources", [])

        lines.extend([
            f"\n--- CLAIM #{idx}: \"{claim_text}\" ---",
            f"  * Derived Search Query: {query}",
            f"  * Retrieved Evidence Snippets ({len(snippets)}):",
        ])

        if not snippets:
            lines.append("      (No snippets retrieved)")
        else:
            for s_idx, snip in enumerate(snippets, 1):
                title = snip.get("title", "Untitled")
                url = snip.get("url", "")
                snippet_text = snip.get("snippet", "")
                lines.append(f"      [{s_idx}] {title} ({url})")
                lines.append(f"          Snippet: {snippet_text[:180]}...")

        lines.extend([
            "  * Judge User Prompt Sent:",
            f"      {judge_user.replace(chr(10), chr(10) + '      ')}",
            f"  * Judge Raw Output: {judge_out.strip()}",
            f"  * Verdict:          {badge}",
            f"  * Reasoning:        {reasoning}",
            f"  * Sources Used:     {', '.join(sources) if sources else '(None)'}",
        ])

    summary = result.get("summary", {})
    cost_log = result.get("cost_log", {})
    score = summary.get("factuality_score", 0.0)
    vp = summary.get("verifiable_precision")
    vp_str = f"{vp * 100:.1f}%" if vp is not None else "N/A"
    ur = summary.get("unverifiable_rate", 0.0)

    lines.extend([
        "-" * 78,
        "[STAGE 4: FINAL METRICS]",
        f"  Total Claims:         {summary.get('total', 0)}",
        f"  [✓] Supported:        {summary.get('supported', 0)}",
        f"  [✗] Contradicted:     {summary.get('contradicted', 0)}",
        f"  [?] Unsupported:      {summary.get('unsupported', 0)}",
        f"  Factuality Score:     {score * 100:.1f}%",
        f"  Verifiable Precision: {vp_str}",
        f"  Unverifiable Rate:    {ur * 100:.1f}%",
        f"  LLM Calls:            {cost_log.get('llm_calls', 0)} | Search Calls: {cost_log.get('search_calls', 0)}",
        "=" * 78,
    ])

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
        "--no-retrieval",
        action="store_true",
        help="Run no-retrieval baseline (evaluate using model memory only).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspector mode: display detailed intermediate data of each pipeline stage.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output evaluation results as raw JSON.",
    )

    args = parser.parse_args()

    try:
        result = evaluate(
            question=args.question,
            answer=args.answer,
            retrieval=not args.no_retrieval,
            inspect=args.inspect or args.json,
        )
    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    elif args.inspect:
        print(_format_inspector_output(result))
    else:
        print(_format_terminal_output(result))


if __name__ == "__main__":
    main()
