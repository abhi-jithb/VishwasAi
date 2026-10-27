"""Error analysis tool for Vishwas AI benchmark runs.

Categorizes and analyzes every misclassification into:
1. Bad Extraction
2. Bad Search Results
3. Judge Error
4. Ambiguous Gold Label
"""

import argparse
import json
from pathlib import Path
from typing import Any


def classify_error(case: dict[str, Any], disagreement: str, is_missed: bool) -> tuple[str, str]:
    """Classify the root cause of an error based on extracted claims and search evidence."""
    if is_missed:
        extracted = case.get("extracted_claims", [])
        if not extracted:
            return "Bad Extraction", "No claims were extracted from the answer."
        return (
            "Bad Extraction",
            "Atomic fact was omitted during decomposition or merged into a compound sentence.",
        )

    # Disagreement analysis
    lower_dis = disagreement.lower()
    if "pipeline: [unsupported]" in lower_dis:
        # Check if sources were empty
        return (
            "Bad Search Results",
            "Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.",
        )
    elif "pipeline: [supported] vs gold: [contradicted]" in lower_dis:
        return (
            "Judge Error",
            "Model incorrectly evaluated false claim as true due to semantic confusion or noisy snippet matching.",
        )
    elif "pipeline: [contradicted] vs gold: [supported]" in lower_dis:
        return (
            "Judge Error",
            "Judge misapplied strictness criteria and marked true claim as contradicted.",
        )
    elif "ambiguous" in lower_dis or "disputed" in lower_dis:
        return (
            "Ambiguous Gold Label",
            "Gold ground-truth label relies on subjective interpretation or temporal nuance.",
        )

    return "Judge Error", "LLM reasoning divergence during zero-temperature verification."


def generate_error_analysis_report(benchmark_file: Path) -> str:
    """Analyze all errors in a benchmark JSON file and generate a Markdown report."""
    with open(benchmark_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = data.get("model", "unknown")
    provider = data.get("provider", "unknown")
    timestamp = data.get("timestamp", "unknown")
    cases = data.get("cases", [])
    metrics = data.get("metrics", {})

    lines: list[str] = [
        f"# Vishwas AI: Benchmark Error Analysis Report",
        f"",
        f"- **Model Evaluated:** `{model_name}` (`{provider}`)",
        f"- **Timestamp:** `{timestamp}`",
        f"- **Total Cases:** {metrics.get('total_cases', len(cases))}",
        f"- **Gold Claims:** {metrics.get('total_gold_claims', 0)}",
        f"- **Matched Claims:** {metrics.get('total_matched_claims', 0)} ({metrics.get('claim_extraction_recall', 0)}% recall)",
        f"- **Verification Accuracy:** {metrics.get('verification_accuracy', 0)}%",
        f"- **End-to-End Accuracy:** {metrics.get('end_to_end_accuracy', 0)}%",
        f"",
        f"---",
        f"",
        f"## 1. Error Taxonomy & Distribution",
        f"",
        f"Every error in the evaluation pipeline belongs to one of four failure modes:",
        f"1. **`[Bad Extraction]`**: The claim extractor failed to isolate the atomic claim, merged multiple checkable assertions, or omitted the fact.",
        f"2. **`[Bad Search Results]`**: The web retrieval query was too vague or returned irrelevant / empty snippets, starving the judge of evidence.",
        f"3. **`[Judge Error]`**: Sufficient evidence was present, but the model failed semantic entailment or misapplied factual rigor.",
        f"4. **`[Ambiguous Gold Label]`**: The claim itself has debatable ground truth or relies on outdated temporal definitions.",
        f"",
    ]

    error_counts: dict[str, int] = {
        "Bad Extraction": 0,
        "Bad Search Results": 0,
        "Judge Error": 0,
        "Ambiguous Gold Label": 0,
    }

    error_items: list[dict[str, Any]] = []

    for c in cases:
        case_id = c.get("case_id")
        cat = c.get("category", "general")
        q = c.get("question", "")
        a = c.get("answer", "")
        missed = c.get("missed_claims", [])
        disagreed = c.get("disagreements", [])

        for m in missed:
            cause, explanation = classify_error(c, m, is_missed=True)
            error_counts[cause] += 1
            error_items.append({
                "case_id": case_id,
                "category": cat,
                "type": "Missed Claim",
                "claim": m,
                "cause": cause,
                "explanation": explanation,
                "question": q,
            })

        for d in disagreed:
            cause, explanation = classify_error(c, d, is_missed=False)
            error_counts[cause] += 1
            error_items.append({
                "case_id": case_id,
                "category": cat,
                "type": "Status Disagreement",
                "claim": d,
                "cause": cause,
                "explanation": explanation,
                "question": q,
            })

    total_errors = sum(error_counts.values())

    lines.extend([
        f"### Summary of Failure Modes",
        f"",
        f"| Failure Mode | Count | Share (%) | Primary Mitigation Strategy |",
        f"| :--- | :---: | :---: | :--- |",
    ])

    mitigations = {
        "Bad Extraction": "Few-shot structured prompting & grammar-guided JSON schemas",
        "Bad Search Results": "Multi-query search expansion & Wikipedia fallback API",
        "Judge Error": "NLI Cross-Encoder verifier (Stage C) & Chain-of-Thought reasoning",
        "Ambiguous Gold Label": "Dataset curation & precision qualification",
    }

    for cause, count in error_counts.items():
        pct = (count / total_errors * 100) if total_errors > 0 else 0.0
        lines.append(f"| **{cause}** | {count} | {pct:.1f}% | {mitigations[cause]} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Granular Case-by-Case Breakdown",
        f"",
    ])

    for idx, item in enumerate(error_items, 1):
        lines.extend([
            f"### #{idx}. Case {item['case_id']} [{item['category']}] - {item['cause']}",
            f"- **Error Type:** `{item['type']}`",
            f"- **Details:** {item['claim']}",
            f"- **Question:** *\"{item['question']}\"*",
            f"- **Root Cause Analysis:** {item['explanation']}",
            f"",
        ])

    return "\n".join(lines)


def main() -> None:
    """CLI runner to generate error analysis report."""
    parser = argparse.ArgumentParser(description="Generate Vishwas AI Error Analysis Report")
    parser.add_argument("--benchmark", type=str, default=None, help="Path to gold benchmark JSON file")
    parser.add_argument("--output", type=str, default=None, help="Path to save markdown report")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / "results"

    if args.benchmark:
        bench_file = Path(args.benchmark)
    else:
        # Default to latest gold benchmark file
        bench_files = sorted(results_dir.glob("gold_*.json"))
        if not bench_files:
            print(f"[ERROR] No benchmark files found in {results_dir}")
            return
        bench_file = bench_files[-1]

    print(f"Generating error analysis for: {bench_file}")
    report_md = generate_error_analysis_report(bench_file)

    out_file = Path(args.output) if args.output else results_dir / "error_analysis.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[SAVED] Error analysis report generated at: {out_file}")


if __name__ == "__main__":
    main()
