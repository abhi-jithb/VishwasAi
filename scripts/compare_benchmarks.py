"""Comparison script to aggregate Vishwas AI benchmark runs into a paper-ready table."""

import argparse
import json
from pathlib import Path
from typing import Any


def load_benchmark_files(results_dir: Path) -> list[dict[str, Any]]:
    """Scan and parse all valid gold benchmark JSON files."""
    if not results_dir.exists():
        return []

    benchmarks: list[dict[str, Any]] = []
    for file_path in sorted(results_dir.glob("gold_*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "metrics" in data and "model" in data:
                    data["_file_name"] = file_path.name
                    benchmarks.append(data)
        except Exception:
            continue

    return benchmarks


def format_comparison_table(benchmarks: list[dict[str, Any]], format_type: str = "table") -> str:
    """Format benchmark metrics into a publication-ready comparative table."""
    if not benchmarks:
        return "No benchmark results found in results/ directory."

    headers = [
        "Model",
        "Provider",
        "Retrieval",
        "Recall (%)",
        "Verif. Acc (%)",
        "E2E Acc (%)",
        "Avg Latency (s)",
        "Correct Acc (%)",
        "Planted Err (%)",
        "Fab. Entity (%)",
    ]

    rows: list[list[str]] = []
    for b in benchmarks:
        metrics = b.get("metrics", {})
        settings = b.get("settings", {})
        cat_stats = metrics.get("category_stats", {})

        def _cat_acc(name: str) -> str:
            stat = cat_stats.get(name, {})
            matched = stat.get("matched", 0)
            agreed = stat.get("agreed", 0)
            if matched > 0:
                return f"{(agreed / matched * 100):.1f}%"
            return "N/A"

        retrieval_status = "ON" if settings.get("retrieval", True) else "OFF (baseline)"
        model_name = b.get("model", "unknown")
        provider = b.get("provider", "unknown")

        recall = f"{metrics.get('claim_extraction_recall', 0.0):.1f}%"
        verif_acc = f"{metrics.get('verification_accuracy', 0.0):.1f}%"
        e2e_acc = f"{metrics.get('end_to_end_accuracy', 0.0):.1f}%"
        avg_latency = f"{metrics.get('avg_latency_per_case_seconds', 0.0):.2f}s"

        correct_acc = _cat_acc("correct")
        planted_acc = _cat_acc("planted_error")
        fabricated_acc = _cat_acc("fabricated_entity")

        rows.append([
            model_name,
            provider,
            retrieval_status,
            recall,
            verif_acc,
            e2e_acc,
            avg_latency,
            correct_acc,
            planted_acc,
            fabricated_acc,
        ])

    if format_type == "markdown":
        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            md_lines.append("| " + " | ".join(row) + " |")
        return "\n".join(md_lines)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    table_lines = [
        sep,
        header_str,
        sep,
    ]
    for row in rows:
        row_str = "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        table_lines.append(row_str)
    table_lines.append(sep)

    return "\n".join(table_lines)


def main() -> None:
    """CLI entrypoint for benchmark comparison."""
    parser = argparse.ArgumentParser(description="Vishwas AI Benchmark Comparison Table Generator")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory containing gold benchmark JSONs")
    parser.add_argument("--markdown", action="store_true", help="Output table in Markdown syntax")
    parser.add_argument("--save", action="store_true", help="Save comparison table to results/benchmark_comparison.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).resolve().parent.parent / "results"
    benchmarks = load_benchmark_files(results_dir)

    print("\n" + "=" * 90)
    print("                    VISHWAS AI MODEL & RETRIEVAL COMPARISON")
    print("=" * 90)
    print(f"Loaded {len(benchmarks)} benchmark run(s) from: {results_dir}\n")

    table_text = format_comparison_table(benchmarks, format_type="markdown" if args.markdown else "table")
    print(table_text)
    print("=" * 90)

    if args.save:
        md_table = format_comparison_table(benchmarks, format_type="markdown")
        save_path = results_dir / "benchmark_comparison.md"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("# Vishwas AI Benchmark Comparison\n\n")
            f.write(md_table + "\n")
        print(f"\n[SAVED] Comparison table written to: {save_path}")


if __name__ == "__main__":
    main()
