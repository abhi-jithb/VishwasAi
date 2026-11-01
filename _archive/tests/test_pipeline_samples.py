"""Manual review test runner for Vishwas AI sample evaluation cases."""

import json
from pathlib import Path
import sys
from typing import Any

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.evaluation.pipeline import evaluate


def run_all_samples() -> None:
    """Run all 10 sample cases through the evaluation pipeline and display results."""
    root_dir = Path(__file__).resolve().parent.parent
    samples_path = root_dir / "data" / "samples.json"

    if not samples_path.exists():
        print(f"[ERROR] Samples file not found at {samples_path}", file=sys.stderr)
        sys.exit(1)

    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print("=" * 80)
    print(f"RUNNING VISHWAS AI EVALUATION PIPELINE ON {len(samples)} SAMPLES")
    print("=" * 80)

    summary_rows: list[dict[str, Any]] = []

    for item in samples:
        sample_id = item.get("id")
        category = item.get("category", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        expected = item.get("expected_note", "")

        print(f"\n>>> Running Sample #{sample_id} [{category}]")
        print(f"    Q: {question}")
        print(f"    A: {answer[:90]}...")

        try:
            result = evaluate(question=question, answer=answer)
            claims = result.get("claims", [])
            summary = result.get("summary", {})

            print(f"    Found {len(claims)} claims:")
            for idx, c in enumerate(claims, 1):
                status_icon = (
                    "[✓]" if c["status"] == "supported"
                    else "[✗]" if c["status"] == "contradicted"
                    else "[?]"
                )
                print(f"      {idx}. {status_icon} ({c['status']}): {c['text']}")

            summary_rows.append(
                {
                    "id": sample_id,
                    "category": category,
                    "claims_found": summary.get("total", 0),
                    "supported": summary.get("supported", 0),
                    "contradicted": summary.get("contradicted", 0),
                    "unsupported": summary.get("unsupported", 0),
                    "score": f"{summary.get('factuality_score', 0.0) * 100:.1f}%",
                    "expected": expected,
                }
            )
        except Exception as exc:
            print(f"    [ERROR] Sample #{sample_id} evaluation failed: {exc}")
            summary_rows.append(
                {
                    "id": sample_id,
                    "category": category,
                    "claims_found": 0,
                    "supported": 0,
                    "contradicted": 0,
                    "unsupported": 0,
                    "score": "ERROR",
                    "expected": expected,
                }
            )

    # Print summary review table
    print("\n" + "=" * 80)
    print("                    EVALUATION PIPELINE SAMPLES SUMMARY TABLE")
    print("=" * 80)
    header = f"{'ID':<3} | {'Category':<18} | {'Claims':<6} | {'Supp':<5} | {'Contr':<5} | {'Unsupp':<6} | {'Score':<7}"
    print(header)
    print("-" * len(header))

    for r in summary_rows:
        print(
            f"{r['id']:<3} | {r['category']:<18} | {r['claims_found']:<6} | "
            f"{r['supported']:<5} | {r['contradicted']:<5} | {r['unsupported']:<6} | {r['score']:<7}"
        )

    print("=" * 80)
    print("Review Notes:")
    for r in summary_rows:
        print(f"Sample #{r['id']} ({r['category']}): {r['expected']}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_samples()
