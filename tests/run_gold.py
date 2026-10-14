"""Evaluation runner against gold benchmark claims dataset."""

import json
from pathlib import Path
import re
import sys
from typing import Any

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.evaluation.pipeline import evaluate

STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "was", "were", "by", "as", "with", "it", "its", "that", "this",
    "from", "first", "also", "into", "he", "she", "they", "his", "her"
}


def _claim_similarity(s1: str, s2: str) -> float:
    """Compute token-based similarity score between two claim texts."""
    w1 = set(re.findall(r"\w+", s1.lower())) - STOP_WORDS
    w2 = set(re.findall(r"\w+", s2.lower())) - STOP_WORDS
    if not w1 or not w2:
        return 0.0
    intersection = len(w1 & w2)
    union = len(w1 | w2)
    jaccard = intersection / union
    overlap = intersection / min(len(w1), len(w2))
    return 0.4 * jaccard + 0.6 * overlap


def run_gold_evaluation(limit: int | None = None) -> None:
    """Run pipeline against data/gold_claims.json and evaluate extraction & verification accuracy."""
    gold_path = root_dir / "data" / "gold_claims.json"
    if not gold_path.exists():
        print(f"[ERROR] Gold claims file not found at {gold_path}", file=sys.stderr)
        sys.exit(1)

    with open(gold_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if limit is not None:
        test_cases = test_cases[:limit]

    print("=" * 80)
    print(f"RUNNING VISHWAS AI GOLD EVALUATION BENCHMARK ({len(test_cases)} CASES)")
    print("=" * 80)

    total_gold_claims = 0
    total_matched_claims = 0
    total_missed_claims = 0
    total_agreed_status = 0
    total_disagreed_status = 0

    category_stats: dict[str, dict[str, int]] = {}

    for case in test_cases:
        case_id = case.get("id")
        category = case.get("category", "general")
        question = case.get("question", "")
        answer = case.get("answer", "")
        gold_claims = case.get("gold_claims", [])

        if category not in category_stats:
            category_stats[category] = {"gold": 0, "matched": 0, "agreed": 0, "disagreed": 0, "missed": 0}

        total_gold_claims += len(gold_claims)
        category_stats[category]["gold"] += len(gold_claims)

        print(f"\n>>> Case #{case_id} [{category}]", flush=True)
        print(f"    Q: {question}", flush=True)
        print(f"    A: {answer[:100]}..." if len(answer) > 100 else f"    A: {answer}", flush=True)

        # Run pipeline evaluation
        try:
            eval_result = evaluate(question=question, answer=answer)
            extracted = eval_result.get("claims", [])
        except Exception as exc:
            print(f"    [ERROR] Evaluation execution failed: {exc}", flush=True)
            extracted = []

        print(f"    Extracted {len(extracted)} claim(s):", flush=True)
        for i, c in enumerate(extracted, 1):
            badge = "[✓]" if c["status"] == "supported" else "[✗]" if c["status"] == "contradicted" else "[?]"
            print(f"      {i}. {badge} ({c['status']}): {c['text']}", flush=True)

        # Match gold claims with extracted claims
        matched_extracted_indices: set[int] = set()
        case_agreed = 0
        case_disagreed = 0
        case_missed = 0

        disagreements: list[str] = []
        agreements: list[str] = []
        missed_list: list[str] = []

        for g in gold_claims:
            g_text = g["text"]
            g_label = g["label"].lower()

            best_sim = 0.0
            best_idx = -1

            for idx, ext in enumerate(extracted):
                if idx in matched_extracted_indices:
                    continue
                sim = _claim_similarity(g_text, ext["text"])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            # Similarity threshold for match
            if best_sim >= 0.35 and best_idx != -1:
                matched_extracted_indices.add(best_idx)
                ext_claim = extracted[best_idx]
                ext_status = ext_claim["status"].lower()

                if ext_status == g_label:
                    total_agreed_status += 1
                    case_agreed += 1
                    category_stats[category]["agreed"] += 1
                    agreements.append(f"✓ '{g_text}' -> status: {ext_status}")
                else:
                    total_disagreed_status += 1
                    case_disagreed += 1
                    category_stats[category]["disagreed"] += 1
                    disagreements.append(
                        f"✗ '{g_text}' -> Pipeline: [{ext_status}] vs Gold: [{g_label}] "
                        f"(Matched: '{ext_claim['text']}')"
                    )

                total_matched_claims += 1
                category_stats[category]["matched"] += 1
            else:
                total_missed_claims += 1
                case_missed += 1
                category_stats[category]["missed"] += 1
                missed_list.append(f"• Missed gold claim: '{g_text}' (Gold: {g_label})")

        # Report case-level comparison
        if missed_list:
            print("    Missed Gold Claims:", flush=True)
            for m in missed_list:
                print(f"      {m}", flush=True)

        if disagreements:
            print("    Status Disagreements:", flush=True)
            for d in disagreements:
                print(f"      {d}", flush=True)

        if not disagreements and not missed_list:
            print("    Status: Perfect match with all gold labels.", flush=True)

    # Print overall benchmark scorecard
    print("\n" + "=" * 80)
    print("                     GOLD BENCHMARK EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Gold Claims:           {total_gold_claims}")
    print(f"Total Matched Claims:         {total_matched_claims} ({(total_matched_claims / total_gold_claims * 100):.1f}% recall)")
    print(f"Missed Gold Claims:           {total_missed_claims}")
    print(f"Status Agreed (Correct):      {total_agreed_status}")
    print(f"Status Disagreed:             {total_disagreed_status}")

    overall_accuracy = (total_agreed_status / total_matched_claims * 100) if total_matched_claims > 0 else 0.0
    end_to_end_accuracy = (total_agreed_status / total_gold_claims * 100) if total_gold_claims > 0 else 0.0

    print(f"Verification Accuracy (Matched): {overall_accuracy:.1f}%")
    print(f"End-to-End Accuracy (All Gold):  {end_to_end_accuracy:.1f}%")

    print("\n" + "-" * 80)
    print(f"{'Category':<20} | {'Gold':<5} | {'Matched':<7} | {'Missed':<6} | {'Agreed':<6} | {'Accuracy':<8}")
    print("-" * 80)
    for cat, stats in category_stats.items():
        cat_matched = stats["matched"]
        cat_agreed = stats["agreed"]
        cat_acc = (cat_agreed / cat_matched * 100) if cat_matched > 0 else 0.0
        print(f"{cat:<20} | {stats['gold']:<5} | {cat_matched:<7} | {stats['missed']:<6} | {cat_agreed:<6} | {cat_acc:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_gold_evaluation()
