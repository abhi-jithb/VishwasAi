import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.config import get_active_model, get_active_provider
from backend.evaluation.pipeline import evaluate

try:
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding.lower() != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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


def _detect_system_hardware() -> dict[str, Any]:
    """Detect local runtime hardware specifications."""
    import platform
    hw = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": 16,
        "gpu": "CPU / Integrated",
    }
    try:
        import psutil
        hw["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3))
    except Exception:
        pass
    try:
        import subprocess
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=2.0,
        ).strip()
        if gpu_info:
            hw["gpu"] = gpu_info.split("\n")[0]
    except Exception:
        pass
    return hw


def run_gold_evaluation(
    limit: int | None = None,
    retrieval: bool = True,
    save_results: bool = True,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run pipeline against data/gold_claims.json and evaluate extraction & verification accuracy."""
    gold_path = root_dir / "data" / "gold_claims.json"
    if not gold_path.exists():
        print(f"[ERROR] Gold claims file not found at {gold_path}", file=sys.stderr)
        sys.exit(1)

    with open(gold_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if limit is not None:
        test_cases = test_cases[:limit]

    model_name = get_active_model()
    provider_name = get_active_provider()
    mode_label = "WITH RETRIEVAL" if retrieval else "NO-RETRIEVAL BASELINE (MEMORY ONLY)"

    print("=" * 80)
    print(f"RUNNING VISHWAS AI GOLD BENCHMARK ({len(test_cases)} CASES)")
    print(f"Model: {model_name} | Provider: {provider_name} | Mode: {mode_label}")
    print("=" * 80)

    total_gold_claims = 0
    total_matched_claims = 0
    total_missed_claims = 0
    total_agreed_status = 0
    total_disagreed_status = 0

    category_stats: dict[str, dict[str, int]] = {}
    detailed_cases: list[dict[str, Any]] = []

    bench_start_time = time.time()

    for case in test_cases:
        case_id = case.get("id")
        category = case.get("category", "general")
        question = case.get("question", "")
        answer = case.get("answer", "")
        gold_claims = case.get("gold_claims", [])

        if category not in category_stats:
            category_stats[category] = {
                "gold": 0, "matched": 0, "agreed": 0, "disagreed": 0, "missed": 0
            }

        total_gold_claims += len(gold_claims)
        category_stats[category]["gold"] += len(gold_claims)

        print(f"\n>>> Case #{case_id} [{category}]", flush=True)
        print(f"    Q: {question}", flush=True)
        print(f"    A: {answer[:100]}..." if len(answer) > 100 else f"    A: {answer}", flush=True)

        # Run pipeline evaluation with per-case latency timing
        case_t0 = time.time()
        try:
            eval_result = evaluate(question=question, answer=answer, retrieval=retrieval)
            extracted = eval_result.get("claims", [])
            cost_log = eval_result.get("cost_log", {})
        except Exception as exc:
            print(f"    [ERROR] Evaluation execution failed: {exc}", flush=True)
            extracted = []
            cost_log = {}
        case_duration = time.time() - case_t0

        print(f"    Extracted {len(extracted)} claim(s) in {case_duration:.2f}s:", flush=True)
        for i, c in enumerate(extracted, 1):
            badge = "[✓]" if c.get("status") == "supported" else "[✗]" if c.get("status") == "contradicted" else "[?]"
            print(f"      {i}. {badge} ({c.get('status')}): {c.get('text')}", flush=True)

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
                sim = _claim_similarity(g_text, ext.get("text", ""))
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            # Similarity threshold for match
            if best_sim >= 0.35 and best_idx != -1:
                matched_extracted_indices.add(best_idx)
                ext_claim = extracted[best_idx]
                ext_status = ext_claim.get("status", "unsupported").lower()

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
                        f"(Matched: '{ext_claim.get('text')}')"
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

        detailed_cases.append(
            {
                "case_id": case_id,
                "category": category,
                "question": question,
                "answer": answer,
                "gold_claims": gold_claims,
                "extracted_claims": extracted,
                "agreements": agreements,
                "disagreements": disagreements,
                "missed_claims": missed_list,
                "latency_seconds": round(case_duration, 2),
                "cost_log": cost_log,
            }
        )

    total_bench_duration = time.time() - bench_start_time
    avg_latency = total_bench_duration / len(test_cases) if test_cases else 0.0

    # Print overall benchmark scorecard
    print("\n" + "=" * 80)
    print("                     GOLD BENCHMARK EVALUATION REPORT")
    print("=" * 80)
    print(f"Model:                       {model_name}")
    print(f"Provider:                    {provider_name}")
    print(f"Total Test Cases:            {len(test_cases)}")
    print(f"Total Benchmark Time:        {total_bench_duration:.2f}s (avg {avg_latency:.2f}s/case)")
    print(f"Total Gold Claims:           {total_gold_claims}")
    recall_pct = (total_matched_claims / total_gold_claims * 100) if total_gold_claims > 0 else 0.0
    print(f"Total Matched Claims:         {total_matched_claims} ({recall_pct:.1f}% recall)")
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

    # Compile structured benchmark payload
    now_iso = datetime.now().isoformat()
    clean_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = re.sub(r"[^a-zA-Z0-9_\.]", "_", model_name)

    benchmark_record = {
        "benchmark": "Vishwas AI Gold Benchmark",
        "timestamp": now_iso,
        "model": model_name,
        "provider": provider_name,
        "hardware": _detect_system_hardware(),
        "settings": {
            "temperature": 0.0,
            "retrieval": retrieval,
            "max_search_results": 3 if retrieval else 0,
        },
        "metrics": {
            "total_cases": len(test_cases),
            "total_gold_claims": total_gold_claims,
            "total_matched_claims": total_matched_claims,
            "claim_extraction_recall": round(recall_pct, 2),
            "missed_gold_claims": total_missed_claims,
            "status_agreed": total_agreed_status,
            "status_disagreed": total_disagreed_status,
            "verification_accuracy": round(overall_accuracy, 2),
            "end_to_end_accuracy": round(end_to_end_accuracy, 2),
            "total_latency_seconds": round(total_bench_duration, 2),
            "avg_latency_per_case_seconds": round(avg_latency, 2),
            "category_stats": category_stats,
        },
        "cases": detailed_cases,
    }

    if save_results:
        results_dir = root_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        retrieval_tag = "" if retrieval else "_no_retrieval"
        if output_path:
            save_file = Path(output_path)
        else:
            save_file = results_dir / f"gold_{model_slug}{retrieval_tag}_{clean_timestamp}.json"
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_record, f, indent=2)
        print(f"\n[SAVED] Benchmark results saved to: {save_file}")

    return benchmark_record


def main() -> None:
    """Parse CLI options and run gold evaluation."""
    parser = argparse.ArgumentParser(description="Vishwas AI Gold Benchmark Runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases to run")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model name")
    parser.add_argument("--provider", type=str, default=None, help="Override LLM provider (e.g. ollama, groq)")
    parser.add_argument("--no-retrieval", action="store_true", help="Run no-retrieval baseline (parametric memory only)")
    parser.add_argument("--output", type=str, default=None, help="Specific JSON file to save results")
    parser.add_argument("--no-save", action="store_true", help="Do not save results file")
    args = parser.parse_args()

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["BENCHMARK_MODEL"] = args.model
        os.environ["LLM_MODEL"] = args.model

    run_gold_evaluation(
        limit=args.limit,
        retrieval=not args.no_retrieval,
        save_results=not args.no_save,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
