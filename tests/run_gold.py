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
from backend.llm import get_models_used, model_fallback_disabled

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


SIMILARITY_THRESHOLD = 0.35
LABELS = ["supported", "contradicted", "unsupported"]


def match_gold_to_extracted(
    gold_claims: list[dict[str, Any]],
    extracted: list[dict[str, Any]],
) -> list[tuple[int, float]]:
    """Greedy one-to-one matching of gold claims to extracted claims.

    Gold claims are processed in file order; each takes the most similar extracted
    claim that is still unused. Returns (extracted_index or -1, best_similarity) per gold claim.
    """
    used: set[int] = set()
    matches: list[tuple[int, float]] = []
    for g in gold_claims:
        best_sim = 0.0
        best_idx = -1
        for idx, ext in enumerate(extracted):
            if idx in used:
                continue
            sim = _claim_similarity(g["text"], ext.get("text", ""))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        # Similarity threshold for match
        if best_sim >= SIMILARITY_THRESHOLD and best_idx != -1:
            used.add(best_idx)
            matches.append((best_idx, best_sim))
        else:
            matches.append((-1, best_sim))
    return matches


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Build a confusion matrix (rows gold, columns predicted) from (gold, predicted) label pairs."""
    matrix = {g: {p: 0 for p in LABELS} for g in LABELS}
    for gold, pred in pairs:
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


def per_class_prf(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float | None]]:
    """Per-class precision, recall and F1 (as percentages) from a confusion matrix."""
    out: dict[str, dict[str, float | None]] = {}
    for label in LABELS:
        tp = matrix[label][label]
        predicted = sum(matrix[g][label] for g in LABELS)
        actual = sum(matrix[label].values())
        precision = tp / predicted if predicted else None
        recall = tp / actual if actual else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1: float | None = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        out[label] = {
            "precision": round(precision * 100, 2) if precision is not None else None,
            "recall": round(recall * 100, 2) if recall is not None else None,
            "f1": round(f1 * 100, 2) if f1 is not None else None,
            "support": actual,
        }
    return out


def _run_cmd(cmd: list[str], timeout: float = 5.0) -> str:
    import subprocess
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _detect_system_hardware() -> dict[str, Any]:
    """Read hardware specifications from the real machine (nothing hard-coded)."""
    import platform
    hw: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads": os.cpu_count(),
        "ram_gb": None,
        "gpu": None,
    }
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                pretty = line.split("=", 1)[1].strip('"')
                hw["os"] = f"{pretty} ({platform.release()})"
    except Exception:
        pass
    for line in _run_cmd(["lscpu"]).splitlines():
        if line.startswith("Model name:"):
            hw["cpu"] = line.split(":", 1)[1].strip()
    for line in _run_cmd(["free", "-b"]).splitlines():
        if line.startswith("Mem:"):
            hw["ram_gb"] = round(int(line.split()[1]) / (1024 ** 3), 1)
    gpu_info = _run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    if gpu_info:
        hw["gpu"] = gpu_info.split("\n")[0]
    return hw


def _ollama_model_info(model: str) -> dict[str, str]:
    """Parse `ollama show <model>` for architecture, parameter count and quantization."""
    info: dict[str, str] = {}
    for line in _run_cmd(["ollama", "show", model], timeout=15.0).splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("architecture", "parameters", "quantization"):
            info[parts[0]] = parts[1]
    version = _run_cmd(["ollama", "--version"])
    if version:
        info["ollama_version"] = version
    return info


def _git_commit() -> dict[str, Any]:
    commit = _run_cmd(["git", "-C", str(root_dir), "rev-parse", "HEAD"])
    dirty = bool(_run_cmd(["git", "-C", str(root_dir), "status", "--porcelain", "--untracked-files=no"]))
    return {"commit": commit, "dirty_worktree": dirty}


def _compact_trace(inspection: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep the evidence and raw judge output from an inspection trace (drop constant prompts)."""
    if not inspection:
        return None
    return {
        "extraction_raw_response": inspection.get("extraction", {}).get("raw_response", ""),
        "claims": [
            {
                "claim": c.get("claim", ""),
                "search_query": c.get("search_query", ""),
                "snippets": c.get("snippets", []),
                "judge_raw_output": c.get("judge_raw_output", ""),
            }
            for c in inspection.get("claims", [])
        ],
    }


def run_gold_evaluation(
    limit: int | None = None,
    retrieval: bool = True,
    save_results: bool = True,
    output_path: str | None = None,
    cold: bool = False,
    case_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Run pipeline against data/gold_claims.json and evaluate extraction & verification accuracy."""
    gold_path = root_dir / "data" / "gold_claims.json"
    if not gold_path.exists():
        print(f"[ERROR] Gold claims file not found at {gold_path}", file=sys.stderr)
        sys.exit(1)

    with open(gold_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if case_ids is not None:
        test_cases = [c for c in test_cases if c.get("id") in case_ids]
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
    total_extracted_claims = 0
    total_llm_calls = 0
    total_search_calls = 0
    label_pairs: list[tuple[str, str]] = []
    failed_case_ids: list[Any] = []
    search_provider_counts: dict[str, int] = {}

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
        case_error: str | None = None
        trace: dict[str, Any] | None = None
        try:
            # inspect=True only records traces (evidence snippets, raw judge output); it does not change verdicts
            eval_result = evaluate(question=question, answer=answer, retrieval=retrieval, inspect=True)
            extracted = eval_result.get("claims", [])
            cost_log = eval_result.get("cost_log", {})
            trace = _compact_trace(eval_result.get("inspection"))
        except Exception as exc:
            print(f"    [ERROR] Evaluation execution failed: {exc}", flush=True)
            extracted = []
            cost_log = {}
            case_error = str(exc)
            failed_case_ids.append(case_id)
        case_duration = time.time() - case_t0
        total_extracted_claims += len(extracted)
        total_llm_calls += cost_log.get("llm_calls", 0)
        total_search_calls += cost_log.get("search_calls", 0)
        for tc in (trace or {}).get("claims", []):
            for snip in tc.get("snippets", []):
                prov = snip.get("provider", "unknown (cached before provider field)")
                search_provider_counts[prov] = search_provider_counts.get(prov, 0) + 1

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

        label_comparison: list[dict[str, Any]] = []
        for g, (best_idx, best_sim) in zip(gold_claims, match_gold_to_extracted(gold_claims, extracted)):
            g_text = g["text"]
            g_label = g["label"].lower()

            if best_idx != -1:
                matched_extracted_indices.add(best_idx)
                ext_claim = extracted[best_idx]
                ext_status = ext_claim.get("status", "unsupported").lower()
                label_pairs.append((g_label, ext_status))
                label_comparison.append(
                    {
                        "gold_text": g_text,
                        "gold_label": g_label,
                        "matched_extracted_index": best_idx,
                        "matched_text": ext_claim.get("text", ""),
                        "similarity": round(best_sim, 4),
                        "predicted_label": ext_status,
                        "agreed": ext_status == g_label,
                    }
                )

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
                label_comparison.append(
                    {
                        "gold_text": g_text,
                        "gold_label": g_label,
                        "matched_extracted_index": None,
                        "matched_text": None,
                        "similarity": round(best_sim, 4),
                        "predicted_label": None,
                        "agreed": False,
                    }
                )

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
                "num_extracted_claims": len(extracted),
                "num_extra_claims": len(extracted) - len(matched_extracted_indices),
                "label_comparison": label_comparison,
                "error": case_error,
                "trace": trace,
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
    matrix = confusion_matrix(label_pairs)
    if failed_case_ids:
        print(f"[WARNING] {len(failed_case_ids)} case(s) failed: {failed_case_ids}. Re-run them with --case-ids.")

    benchmark_record = {
        "benchmark": "Vishwas AI Gold Benchmark",
        "timestamp": now_iso,
        "model": model_name,
        "provider": provider_name,
        "models_used": get_models_used(),
        "model_fallback_disabled": model_fallback_disabled(),
        "hardware": _detect_system_hardware(),
        "run_info": {
            "git": _git_commit(),
            "cold": cold,
            "cache_disabled": os.getenv("DISABLE_CACHE", "0").lower() in ("1", "true", "yes"),
            "model_info": _ollama_model_info(model_name) if provider_name == "ollama" else {},
            "command": " ".join(sys.argv),
            "case_ids": sorted(case_ids) if case_ids is not None else None,
            "failed_case_ids": failed_case_ids,
            "search_provider_counts": search_provider_counts,
        },
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
            "total_extracted_claims": total_extracted_claims,
            "extra_extracted_claims": total_extracted_claims - total_matched_claims,
            "total_llm_calls": total_llm_calls,
            "total_search_calls": total_search_calls,
            "confusion_matrix_matched": matrix,
            "per_class_metrics": per_class_prf(matrix),
        },
        "cases": detailed_cases,
    }

    if save_results:
        results_dir = root_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        retrieval_tag = "" if retrieval else "_no_retrieval"
        retrieval_tag += "_cold" if cold else ""
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
    parser.add_argument("--cold", action="store_true", help="Disable the SQLite cache for this run (DISABLE_CACHE=1)")
    parser.add_argument("--case-ids", type=str, default=None, help="Comma-separated case ids to run (e.g. to resume failed cases)")
    args = parser.parse_args()

    if args.cold:
        os.environ["DISABLE_CACHE"] = "1"

    # Benchmarks must never silently switch models on rate limits.
    os.environ["NO_MODEL_FALLBACK"] = "1"
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
        cold=args.cold,
        case_ids={int(x) for x in args.case_ids.split(",")} if args.case_ids else None,
    )


if __name__ == "__main__":
    main()
