"""Recompute every headline number from results/*.json and compare with the claimed values.

Usage:
    python scripts/verify_numbers.py                      # official run + any *_cold_* runs
    python scripts/verify_numbers.py --runs results/a.json results/b.json

Integers must match exactly. Percentages and seconds pass within 0.05 (percentage points / s).
Recomputation uses the stored gold claims and extracted claims, re-running the same greedy matcher
(tests/run_gold.py: match_gold_to_extracted) rather than trusting the stored metrics block.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.generate_error_analysis import classify_error
from tests.run_gold import LABELS, confusion_matrix, match_gold_to_extracted, per_class_prf

OFFICIAL = root_dir / "results" / "gold_qwen2.5_7b_20261028_000521.json"
TOL = 0.05
CATS = ["correct", "planted_error", "fabricated_entity", "outdated_fact", "long_multi_fact"]

# Values as written in the slides / report for the official run.
CLAIMED: dict[str, Any] = {
    "cases": 30,
    "gold_claims": 96,
    "gold_supported": 67,
    "gold_contradicted": 16,
    "gold_unsupported": 13,
    "matched": 79,
    "missed": 17,
    "extracted": 94,
    "extra": 15,
    "agreed": 54,
    "disagreed": 25,
    "recall_pct": 82.29,
    "verification_accuracy_pct": 68.35,
    "end_to_end_pct": 56.25,
    "total_time_s": 742.06,
    "per_case_s": 24.74,
    "llm_calls": 115,
    "llm_calls_per_case": 3.83,
    "search_calls": 94,
    "search_calls_per_case": 3.13,
    "cat_correct": "19/13/11",
    "cat_planted_error": "18/12/7",
    "cat_fabricated_entity": "13/12/10",
    "cat_outdated_fact": "9/6/2",
    "cat_long_multi_fact": "37/36/24",
    "cm_supported": "39/3/13",
    "cm_contradicted": "1/5/6",
    "cm_unsupported": "1/1/10",
    "prf_supported": "95.1/70.9/81.2",
    "prf_contradicted": "55.6/41.7/47.6",
    "prf_unsupported": "34.5/83.3/48.8",
    "err_bad_extraction": 17,
    "err_bad_search": 19,
    "err_judge": 6,
    "err_total": 42,
    "err_bad_extraction_pct": 40.5,
    "err_bad_search_pct": 45.2,
    "err_judge_pct": 14.3,
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round((centre - half) * 100, 2), round((centre + half) * 100, 2))


def recompute(path: Path) -> dict[str, Any]:
    """Recompute all figures for one benchmark result file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    r: dict[str, Any] = {}
    gold_labels = [g["label"].lower() for c in cases for g in c["gold_claims"]]
    r["cases"] = len(cases)
    r["gold_claims"] = len(gold_labels)
    for lab in LABELS:
        r[f"gold_{lab}"] = gold_labels.count(lab)

    pairs: list[tuple[str, str]] = []
    cat = {k: [0, 0, 0] for k in CATS}
    matched = agreed = extracted = 0
    for c in cases:
        ext = c["extracted_claims"]
        extracted += len(ext)
        for g, (idx, _) in zip(c["gold_claims"], match_gold_to_extracted(c["gold_claims"], ext)):
            cs = cat.setdefault(c["category"], [0, 0, 0])
            cs[0] += 1
            if idx == -1:
                continue
            gl, pl = g["label"].lower(), ext[idx]["status"].lower()
            pairs.append((gl, pl))
            matched += 1
            cs[1] += 1
            if gl == pl:
                agreed += 1
                cs[2] += 1
    n_gold = r["gold_claims"]
    r.update(
        matched=matched,
        missed=n_gold - matched,
        extracted=extracted,
        extra=extracted - matched,
        agreed=agreed,
        disagreed=matched - agreed,
        recall_pct=round(matched / n_gold * 100, 2),
        verification_accuracy_pct=round(agreed / matched * 100, 2) if matched else 0.0,
        end_to_end_pct=round(agreed / n_gold * 100, 2),
    )
    m = data["metrics"]
    r["total_time_s"] = m["total_latency_seconds"]
    r["per_case_s"] = m["avg_latency_per_case_seconds"]
    r["sum_case_latency_s"] = round(sum(c.get("latency_seconds", 0) for c in cases), 2)
    r["cases_under_1s"] = sum(1 for c in cases if c.get("latency_seconds", 0) < 1.0)
    llm = sum(c.get("cost_log", {}).get("llm_calls", 0) for c in cases)
    srch = sum(c.get("cost_log", {}).get("search_calls", 0) for c in cases)
    r.update(
        llm_calls=llm,
        llm_calls_per_case=round(llm / len(cases), 2),
        search_calls=srch,
        search_calls_per_case=round(srch / len(cases), 2),
    )
    for k, (g, mt, a) in cat.items():
        r[f"cat_{k}"] = f"{g}/{mt}/{a}"
    cm = confusion_matrix(pairs)
    for lab in LABELS:
        r[f"cm_{lab}"] = "/".join(str(cm[lab][p]) for p in LABELS)
    prf = per_class_prf(cm)
    for lab in LABELS:
        vals = [prf[lab][k] for k in ("precision", "recall", "f1")]
        r[f"prf_{lab}"] = "/".join("-" if v is None else f"{v:.1f}" for v in vals)

    # Heuristic error causes (scripts/generate_error_analysis.py: classify_error)
    counts = {"Bad Extraction": 0, "Bad Search Results": 0, "Judge Error": 0, "Ambiguous Gold Label": 0}
    for c in cases:
        for mm in c.get("missed_claims", []):
            counts[classify_error(c, mm, is_missed=True)[0]] += 1
        for d in c.get("disagreements", []):
            counts[classify_error(c, d, is_missed=False)[0]] += 1
    tot = sum(counts.values())
    r["err_bad_extraction"] = counts["Bad Extraction"]
    r["err_bad_search"] = counts["Bad Search Results"]
    r["err_judge"] = counts["Judge Error"]
    r["err_total"] = tot
    for key, name in (("bad_extraction", "Bad Extraction"), ("bad_search", "Bad Search Results"), ("judge", "Judge Error")):
        r[f"err_{key}_pct"] = round(counts[name] / tot * 100, 1) if tot else 0.0

    r["ci95_recall"] = wilson(matched, n_gold)
    r["ci95_verification_accuracy"] = wilson(agreed, matched)
    r["ci95_end_to_end"] = wilson(agreed, n_gold)
    r["_stored_metrics_agree"] = (
        m["total_matched_claims"] == matched and m["status_agreed"] == agreed and m["total_gold_claims"] == n_gold
    )
    return r


def compare(claimed: Any, actual: Any) -> str:
    if isinstance(claimed, str):
        if "/" in claimed and all("." in x for x in claimed.split("/")):
            try:
                ok = all(abs(float(a) - float(b)) <= TOL for a, b in zip(claimed.split("/"), actual.split("/")))
                return "PASS" if ok else "FAIL"
            except ValueError:
                return "FAIL"
        return "PASS" if claimed == actual else "FAIL"
    if isinstance(claimed, int) and not isinstance(claimed, bool):
        return "PASS" if claimed == actual else "FAIL"
    return "PASS" if abs(float(claimed) - float(actual)) <= TOL else "FAIL"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="*", default=None, help="Extra result files to recompute side by side")
    ap.add_argument("--markdown", action="store_true", help="Print markdown tables")
    args = ap.parse_args()

    runs = [Path(p) for p in args.runs] if args.runs is not None else sorted((root_dir / "results").glob("gold_*_cold_*.json"))
    official = recompute(OFFICIAL)
    others = [(p.name, recompute(p)) for p in runs]

    print(f"Official run: {OFFICIAL.relative_to(root_dir)}")
    print(f"Stored metrics agree with recomputation: {official['_stored_metrics_agree']}")
    head = ["figure", "claimed", "recomputed", "result"] + [n for n, _ in others]
    rows = []
    fails = 0
    for key, claimed in CLAIMED.items():
        res = compare(claimed, official[key])
        fails += res == "FAIL"
        rows.append([key, str(claimed), str(official[key]), res] + [str(o.get(key, "")) for _, o in others])
    extra_keys = ["sum_case_latency_s", "cases_under_1s", "ci95_recall", "ci95_verification_accuracy", "ci95_end_to_end"]
    for key in extra_keys:
        rows.append([key, "-", str(official[key]), "info"] + [str(o.get(key, "")) for _, o in others])

    if args.markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for row in rows:
            print("| " + " | ".join(row) + " |")
    else:
        widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
        for row in [head] + rows:
            print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
    print(f"\n{len(CLAIMED) - fails} PASS, {fails} FAIL (tolerance {TOL})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
