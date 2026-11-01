"""Write EVIDENCE/ERROR_AUDIT.csv: one row per failed gold claim (missed or disagreed).

Usage:
    python scripts/error_audit.py results/gold_qwen2.5_7b_cold_<ts>.json [--output EVIDENCE/ERROR_AUDIT.csv]

Needs a result file written by the current tests/run_gold.py (fields `label_comparison` and `trace`).
The heuristic cause is a first-pass guess; fill `manual_cause` by hand.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.generate_error_analysis import classify_error
from tests.run_gold import STOP_WORDS, SIMILARITY_THRESHOLD, _claim_similarity

RELEVANCE_THRESHOLD = 0.6

COLUMNS = [
    "case_id", "category", "error_type", "gold_claim", "gold_label",
    "matched_extracted_claim", "similarity", "predicted_label", "rationale",
    "evidence_1", "evidence_2", "evidence_3", "search_providers", "evidence_token_coverage",
    "legacy_heuristic_cause", "heuristic_cause", "heuristic_detail", "manual_cause",
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower())) - STOP_WORDS


def token_coverage(claim: str, snippets: list[dict[str, Any]]) -> float:
    """Share of the claim's content words that appear anywhere in the evidence text."""
    words = _tokens(claim)
    if not words:
        return 0.0
    ev = _tokens(" ".join(f"{s.get('title', '')} {s.get('snippet', '')}" for s in snippets))
    return len(words & ev) / len(words)


def heuristic_cause(row: dict[str, Any], case: dict[str, Any], snippets: list[dict[str, Any]]) -> tuple[str, str]:
    """Guess extraction / matching / retrieval / judge from stored data (no LLM)."""
    gold, pred = row["gold_label"], row["predicted_label"]
    if pred is None:
        sims = [_claim_similarity(row["gold_text"], e.get("text", "")) for e in case["extracted_claims"]]
        best = max(sims, default=0.0)
        if not case["extracted_claims"]:
            return "extraction", "no claims were extracted for this case"
        if best >= SIMILARITY_THRESHOLD:
            return "matching", f"an extracted claim scores {best:.2f} but was already taken by another gold claim (greedy)"
        return "extraction", f"best similarity to any extracted claim is {best:.2f} < {SIMILARITY_THRESHOLD}"
    cov = token_coverage(row["gold_text"], snippets)
    if pred == "unsupported":
        if not snippets:
            return "retrieval", "search returned no results; judge skipped"
        if cov >= RELEVANCE_THRESHOLD:
            return "judge", f"evidence covers {cov:.0%} of the claim's words but judge said unsupported"
        return "retrieval", f"evidence covers only {cov:.0%} of the claim's words"
    if gold == "unsupported":
        return "judge", f"gold is unsupported (unverifiable) but judge said {pred}"
    return "judge", f"verdict swap {gold} -> {pred} (evidence covers {cov:.0%} of words)"


def build_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case in data["cases"]:
        trace_claims = (case.get("trace") or {}).get("claims", [])
        by_text = {t.get("claim", ""): t for t in trace_claims}
        ext = case["extracted_claims"]
        for row in case.get("label_comparison", []):
            if row["agreed"]:
                continue
            idx = row["matched_extracted_index"]
            ext_claim = ext[idx] if idx is not None else {}
            tr = by_text.get(ext_claim.get("text", ""), {}) if idx is not None else {}
            snippets = tr.get("snippets", [])
            missed = idx is None
            if missed:
                legacy_text = f"Missed gold claim: '{row['gold_text']}'"
            else:
                legacy_text = f"'{row['gold_text']}' -> Pipeline: [{row['predicted_label']}] vs Gold: [{row['gold_label']}]"
            legacy = classify_error(case, legacy_text, is_missed=missed)[0]
            cause, detail = heuristic_cause(row, case, snippets)
            ev = [f"{s.get('title', '')} | {s.get('url', '')}" for s in snippets[:3]]
            ev += [""] * (3 - len(ev))
            out.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "error_type": "missed" if missed else "disagreed",
                    "gold_claim": row["gold_text"],
                    "gold_label": row["gold_label"],
                    "matched_extracted_claim": row["matched_text"] or "",
                    "similarity": row["similarity"],
                    "predicted_label": row["predicted_label"] or "",
                    "rationale": ext_claim.get("reasoning", ""),
                    "evidence_1": ev[0],
                    "evidence_2": ev[1],
                    "evidence_3": ev[2],
                    "search_providers": ";".join(sorted({s.get("provider", "?") for s in snippets})),
                    "evidence_token_coverage": round(token_coverage(row["gold_text"], snippets), 2) if not missed else "",
                    "legacy_heuristic_cause": legacy,
                    "heuristic_cause": cause,
                    "heuristic_detail": detail,
                    "manual_cause": "",
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("benchmark", help="Result JSON from tests/run_gold.py (retrieval ON)")
    ap.add_argument("--output", default=str(root_dir / "EVIDENCE" / "ERROR_AUDIT.csv"))
    args = ap.parse_args()
    data = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    rows = build_rows(data)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    legacy: dict[str, int] = {}
    for r in rows:
        counts[r["heuristic_cause"]] = counts.get(r["heuristic_cause"], 0) + 1
        legacy[r["legacy_heuristic_cause"]] = legacy.get(r["legacy_heuristic_cause"], 0) + 1
    print(f"{len(rows)} failed gold claims written to {args.output}")
    print("heuristic_cause:", counts)
    print("legacy_heuristic_cause:", legacy)


if __name__ == "__main__":
    main()
