"""Print markdown tables comparing a retrieval-ON run with a retrieval-OFF run (same model).

Usage:
    python scripts/ablation_report.py results/<on>.json results/<off>.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.verify_numbers import CATS, recompute
from tests.run_gold import LABELS


def _pct(k: int, n: int) -> str:
    return f"{k / n * 100:.1f}%" if n else "-"


def _labels(data: dict[str, Any]) -> dict[tuple[Any, str], dict[str, Any]]:
    out = {}
    for c in data["cases"]:
        trace = {t.get("claim", ""): t for t in (c.get("trace") or {}).get("claims", [])}
        for row in c.get("label_comparison", []):
            ext = c["extracted_claims"][row["matched_extracted_index"]] if row["matched_extracted_index"] is not None else {}
            out[(c["case_id"], row["gold_text"])] = {
                **row,
                "case_id": c["case_id"],
                "category": c["category"],
                "reasoning": ext.get("reasoning", ""),
                "snippets": trace.get(ext.get("text", ""), {}).get("snippets", []),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("on")
    ap.add_argument("off")
    args = ap.parse_args()
    on_d = json.loads(Path(args.on).read_text(encoding="utf-8"))
    off_d = json.loads(Path(args.off).read_text(encoding="utf-8"))
    on, off = recompute(Path(args.on)), recompute(Path(args.off))

    print(f"ON: `{Path(args.on).name}`  (commit {on_d['run_info']['git']['commit'][:7]}, cold={on_d['run_info']['cold']}, models_used={on_d.get('models_used')})")
    print(f"OFF: `{Path(args.off).name}` (commit {off_d['run_info']['git']['commit'][:7]}, cold={off_d['run_info']['cold']}, models_used={off_d.get('models_used')})")
    print(f"Hardware: {on_d['hardware']}; model info: {on_d['run_info'].get('model_info')}")
    print(f"Search providers used (ON, snippet count): {on_d['run_info'].get('search_provider_counts')}")
    print(f"Failed cases: ON {on_d['run_info'].get('failed_case_ids')}, OFF {off_d['run_info'].get('failed_case_ids')}\n")

    print("## Headline\n")
    print("| Metric | Retrieval ON | Retrieval OFF | Δ (ON − OFF) |")
    print("|---|---|---|---|")
    rows = [
        ("Cases", "cases"), ("Gold claims", "gold_claims"), ("Extracted claims", "extracted"),
        ("Matched", "matched"), ("Missed", "missed"), ("Extra", "extra"), ("Agreed", "agreed"),
        ("Extraction recall %", "recall_pct"), ("Verification accuracy %", "verification_accuracy_pct"),
        ("End-to-end accuracy %", "end_to_end_pct"), ("LLM calls", "llm_calls"),
        ("LLM calls / case", "llm_calls_per_case"), ("Search calls", "search_calls"),
        ("Total time (s)", "total_time_s"), ("Time / case (s)", "per_case_s"),
    ]
    for name, key in rows:
        a, b = on[key], off[key]
        delta = round(a - b, 2) if isinstance(a, (int, float)) else ""
        print(f"| {name} | {a} | {b} | {delta} |")
    for name, key in (("Recall 95% CI", "ci95_recall"), ("Verification acc. 95% CI", "ci95_verification_accuracy"), ("End-to-end 95% CI", "ci95_end_to_end")):
        print(f"| {name} | {on[key][0]}–{on[key][1]} | {off[key][0]}–{off[key][1]} | |")

    print("\n## Per category (gold / matched / agreed → verification accuracy)\n")
    print("| Category | ON | OFF | Δ acc (pp) |")
    print("|---|---|---|---|")
    for cat in CATS:
        a = [int(x) for x in on[f"cat_{cat}"].split("/")]
        b = [int(x) for x in off[f"cat_{cat}"].split("/")]
        acc_a = a[2] / a[1] * 100 if a[1] else 0
        acc_b = b[2] / b[1] * 100 if b[1] else 0
        print(f"| {cat} | {on[f'cat_{cat}']} → {acc_a:.1f}% | {off[f'cat_{cat}']} → {acc_b:.1f}% | {acc_a - acc_b:+.1f} |")

    for label, r in (("ON", on), ("OFF", off)):
        print(f"\n## Confusion matrix, retrieval {label} (rows gold, cols predicted S/C/U)\n")
        print("| Gold \\ Pred | S | C | U |")
        print("|---|---|---|---|")
        for lab in LABELS:
            print(f"| {lab} | " + " | ".join(r[f"cm_{lab}"].split("/")) + " |")

    print("\n## Per-class precision / recall / F1 (%)\n")
    print("| Class | ON P/R/F1 | OFF P/R/F1 |")
    print("|---|---|---|")
    for lab in LABELS:
        print(f"| {lab} | {on[f'prf_{lab}']} | {off[f'prf_{lab}']} |")

    # Verdict changes on gold claims matched in both runs
    la, lb = _labels(on_d), _labels(off_d)
    fixed, broke, same_ok, same_bad = [], [], [], []
    for key, a in la.items():
        b = lb.get(key)
        if not b or a["predicted_label"] is None or b["predicted_label"] is None:
            continue
        if a["agreed"] and not b["agreed"]:
            fixed.append((a, b))
        elif b["agreed"] and not a["agreed"]:
            broke.append((a, b))
        elif a["agreed"]:
            same_ok.append((a, b))
        else:
            same_bad.append((a, b))
    print("\n## Verdict changes (gold claims matched in both runs)\n")
    print(f"- Fixed by retrieval (OFF wrong → ON right): **{len(fixed)}**")
    print(f"- Broken by retrieval (OFF right → ON wrong): **{len(broke)}**")
    print(f"- Unchanged correct: **{len(same_ok)}**; unchanged wrong: **{len(same_bad)}**")
    for title, items in (("FIXED", fixed), ("BROKE", broke), ("UNCHANGED", same_ok)):
        print(f"\n### {title} examples\n")
        for a, b in items[:4]:
            print(f"- Case {a['case_id']} ({a['category']}) | gold [{a['gold_label']}] **{a['gold_text']}**")
            print(f"  - OFF: {b['predicted_label']} — {b['reasoning'][:220]}")
            print(f"  - ON: {a['predicted_label']} — {a['reasoning'][:220]}")
            for s in a["snippets"][:3]:
                print(f"    - [{s.get('provider', '?')}] {s.get('title', '')} <{s.get('url', '')}>: \"{s.get('snippet', '')[:200]}\"")


if __name__ == "__main__":
    main()
