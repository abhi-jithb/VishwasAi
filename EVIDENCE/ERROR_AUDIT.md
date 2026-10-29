# Error audit: how `ERROR_AUDIT.csv` is built and how far to trust it

**File:** `EVIDENCE/ERROR_AUDIT.csv`. It has one row per failed gold claim (missed or disagreed) in the new cold retrieval-ON run (run A, `results/gold_qwen2.5_7b_cold_*.json`).
**Regenerate:** `venv/bin/python scripts/error_audit.py results/gold_qwen2.5_7b_cold_<ts>.json`

Run A is a fresh, uncached run of the same 30 cases with the same model, so its errors are not identical to the official run's 42. Both runs are reported. The official run's file does not store evidence snippets, which is why it cannot be audited row by row.

## Columns

| Column | Meaning |
|---|---|
| `case_id`, `category` | Gold case and its category |
| `error_type` | `missed` (no extracted claim matched at ≥ 0.35) or `disagreed` (matched, but the predicted label ≠ the gold label) |
| `gold_claim`, `gold_label` | From `data/gold_claims.json` |
| `matched_extracted_claim`, `similarity` | The extracted claim the greedy matcher paired with this gold claim. For missed rows, `similarity` is the best score among the extracted claims that were still free. |
| `predicted_label`, `rationale` | The judge's verdict and its reasoning text |
| `evidence_1..3` | `title \| url` of the 3 snippets the judge saw (from `trace`) |
| `search_providers` | Which provider served the snippets (tavily / duckduckgo / wikipedia; "unknown" only for cache entries written before the provider field existed, which cannot happen in a `--cold` run) |
| `evidence_token_coverage` | Share of the gold claim's content words (stop words removed) found anywhere in the 3 snippets |
| `legacy_heuristic_cause` | The old rule from `scripts/generate_error_analysis.py` (the one behind the slide's 17/19/6) |
| `heuristic_cause`, `heuristic_detail` | The improved rule below |
| `manual_cause` | **Empty. Fill this by hand:** `extraction`, `retrieval`, `judge`, or `ambiguous gold` |

## The improved heuristic (`scripts/error_audit.py`, `heuristic_cause`)

1. **Missed claim**
   - No claims were extracted → `extraction`.
   - Some extracted claim scores ≥ 0.35 against the gold claim, but another gold claim had already taken it → `matching`. This is a scoring artefact of greedy one-to-one matching, not a pipeline failure.
   - Otherwise → `extraction`. The fact was dropped, merged, or worded too differently.
2. **Predicted `unsupported` (but gold is S or C)**
   - Search returned 0 snippets → `retrieval`.
   - The snippets contain ≥ 60% of the claim's content words → `judge`. The evidence probably held the answer and the judge was too cautious.
   - Otherwise → `retrieval`. The evidence was off-topic.
3. **Gold `unsupported` but predicted S or C** → `judge`. The judge decided a claim that cannot be verified.
4. **S ↔ C swap** → `judge`.

## Why the old 17/19/6 split is weak

The legacy rule never looks at evidence. It labels **every** missed claim as "Bad Extraction" and **every** wrong "unsupported" verdict as "Bad Search Results". So:
- Claims lost to greedy matching are counted as extraction failures.
- A judge that ignored a snippet containing the answer is counted as a search failure.
- "Ambiguous gold" can only fire if the text contains the words "ambiguous" or "disputed", so it is effectively always 0.

## Limits of the improved heuristic

- Word coverage is not entailment. A snippet can contain all the words ("Fleming", "penicillin", "1928") and still not state the fact, or it can state the fact in synonyms. The 60% cut-off is a judgement call, not a tuned value.
- It cannot detect **ambiguous gold** at all. Only a human can.
- It sees only the 3 snippets the judge saw, not what a better query would have found.
- It assigns exactly one cause, but real failures often have two (for example, a weak query *and* a cautious judge).

**Use it to sort the rows, then decide `manual_cause` yourself.** When reporting, quote the manual counts. If there is no time, quote the heuristic counts and call them "automatic triage".
