# Evidence table: every number and setting, traced

Branch `final-presentation`. Line numbers refer to this branch's HEAD.

**Official run file:** `results/gold_qwen2.5_7b_20261028_000521.json` (timestamp 2026-10-28T00:05:21, model `qwen2.5:7b`, provider `ollama`, retrieval ON). The file was first committed in `a0bc52b` and is unchanged in `6ec27ce`. The judging logic for retrieval mode is the same in `a0bc52b` and `6ec27ce`; `6ec27ce` only added the no-retrieval branch and tracing.

**One command checks every number below:**

```bash
venv/bin/python scripts/verify_numbers.py --runs
```

It does not trust the stored `metrics` block. It re-runs the same greedy matcher (`tests/run_gold.py:51` `match_gold_to_extracted`) over the stored gold claims and extracted claims, then compares the result with the claimed values. Result: **38 PASS, 0 FAIL**, and the stored metrics agree with the recomputation. The full table is in `FINAL_SUMMARY.md`.

Field paths below are relative to the official JSON. `cases[*]` means "summed over all 30 cases".

## A. Headline numbers (official run)

| Claim in slides/report | Value | Source file + field or code location | Exact command to reproduce | Caveat |
|---|---|---|---|---|
| Test cases | 30 | `metrics.total_cases`; `data/gold_claims.json` has 30 entries | `scripts/verify_numbers.py` | The set is small and was written by the author. |
| Gold claims | 96 | `metrics.total_gold_claims`; `len(cases[*].gold_claims)` | same | Gold labels were written by one person. Nobody checked agreement with a second labeller. |
| Gold label split S/C/U | 67 / 16 / 13 | `cases[*].gold_claims[*].label` | same | Classes are imbalanced (70% Supported), so per-class numbers for C and U rest on 16 and 13 claims. |
| Matched gold claims | 79 | `metrics.total_matched_claims`; matcher `tests/run_gold.py:51` | same | Matching is lexical (Jaccard + overlap), not semantic. See LIMITATIONS. |
| Missed gold claims | 17 | `metrics.missed_gold_claims` = 96 − 79 | same | "Missed" can also mean the claim was extracted but worded too differently to pass 0.35. |
| Extracted claims | 94 | `sum(len(cases[*].extracted_claims))` | same | The original file has no stored total. The total is recomputed from the per-case lists. |
| Extra claims | 15 | 94 − 79 | same | An extra claim can be a real fact that the gold set did not list. It is not always an error. |
| Agreed | 54 | `metrics.status_agreed` | same | |
| Disagreed | 25 | `metrics.status_disagreed` = 79 − 54 | same | |
| Extraction recall | 82.29% | `metrics.claim_extraction_recall`; formula `tests/run_gold.py:397` (matched / gold) | same | 95% Wilson CI: 73.46 – 88.64%. |
| Verification accuracy | 68.35% | `metrics.verification_accuracy`; formula `tests/run_gold.py:403` (agreed / matched) | same | Measured only on matched claims. 95% Wilson CI: 57.45 – 77.55%. |
| End-to-end accuracy | 56.25% | `metrics.end_to_end_accuracy`; formula `tests/run_gold.py:404` (agreed / gold) | same | A missed claim counts as wrong. 95% Wilson CI: 46.28 – 65.74%. |
| Total time | 742.06 s | `metrics.total_latency_seconds`; `tests/run_gold.py:385` | same | **Warm cache.** 20 of 30 cases took < 1 s (`cases[*].latency_seconds`), so they were served from the SQLite cache. 4 cases (27–30) account for 696 s. This is not a cold-start latency. |
| Time per case | 24.74 s | `metrics.avg_latency_per_case_seconds` = 742.06 / 30 | same | Same warm-cache caveat. For cold timings see `RETRIEVAL_ABLATION.md`. |
| LLM calls | 115 (3.83/case) | `sum(cases[*].cost_log.llm_calls)` | same | Formula: 30 extractions + one judge call per claim with non-empty evidence + parse retries. A call is counted even when the cache served it. |
| Search calls | 94 (3.13/case) | `sum(cases[*].cost_log.search_calls)` | same | One search per extracted claim (94 = extracted). It is counted even when the cache served it. |
| Hardware in file | Windows 11, Ryzen 7 250, 24 GB, RTX 5050 4 GB | `hardware` | n/a | The original run used a different laptop from the new runs (Fedora 43, i5-11400H, 16 GB, RTX 3050 4 GB). The old `_detect_system_hardware` had a hard-coded 16 GB default. It now reads os-release, lscpu, free and nvidia-smi (`tests/run_gold.py`, `_detect_system_hardware`). |

## B. Per category (official run): gold / matched / agreed

Source: `metrics.category_stats.<cat>.{gold,matched,agreed}`. Command: `scripts/verify_numbers.py`.

| Category | Gold | Matched | Agreed | Verification acc. | Caveat |
|---|---|---|---|---|---|
| correct | 19 | 13 | 11 | 84.6% | 6 misses come from extraction or matching |
| planted_error | 18 | 12 | 7 | 58.3% | |
| fabricated_entity | 13 | 12 | 10 | 83.3% | |
| outdated_fact | 9 | 6 | 2 | 33.3% | Only 6 matched claims. Live web results also change over time. |
| long_multi_fact | 37 | 36 | 24 | 66.7% | |

## C. Confusion matrix and per-class metrics (official run, matched claims only)

Code: `tests/run_gold.py:81` `confusion_matrix`, `tests/run_gold.py:90` `per_class_prf`. The original file has no stored matrix. `verify_numbers.py` recomputes it from `cases[*].gold_claims` + `cases[*].extracted_claims[*].status`.

| Gold \ Predicted | S | C | U |
|---|---|---|---|
| Supported (55 matched) | 39 | 3 | 13 |
| Contradicted (12) | 1 | 5 | 6 |
| Unsupported (12) | 1 | 1 | 10 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Supported | 95.1 | 70.9 | 81.2 |
| Contradicted | 55.6 | 41.7 | 47.6 |
| Unsupported | 34.5 | 83.3 | 48.8 |

Caveat: the C and U rows have only 12 claims each. One claim changes recall by about 8 percentage points.

## D. Error causes (official run): 17 / 19 / 6 of 42

| Claim | Value | Source | Command | Caveat |
|---|---|---|---|---|
| Bad Extraction | 17 (40.5%) | `scripts/generate_error_analysis.py:16` `classify_error`, missed → extraction (line 18) | `python scripts/generate_error_analysis.py --benchmark results/gold_qwen2.5_7b_20261028_000521.json --output /tmp/ea.md` or `scripts/verify_numbers.py` | **Heuristic.** Every missed claim is labelled extraction, even when the real cause is the lexical matcher. |
| Bad Search Results | 19 (45.2%) | same, line 29: any disagreement where the pipeline said `unsupported` | same | **Heuristic.** It does not open the evidence. A judge that ignored good evidence is also labelled "search". |
| Judge Error | 6 (14.3%) | same: S↔C swaps | same | |
| Ambiguous gold | 0 | same | same | The heuristic almost never fires (it needs the words "ambiguous" or "disputed"). Use the manual audit in `ERROR_AUDIT.csv`. |
| Total errors | 42 | 17 missed + 25 disagreed | same | |

## E. Settings

| Setting | Value | Where defined | How to show it | Caveat |
|---|---|---|---|---|
| top-k search results | 3 | `backend/evaluation/pipeline.py:23` (`max_results=3`) | `grep -n max_results=3 backend/evaluation/pipeline.py` | `web_search` defaults to 5 (`backend/search.py:11`), but the pipeline always passes 3. |
| Snippet length | 400 chars (+ "...") | `backend/search.py:39` (Tavily), `backend/search.py:63` (DuckDuckGo) | `tests/test_offline.py::test_search_duckduckgo_when_tavily_disabled` | Wikipedia snippets are not truncated. The API already returns them short (~150–300 chars). |
| Search query length | ≤ 200 chars, cut at a word boundary | `backend/evaluation/evidence.py:17-18` | `test_search_query_derivation_truncates_to_200_chars` | The query is the claim text with quotes and trailing punctuation removed. There is no query rewriting. |
| Worker threads | 2 | `backend/evaluation/pipeline.py:96` (`ThreadPoolExecutor(max_workers=min(len(raw_claims), 2))`) | grep | Claims inside one case run in parallel. Cases run one after another. |
| LLM retries | 5 attempts | `backend/llm.py:88` (`max_retries = 5`, `for attempt in range(max_retries)`) | grep | Precisely: 5 attempts = 1 try + up to 4 retries, with 4 sleeps. This is separate from the single JSON-parse retry in `claims.py` and `verify.py`. |
| Search retries | 3 attempts | `backend/evaluation/evidence.py:53` | grep | `web_search` catches provider errors itself and falls through to the next provider, so this retry loop rarely fires in practice. |
| LLM back-off formula | `2.0 × (attempt+1)` s + jitter U(1.0, 3.5) s | `backend/llm.py:266`, jitter `backend/llm.py:273` | grep | **Linear, not exponential.** Waits are about 2, 4, 6, 8 s plus jitter. If the error says "try again in X s", the wait is X + 1 s (`llm.py:267`). |
| Wait on HTTP 429 | at least 6 s (+ jitter) | `backend/llm.py:271` (`max(wait_sec, 6.0)`) | grep | Only relevant for hosted APIs. Ollama never returns 429. |
| Search back-off | `1.0 × (attempt+1)` s, no jitter | `backend/evaluation/evidence.py:66` | grep | Linear. |
| Temperature | 0.0 | `backend/evaluation/claims.py:108`, `verify.py:165`, default `backend/llm.py:48` | grep | Temperature 0 makes the model's choice as deterministic as possible. Results can still differ across hardware and versions, and web results change. |
| JSON mode | on when "json" appears in the prompt | `backend/llm.py:70`; Ollama `format=json` at `backend/llm.py:110` | grep | Every extraction and verification prompt contains "JSON", so JSON mode is always on for Ollama. |
| Similarity threshold | 0.35 | `tests/run_gold.py:47` (`SIMILARITY_THRESHOLD`) | `test_greedy_matching_fixture` | The official run used the same 0.35, inline (`git show a0bc52b:tests/run_gold.py`, line 144). |
| Similarity weights | 0.4 × Jaccard + 0.6 × overlap | `tests/run_gold.py:44` | `test_claim_similarity_worked_example` | The overlap coefficient is lenient. A short claim inside a long one scores high. |
| Stop words | 30 words | `tests/run_gold.py:27` | | Removed before computing similarity. |
| LLM cache key | SHA-256 of `provider‖model‖system‖user‖temperature(4dp)‖format`, lowercased provider/model/format, stripped prompts | `backend/cache.py:77-87` | `test_llm_cache_key_deterministic` | Same prompt + model + temperature returns the cached answer. Changing the prompt text makes a new key. |
| Search cache key | SHA-256 of `lower(query)‖max_results` | `backend/cache.py:90-93` | `test_search_cache_key_deterministic` | Empty search results are not cached (`backend/search.py`, `if results:`). |
| Cache store | SQLite `data/cache.db`, WAL journal | `backend/cache.py:25-26` | | `--cold` / `DISABLE_CACHE=1` turns off both read and write (`backend/cache.py:72`). |
| Ollama request timeout | 180 s | `backend/llm.py:120` | grep | |
| Model | qwen2.5:7b, 7.6B parameters, Q4_K_M, 4.7 GB | `ollama show qwen2.5:7b`; stored in new runs at `run_info.model_info` | `ollama show qwen2.5:7b` | Quantisation was checked on this machine. The official file does not record it. The Ollama default tag `qwen2.5:7b` is Q4_K_M. On the 4 GB GPU, Ollama splits the model about 37% CPU / 63% GPU (`ollama ps`). |
| Model fallback (Groq only) | off for benchmarks | `backend/llm.py:39` `model_fallback_disabled`; set in `tests/run_gold.py:506` | `test_model_fallback_switch` | Ollama never falls back. New runs record `models_used`. |
| Answer length limit (API) | 3000 chars | `backend/main.py:59` | | |

## F. New runs (ablation)

See `RETRIEVAL_ABLATION.md`. Reproduce with:

```bash
scripts/run_ablation.sh      # runs A then B, logs in results/logs/
venv/bin/python scripts/verify_numbers.py --runs results/gold_qwen2.5_7b_cold_*.json results/gold_qwen2.5_7b_no_retrieval_cold_*.json
```

Each new result file stores `run_info.git.commit`, `run_info.cold`, `run_info.model_info.quantization`, `models_used`, `hardware`, per-case `latency_seconds`, `label_comparison` (gold vs predicted for each gold claim), `num_extracted_claims`, `num_extra_claims`, `trace` (search query, evidence snippets with `provider`, raw judge output), and in `metrics`: `confusion_matrix_matched`, `per_class_metrics`, `total_llm_calls`, `total_search_calls`, `extra_extracted_claims`.
