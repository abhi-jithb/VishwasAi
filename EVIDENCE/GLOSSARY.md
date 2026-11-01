# Glossary

Every worked example uses real data. "Official run" means `results/gold_qwen2.5_7b_20261028_000521.json`.

---

### Atomic claim
A single sentence that states exactly one fact that can be checked, with pronouns replaced by names. The extractor LLM splits an answer into these so that each fact gets its own true or false verdict.
**Code:** prompt `backend/evaluation/prompts.py` (`EXTRACT_CLAIMS_SYSTEM_PROMPT`); parser `backend/evaluation/claims.py` (`_clean_and_parse_claims_json`).
**Example:** case 7 answer "Sir Edmund Hillary and Neil Armstrong first summited Mount Everest in May 1953" was split into "Sir Edmund Hillary first summited Mount Everest in May 1953." and "Neil Armstrong first summited Mount Everest in May 1953."

### Supported / Contradicted / Unsupported
The three verdicts the judge can give. **Supported**: the evidence confirms the claim. **Contradicted**: the evidence states something that conflicts with it. **Unsupported**: the evidence neither confirms nor refutes it (also used when search returns nothing). "Unsupported" does not mean false.
**Code:** `backend/evaluation/prompts.py` (`VERIFY_CLAIM_SYSTEM_PROMPT`); parsing `backend/evaluation/verify.py` `_parse_status`.
**Example:** official run gold labels are 67 S, 16 C, 13 U (96 total).

### Factuality score F
The share of an answer's extracted claims that are Supported. **F = S / total** (0 if no claims).
**Code:** `backend/evaluation/scoring.py:69`.
**Example:** case 28 (Ada Lovelace): 4 S, 1 C, 2 U out of 7 claims, so F = 4/7 = 0.571.

### Verifiable precision V
Among the claims the evidence could decide (S or C), the share that are Supported. **V = S / (S + C)**. It is undefined (None) when S + C = 0.
**Code:** `backend/evaluation/scoring.py:73`.
**Example:** case 28: V = 4 / (4 + 1) = 0.80.

### Unverifiable rate U
The share of claims with no deciding evidence. **U = Unsupported / total**.
**Code:** `backend/evaluation/scoring.py:77`.
**Example:** case 28: U = 2/7 = 0.286. Two true facts ("English mathematician", "daughter of Lord Byron") were left Unsupported, which shows that a high U can come from weak retrieval as well as from false content.

### Extraction recall
The share of gold claims that the matcher paired with some extracted claim. **recall = matched / gold**.
**Code:** `tests/run_gold.py:397`.
**Example:** 79 / 96 = **82.29%** (95% CI 73.5–88.6%).

### Verification accuracy
Among matched claims, the share where the predicted verdict equals the gold label. **acc = agreed / matched**.
**Code:** `tests/run_gold.py:403`.
**Example:** 54 / 79 = **68.35%** (95% CI 57.5–77.6%).

### End-to-end accuracy
The share of all gold claims that were extracted, matched and judged correctly. A missed claim counts as wrong. **e2e = agreed / gold** = recall × verification accuracy.
**Code:** `tests/run_gold.py:404`.
**Example:** 54 / 96 = **56.25%** = 0.8229 × 0.6835.

### Matched / missed / extra
**Matched**: a gold claim paired with an extracted claim (similarity ≥ 0.35). **Missed**: a gold claim with no pair. **Extra**: an extracted claim that no gold claim took.
**Code:** `tests/run_gold.py:51` `match_gold_to_extracted`.
**Example:** 94 extracted, 79 matched, so 17 missed (96 − 79) and 15 extra (94 − 79).

### Jaccard similarity
The overlap of two word sets divided by their union, after lower-casing and removing 30 stop words. **J = |A∩B| / |A∪B|**.
**Code:** `tests/run_gold.py:34-44`.
**Example:** gold "Alexander Fleming discovered penicillin at St. Mary's Hospital in London." (9 words) vs extracted "Alexander Fleming discovered penicillin in 1928." (5 words): 4 shared, 10 in the union, so J = 0.40.

### Overlap coefficient
Shared words divided by the size of the smaller set. **O = |A∩B| / min(|A|,|B|)**. It is lenient: a short claim that sits inside a long one scores 1.0.
**Code:** `tests/run_gold.py:42`.
**Example:** same pair: O = 4/5 = 0.80. Combined score = 0.4 × 0.40 + 0.6 × 0.80 = **0.64**.

### Similarity threshold
The minimum combined score (0.4·J + 0.6·O) for a gold claim and an extracted claim to count as the same claim.
**Code:** `tests/run_gold.py:47` (`SIMILARITY_THRESHOLD = 0.35`).
**Example:** 0.64 ≥ 0.35 would match. In case 1, however, that extracted claim was already taken (see the next entry).

### Greedy one-to-one matching
Gold claims are processed in file order. Each one takes the most similar extracted claim that is still free, so an extracted claim can be used only once. This is simple and deterministic, but not globally optimal.
**Code:** `tests/run_gold.py:51`; test `tests/test_offline.py::test_greedy_matching_fixture`.
**Example:** case 1: gold #1 "Fleming discovered penicillin in 1928" took extracted #1 (score 1.0). Gold #2 "...at St. Mary's Hospital" could then only use the Nobel claim (score 0.275 < 0.35), so it was **missed**. The real reason is that the extractor dropped the hospital fact.

### Gold benchmark and its 5 categories
30 hand-written question/answer pairs with 96 hand-labelled atomic claims, in `data/gold_claims.json`. **correct** (all true, 19 claims), **planted_error** (one false detail inserted, 18), **fabricated_entity** (made-up people or places, 13), **outdated_fact** (once true, now stale, 9), **long_multi_fact** (long answers with many facts, 37).
**Example:** case 13 "Cafe Zyxvu" is fabricated_entity, and both gold labels are Unsupported.

### Top-k
The number of search results kept per claim. We use **k = 3**.
**Code:** `backend/evaluation/pipeline.py:23`.
**Example:** official run: 94 search calls, at most 3 snippets each.

### Snippet
A short text excerpt from a search result (title + URL + text). It is cut at **400 characters** for Tavily and DuckDuckGo. The judge sees only snippets, never full pages.
**Code:** `backend/search.py:39`, `:63`.

### Evidence-only verification
The judge prompt says to decide using **only** the given snippets and not the model's own knowledge. If the snippets don't settle the claim, the verdict should be Unsupported. If search returns nothing, the judge is skipped and the verdict is Unsupported without any LLM call.
**Code:** `prompts.py` `VERIFY_CLAIM_SYSTEM_PROMPT`; skip at `backend/evaluation/verify.py:131`.
**Example:** case 28: "Ada Lovelace was the daughter of poet Lord Byron" was marked Unsupported because the snippets did not say it, even though the model surely "knows" it.

### No-retrieval baseline
The same pipeline with search turned off. The judge decides from its own memory (`VERIFY_CLAIM_NO_RETRIEVAL_SYSTEM_PROMPT`). It shows how much retrieval adds.
**Code:** `pipeline.py:21-31`; flag `tests/run_gold.py --no-retrieval`.
**Example:** see `RETRIEVAL_ABLATION.md` (same model qwen2.5:7b, 30 cases, cold cache).

### Confusion matrix
A table counting matched claims by gold label (rows) and predicted label (columns). The diagonal holds the correct verdicts.
**Code:** `tests/run_gold.py:81`.
**Example:** official run row "Supported": 39 S, 3 C, 13 U, so 13 true claims were left Unsupported. That is the single biggest error cell.

### Precision / recall / F1 (per class)
For a class X: **precision** = correct X predictions / all X predictions; **recall** = correct X predictions / all gold X; **F1** = 2PR/(P+R).
**Code:** `tests/run_gold.py:90`.
**Example:** Unsupported: predicted U 29 times (13+6+10), 10 correct, so P = 34.5%. Gold U = 12, 10 found, so R = 83.3%. F1 = 48.8%. The model says "Unsupported" too easily.

### Error taxonomy
Every failure is put into one cause: **Bad Extraction**, **Bad Search Results**, **Judge Error** or **Ambiguous Gold**. The automatic version is a rule of thumb (see `ERROR_AUDIT.md`).
**Code:** `scripts/generate_error_analysis.py:16`.
**Example:** official run: 17 / 19 / 6 / 0 of 42 errors (40.5 / 45.2 / 14.3 / 0%).

### Temperature 0.0
A sampling setting. At 0 the model always picks its most likely next token, which makes outputs (nearly) repeatable.
**Code:** `claims.py:108`, `verify.py:165`.
**Example:** used for all 115 LLM calls in the official run. It does not remove variation from the live web or from different hardware.

### JSON mode
Asks the model server to emit only valid JSON. For Ollama this is `"format": "json"`. It is switched on automatically when the prompt mentions JSON, which every prompt does.
**Code:** `backend/llm.py:70`, `:110`.
**Example:** Ollama sometimes wraps the list as `{"claims": [...]}`. The parser unwraps it (`test_claims_dict_wrapped`).

### GGUF
The single-file model format used by llama.cpp and Ollama. It holds the weights, often quantised, plus metadata.
**Example:** `qwen2.5:7b` is stored by Ollama as a 4.7 GB GGUF.

### Quantisation Q4_K_M
Stores weights in about 4–5 bits instead of 16, grouped in blocks ("K") at a "medium" quality mix. It cuts memory to roughly a third with a small loss in accuracy.
**Example:** `ollama show qwen2.5:7b` → parameters 7.6B, quantization Q4_K_M. 7.6B × ~4.8 bits ≈ 4.6 GB.

### Ollama
A local model server. It runs GGUF models on CPU/GPU and exposes an HTTP API at `localhost:11434`. Our code calls `POST /api/chat`.
**Code:** `backend/llm.py` (Ollama branch); base URL in `backend/config.py`.
**Example:** on this laptop `ollama ps` shows qwen2.5:7b split 37% CPU / 63% GPU, because the 4 GB GPU cannot hold the whole model.

### SQLite cache
A local database file (`data/cache.db`) that stores every LLM response and search result. A repeated identical request is answered from disk instead of calling the model or the web.
**Code:** `backend/cache.py`.
**Example:** in the official run, 20 of 30 cases finished in under 1 s. They came from cache.

### SHA-256 key
A fixed 64-hex-character fingerprint of the request. The cache uses it as the lookup key, so identical requests map to the same row.
**Code:** `backend/cache.py:86` (LLM: `provider‖model‖system‖user‖temperature‖format`), `:92` (search: `query‖max_results`).
**Example:** test `test_llm_cache_key_deterministic`. Changing temperature from 0.0 to 0.1 gives a different key.

### WAL (write-ahead log)
A SQLite journal mode. Writes go to a side file (`cache.db-wal`) first, so readers and writers block each other less. This matters because 2 worker threads write to the cache at the same time.
**Code:** `backend/cache.py:25`. The side files are git-ignored (`.gitignore`).

### Retry
If a call fails, try again. LLM: up to **5 attempts** (`llm.py:88`). Search: up to **3 attempts** (`evidence.py:53`). There is also **1 extra LLM call** when the output is not valid JSON. That retry prompt now contains the original request plus the bad output.
**Example:** official run 115 LLM calls = 30 extractions + judge calls + any JSON-parse retries.

### Exponential vs linear back-off. **We use linear.**
Back-off is the wait between retries. Exponential doubles the wait each time (1, 2, 4, 8 s). Linear adds a fixed step. Ours: LLM wait = **2 × (attempt+1) s** (2, 4, 6, 8 s), at least 6 s on HTTP 429, or the server's "try again in X s" + 1. Search: 1, 2 s.
**Code:** `backend/llm.py:266-273`, `backend/evaluation/evidence.py:66`.

### Jitter
A small random amount added to each wait (here U(1.0, 3.5) s), so that parallel threads don't all retry at the same moment.
**Code:** `backend/llm.py:273`.

### HTTP 429
"Too Many Requests": the hosted API's rate limit. Ollama (local) never returns it. For Groq, the app can switch to a fallback model on a 429. That switch now logs a warning, can be turned off with `NO_MODEL_FALLBACK=1`, and is always off in benchmarks.
**Code:** `backend/llm.py:39`, `:182`.

### Thread pool
A fixed set of worker threads that run jobs in parallel. We verify up to **2 claims at a time** within one case.
**Code:** `backend/evaluation/pipeline.py:96`.
**Example:** a 7-claim case (case 28) runs as about 4 rounds of 2 parallel search + judge jobs.
