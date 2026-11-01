# Vishwas AI: Codebase Report

Vishwas AI audits an answer that another model already wrote. It splits the answer into claims, looks for web evidence, and labels each claim supported, contradicted, or unsupported. Unsupported means no usable evidence, not false.

## Request flow

1. A request enters through `evaluate.py` (CLI) or `POST /api/evaluate` in `backend/main.py`. Both call `evaluate()` in `backend/evaluation/pipeline.py`.
2. **Extract.** `extract_claims()` sends the question and answer to the LLM and parses a JSON list of atomic claims. A parse failure gets one retry.
3. **Retrieve.** Up to 2 worker threads process the claims. For each claim, `gather_evidence()` builds a query and calls `web_search()`, which tries Tavily, then DuckDuckGo, then Wikipedia, and keeps up to 3 results. With `--no-retrieval`, this stage is skipped.
4. **Judge.** `verify_claim()` asks the LLM for a verdict using only the snippets. If there are no snippets, it returns unsupported without calling the LLM. A parse failure gets one retry. URLs not present in the evidence are dropped.
5. **Score.** `compute_factuality()` computes factuality, verifiable precision, and unverifiable rate.
6. The result goes back as a report (CLI), as JSON (`--json`), or to the page in `frontend/index.html`.

All model calls pass through `call_llm()` in `backend/llm.py`. LLM and search responses are cached in `data/cache.db`.

## Files

### Core

| File | Purpose |
|---|---|
| `evaluate.py` | CLI entrypoint |
| `backend/main.py` | FastAPI app: health and evaluate routes, serves the page |
| `backend/config.py` | Loads `.env` and exposes settings |
| `backend/llm.py` | Provider calls for Ollama, Groq, OpenAI, Anthropic |
| `backend/search.py` | Tavily, DuckDuckGo, Wikipedia search chain |
| `backend/cache.py` | SQLite cache for LLM and search results |
| `backend/evaluation/pipeline.py` | Runs the stages |
| `backend/evaluation/prompts.py` | All extractor and judge prompts |
| `backend/evaluation/claims.py` | Claim extraction and JSON parsing |
| `backend/evaluation/evidence.py` | Claim to query, calls search |
| `backend/evaluation/verify.py` | Judge call and verdict parsing |
| `backend/evaluation/scoring.py` | The three scores |
| `frontend/index.html` | Report page |
| `data/gold_claims.json` | 30 cases, 96 labelled claims |

### Tests and scripts

| File | Purpose |
|---|---|
| `tests/run_gold.py` | Gold benchmark runner; other scripts import its matcher |
| `tests/test_offline.py` | 38 mocked tests; no network or cache use |
| `scripts/verify_numbers.py` | Recomputes headline numbers from `results/` |
| `scripts/generate_error_analysis.py` | Error report; holds `classify_error` |
| `scripts/error_audit.py` | Per-failure CSV from a current-format run |
| `scripts/compare_benchmarks.py` | Table of saved runs |
| `scripts/run_ablation.sh` | Cold runs A (retrieval on) then B (retrieval off) |
| `scripts/ablation_report.py` | Compares run A and run B |
| `scripts/demo.py` | Runs 3 fixed demo examples |
| `scripts/prove_search_fallback.py` | Demonstrates the search fallback chain |
| `scripts/test_llm.py`, `scripts/test_search.py` | Connectivity checks; not pytest tests |

### Other folders

| Path | Purpose |
|---|---|
| `results/` | Saved benchmark runs and logs |
| `docs/` | Project status, viva guide, model explainer |
| `EVIDENCE/` | Traced numbers, glossary, limitations, demo checklist |
| `_archive/` | Retired files, not used by the running code |

## Commands

Run these from the repo root with `venv` active. Prefix `LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:7b` to use the local model.

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

# Audit one answer (add --no-retrieval, --inspect, or --json)
python evaluate.py -q "QUESTION" -a "ANSWER"

# Serve the page and API at http://127.0.0.1:8000
uvicorn backend.main:app --port 8000

# Offline tests
python -m pytest tests/test_offline.py -q

# Gold benchmark smoke test, without saving a file
python tests/run_gold.py --limit 1 --no-save

# Full gold benchmark; writes to results/
python tests/run_gold.py
python tests/run_gold.py --no-retrieval

# Check and compare saved numbers
python scripts/verify_numbers.py --runs
python scripts/compare_benchmarks.py

# Connectivity checks
python -m scripts.test_llm
python -m scripts.test_search
```

Other `run_gold.py` flags are `--provider`, `--model`, `--output`, `--cold`, and `--case-ids`.

## Prompt locations in `backend/evaluation/prompts.py`

| Role | Constant | Lines |
|---|---|---|
| Extractor system prompt | `EXTRACT_CLAIMS_SYSTEM_PROMPT` | 7-27 |
| Extractor user prompt | `EXTRACT_CLAIMS_USER_PROMPT_TEMPLATE` | 29-32 |
| Extractor retry prompt | `EXTRACT_CLAIMS_RETRY_PROMPT_TEMPLATE` | 34-42 |
| Judge system prompt | `VERIFY_CLAIM_SYSTEM_PROMPT` | 49-67 |
| Judge user prompt | `VERIFY_CLAIM_USER_PROMPT_TEMPLATE` | 69-75 |
| Judge retry prompt | `VERIFY_CLAIM_RETRY_PROMPT_TEMPLATE` | 77-86 |
| No-retrieval judge system prompt | `VERIFY_CLAIM_NO_RETRIEVAL_SYSTEM_PROMPT` | 93-110 |
| No-retrieval judge user prompt | `VERIFY_CLAIM_NO_RETRIEVAL_USER_PROMPT_TEMPLATE` | 112-115 |
