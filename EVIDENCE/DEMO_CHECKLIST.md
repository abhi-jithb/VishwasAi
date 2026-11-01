# Demo checklist

## 0. Before the panel (do it the night before, on the demo laptop)

```bash
cd ~/vishwas_ai
source venv/bin/activate
python -m pytest tests/test_offline.py -q          # expect: 38 passed
python scripts/verify_numbers.py --runs             # expect: 38 PASS, 0 FAIL
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:7b python scripts/demo.py   # warms the cache for the 3 demo examples
```

After `scripts/demo.py` has run once, the 3 demo examples come from `data/cache.db` and return in well under a second, **even with no internet**. Do not delete `data/cache.db`.

## 1. Start Ollama

```bash
systemctl status ollama          # Fedora service; "active (running)" is good
# if not running:
sudo systemctl start ollama      # or, in a spare terminal: ollama serve
ollama list | grep qwen2.5:7b    # model must be present (4.7 GB)
ollama run qwen2.5:7b "say ok"   # loads the model into memory (first load takes a while)
```

## 2. Environment variables for the demo (provider/model only, no keys)

```bash
export LLM_PROVIDER=ollama
export LLM_MODEL=qwen2.5:7b
export NO_MODEL_FALLBACK=1        # optional; Ollama never falls back anyway
```

The Tavily key (`SEARCH_API_KEY`) is read from `.env`. Do not show `.env` on screen.

## 3. Start the server

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
curl -s http://127.0.0.1:8000/api/health   # {"status":"ok"}
```

## 4. Demo script (in this order)

1. **Web UI**: paste demo example 2 (planted error). Expected verdicts are in `demo_expected_output.md`.
2. Turn on the **inspector** and rerun. Show the extracted claims, the search query, the 3 snippets, and the judge's raw JSON.
3. **CLI inspector**:
   ```bash
   python evaluate.py -q "Who wrote Romeo and Juliet?" \
     -a "William Shakespeare wrote Romeo and Juliet. Shakespeare was born in 1564 in Paris, France." --inspect
   ```
4. **Numbers are traceable**: `python scripts/verify_numbers.py --runs`. Every figure prints PASS.
5. **Ablation**: open `EVIDENCE/RETRIEVAL_ABLATION.md`.

## 5. If something fails

| Problem | Symptom | What to do |
|---|---|---|
| Ollama not running | `Failed to connect to local Ollama server at http://localhost:11434` | `sudo systemctl start ollama` or `ollama serve`. Then `ollama run qwen2.5:7b "ok"` to preload. |
| Ollama very slow | a new (uncached) answer takes minutes | Use only the 3 pre-warmed demo examples (cached, < 1 s). Say: "uncached, the 7B model runs mostly on CPU on this laptop." |
| No internet / Tavily down | search errors or empty evidence | Cached examples still work. For new inputs, the chain falls back Tavily → DuckDuckGo → Wikipedia automatically (proof: `EVIDENCE/search_fallback_proof.txt`). With no network at all, every claim becomes `unsupported` (no evidence). Explain that this is the intended safe behaviour. |
| Port 8000 busy | `address already in use` | `uvicorn backend.main:app --port 8001` |
| UI shows an error box | HTTP 500 with a message | Show the CLI instead (`evaluate.py --inspect`). The message is in the JSON `error` field. |
| Verdict differs from `demo_expected_output.md` | live web changed | Say: "the evidence changed. Here is the exact snippet the judge saw." The inspector shows it. |

## 6. Screenshots to take (save to `EVIDENCE/screenshots/`)

1. `ui_report.png`: web UI report page for demo example 2 (claim cards with ✓ / ✗ / ? and source links, summary F/V/U).
2. `ui_inspector.png`: the same page with the inspector panel expanded (extraction prompt/raw output, per-claim query, snippets, judge raw JSON).
3. `cli_inspect.png`: terminal output of `python evaluate.py ... --inspect` for the same example.
4. `verify_numbers.png`: terminal output of `python scripts/verify_numbers.py --runs` showing all PASS.
5. `pytest.png`: `python -m pytest tests/test_offline.py -q` showing 38 passed.
6. Optional: `search_fallback.png`: `SEARCH_API_KEY= DISABLE_CACHE=1 python scripts/prove_search_fallback.py`.
