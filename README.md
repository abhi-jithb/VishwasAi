# Vishwas AI

The answer already exists. Vishwas audits it.

Someone asked a model a question. The model wrote back in full, confident sentences. A wrong year can hide in the middle of a true paragraph, and asking that same model "are you sure?" only sends it back into its own memory. Vishwas stands outside that conversation. You hand it the question and the answer. It splits the answer into single facts, goes looking for evidence, and labels every fact.

- **supported**: the snippets say it
- **contradicted**: the snippets conflict with it
- **unsupported**: the snippets never decide it

Unsupported means no evidence was found. It does not mean the claim is false.

```
question + answer
       |
       v
 extract atomic claims          temperature 0, JSON
       |
       v
 search each claim              Tavily, then DuckDuckGo, then Wikipedia
       |
       v
 judge each claim               snippets only, or memory if retrieval is off
       |
       v
 three scores + a report        reason and source link on every claim
```

The checker is also a language model. It does not train one, and it does not chat. Hosted Groq, OpenAI, and Anthropic all plug in through `backend/llm.py`. So does a local Ollama model on `http://127.0.0.1:11434`. Repeated prompts are stored in `data/cache.db`, so the second demo of the same claim returns without running the model again.

On the saved 30-case run, local `qwen2.5:7b` with retrieval on extracted **82.29%** of the gold claims, agreed with the gold label on **68.35%** of the claims it matched, and agreed end to end on **56.25%**. That file is `results/gold_qwen2.5_7b_20261028_000521.json` (temperature 0, 3 snippets, 28 Oct 2026). The longer status note, including which numbers not to quote, is `docs/README.md`.

## Run an audit

From the project root, with the virtualenv on:

```bash
source venv/bin/activate
```

One answer, in the terminal:

```bash
python evaluate.py \
  --question "Who discovered penicillin?" \
  --answer "Alexander Fleming discovered penicillin in 1928."
```

Useful flags:

| Flag | What you get |
|---|---|
| `--no-retrieval` | The model judges from memory. No web search. This is the baseline. |
| `--inspect` | The prompts, the raw model text, the search query, and the snippets. This is the live demo. |
| `--json` | The same report as JSON. |

Force the local model for one command, without editing `.env`:

```bash
LLM_PROVIDER=ollama LLM_MODEL=llama3.2:3b \
  python evaluate.py --inspect \
  --question "Who discovered penicillin?" \
  --answer "Alexander Fleming discovered penicillin in 1928."
```

`llama3.2:3b` is the small model that fits a 4 GB GPU. The benchmark model is `qwen2.5:7b`. Pull it once, then swap the name:

```bash
ollama pull qwen2.5:7b
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:7b python evaluate.py -q "..." -a "..."
```

Ollama has to already be running (`ollama serve`, or the system service) on port 11434.

## Open the page

```bash
uvicorn backend.main:app --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

`POST /api/evaluate` takes `question`, `answer`, `retrieval`, and `inspect`. `GET /api/health` returns `{"status":"ok"}`. The retrieval and inspect switches on the page are the same flags as the CLI.

To serve the page with Ollama instead of whatever `.env` says:

```bash
LLM_PROVIDER=ollama LLM_MODEL=llama3.2:3b \
  uvicorn backend.main:app --port 8000
```

## Run the gold set

Thirty labelled answers live in `data/gold_claims.json` (96 claims: correct, planted error, fabricated entity, outdated fact, long multi-fact). The runner writes a scorecard under `results/`.

```bash
python tests/run_gold.py
python tests/run_gold.py --no-retrieval
python tests/run_gold.py --provider ollama --model qwen2.5:7b --limit 3
```

`--limit` is for a smoke test. Cite a paper number only from a saved file that ran all 30.

Print a table of whatever scorecards are already on disk:

```bash
python scripts/compare_benchmarks.py
```

Check that the model server and the search client answer at all:

```bash
python -m scripts.test_llm
python -m scripts.test_search
```

## Setup

Python 3.11 or newer.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set:

| Variable | Role |
|---|---|
| `LLM_PROVIDER` | `ollama`, `groq`, `openai`, or `anthropic` |
| `LLM_MODEL` | `qwen2.5:7b` for Ollama, or the hosted model id |
| `OLLAMA_BASE_URL` | Default `http://127.0.0.1:11434` |
| `LLM_API_KEY` | Required for hosted providers. Unused for Ollama. |
| `SEARCH_API_KEY` | Tavily. If it is empty, search falls through to DuckDuckGo and then Wikipedia. |

Keep `.env` out of git. The cache file `data/cache.db` is created on the first call.

## What the three scores mean

Counted over the claims extracted from one answer:

- **Factuality** = supported / total. Drops when a claim is only unsupported.
- **Verifiable precision** = supported / (supported + contradicted). Ignores claims with no evidence. Empty if nothing was verifiable.
- **Unverifiable rate** = unsupported / total.

The gold runner adds two more, against the labelled file: extraction recall (did we find the gold claim) and end-to-end accuracy (did we find it and label it correctly). End-to-end is the number for the whole system. Verification accuracy is only the judge, on claims the extractor actually produced.

## Map

```
vishwas_ai/
├── evaluate.py                 # CLI audit
├── backend/
│   ├── main.py                 # FastAPI: /api/evaluate, /api/health, and the page
│   ├── llm.py                  # the only door to Ollama, Groq, OpenAI, Anthropic
│   ├── search.py               # Tavily, DuckDuckGo, Wikipedia
│   ├── cache.py                # SQLite cache for model and search calls
│   ├── config.py               # reads .env
│   └── evaluation/
│       ├── prompts.py          # extractor and judge instructions
│       ├── claims.py           # split the answer
│       ├── evidence.py         # turn a claim into a query
│       ├── verify.py           # label one claim
│       ├── scoring.py          # the three scores
│       └── pipeline.py         # runs the four stages
├── frontend/index.html         # the report page
├── data/gold_claims.json       # 30 labelled cases
├── tests/run_gold.py           # benchmark runner
├── scripts/                    # connectivity checks, comparison, error report
├── results/                    # saved scorecards
└── docs/
    ├── README.md               # what is done, what to run next, which numbers to cite
    ├── PITCHING.md             # the viva: transformers, terms, and the product path
    └── HOW_IT_WORKS.md         # Ollama, GGUF, tokens, temperature, context
```

## Read next

- [docs/PITCHING.md](docs/PITCHING.md) if you have to explain attention, temperature, or why unsupported is not false
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) if the questions are about the local model file and the GPU
- [docs/README.md](docs/README.md) if you need the work log and the honest limits
