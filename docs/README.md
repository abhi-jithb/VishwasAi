# Vishwas AI: what is done, and what to run next

This file is the project status and the order of work. `docs/HOW_IT_WORKS.md` is the viva explanation of the local model. Numbers below are copied from files in `results/`. If a number is not in one of those files, do not put it in the paper or the slides.

Vishwas AI is a trust check for an answer some other model already wrote. It is not a chatbot and it is not a new language model. You give it a question and an answer. It splits the answer into atomic facts, looks up evidence for each fact, labels each fact, and prints a report.

A label of **unsupported** means no usable evidence was found. It does not mean the claim is false.

## What the pipeline does

1. **Extract.** An LLM splits the answer into self-contained factual claims. Pronouns are replaced with names. Opinions and hedges are dropped. One checkable fact per claim.
2. **Retrieve.** Each claim becomes a web query. Search order is Tavily, then DuckDuckGo, then Wikipedia. Up to 3 snippets, each cut at 400 characters. Snippets are cached.
3. **Judge.** With retrieval on, the judge may use only those snippets. If the search returns nothing, the claim is `unsupported` and the judge is not called. With retrieval off, the same model judges from its own memory. That second mode is the baseline.
4. **Score.**
   - Factuality = supported / total claims
   - Verifiable precision = supported / (supported + contradicted). Empty if nothing was verifiable.
   - Unverifiable rate = unsupported / total claims
5. **Report.** Each claim is shown with a status, a short reason, and the source links the judge cited.

Temperature is 0 so the same prompt is meant to give the same tokens. Repeated prompts are stored in `data/cache.db` (SQLite). A cache hit does not call the model again.

LLM calls go through `backend/llm.py` only. Providers already wired there: Ollama on this machine (`http://127.0.0.1:11434/api/chat`), Groq, OpenAI, and Anthropic. When `BENCHMARK_MODEL` is set, the Groq path does not switch to a different model.

## What is already built

| Piece | Where | State |
|---|---|---|
| Claim extraction, evidence, judge, scores | `backend/evaluation/` | Working |
| One LLM wrapper | `backend/llm.py` | Working, including local Ollama |
| Disk cache for LLM and search | `backend/cache.py`, `data/cache.db` | Working |
| Web search with fallbacks | `backend/search.py` | Working |
| HTTP API and page | `POST /api/evaluate`, `GET /api/health`, `frontend/index.html` | Working |
| CLI | `evaluate.py`, including `--no-retrieval` and `--inspect` | Working |
| Gold set | `data/gold_claims.json` | 30 cases, 96 gold claims, 5 categories |
| Gold runner | `tests/run_gold.py` | Writes `results/gold_<model>_<timestamp>.json` and records model, time, hardware, retrieval |
| Inspector | CLI `--inspect` and a panel in the page | Built. No saved demo trace in `results/` |
| Error-analysis script | `scripts/generate_error_analysis.py` | Built. One report saved |
| Comparison script | `scripts/compare_benchmarks.py` | Built. It only prints a table from whatever JSON is in `results/`. It is not itself a result |
| Viva notes for the local model | `docs/HOW_IT_WORKS.md` | Stage A only: Ollama, GGUF, tokens, temperature, context, cache |

Not built: a fixed 3-example demo script, a full local no-retrieval run, a hosted-versus-local comparison you can cite, self-consistency, confidence scores, source-reliability tiers, and the J-Space experiment.

## Numbers you may quote

Quote the file name, the model, the date, retrieval on or off, and the machine in that file. The two machines are not the same.

### Full local run (the one that matters)

File: `results/gold_qwen2.5_7b_20261028_000521.json`

- Model: `qwen2.5:7b` through Ollama, temperature 0, retrieval on, 3 snippets max
- When: 2026-10-28 00:05
- Machine recorded in the file: Windows 11, AMD Ryzen 7 250, 24 GB RAM, NVIDIA GeForce RTX 5050 Laptop (4 GB VRAM)
- 30 cases, 96 gold claims, 79 matched
- Claim extraction recall: **82.29%** (17 missed)
- Verification accuracy on matched claims: **68.35%** (54 agreed, 25 disagreed)
- End-to-end accuracy: **56.25%** (agreed claims / all 96 gold claims)
- Time: 742 seconds total, about 24.7 seconds per case

Verification accuracy ignores claims the extractor never produced. End-to-end accuracy counts those misses as failures. Use end-to-end when you talk about the whole system. Use verification accuracy only when you talk about the judge on claims that were actually extracted.

Per category in that same file (agreed / gold claims):

| Category | Gold | Matched | Agreed | Disagreed | Missed |
|---|---:|---:|---:|---:|---:|
| correct | 19 | 13 | 11 | 2 | 6 |
| planted_error | 18 | 12 | 7 | 5 | 6 |
| fabricated_entity | 13 | 12 | 10 | 2 | 1 |
| outdated_fact | 9 | 6 | 2 | 4 | 3 |
| long_multi_fact | 37 | 36 | 24 | 12 | 1 |

Outdated facts are the weak category: 2 of 9 agreed. Long answers extract well (36 of 37 matched) and then the judge still disagrees on 12. That split is useful in the talk. Extraction and judging fail in different places.

### Error labels for that same run

File: `results/error_analysis.md` (same model, same timestamp)

The script counted 42 errors: 17 bad extraction, 19 bad search, 6 judge, 0 ambiguous gold. Those 42 are the 17 misses plus the 25 disagreements.

Treat this as a **heuristic**, not a hand audit. The script labels every missed claim as bad extraction, and it labels every `unsupported` disagreement as bad search without checking whether the snippets were actually empty or just irrelevant. Say that if a teacher asks how the causes were assigned.

### Do not cite these as the system result

`results/gold_qwen2.5_7b_20261027_230737.json` is 1 case. Average latency is 0.02 seconds, which is a cache replay, not a model run.

`results/gold_qwen_qwen3.8_27b_no_retrieval_20261028_012405.json` is Groq `qwen/qwen3.8-27b`, retrieval off, **5 cases, all in the category `correct`**, on this Fedora machine (16 GB RAM, RTX 3050 Laptop 4 GB). It scores 100% on those 5. That is a smoke test of easy true claims. It is not a no-retrieval baseline and it is not comparable to the 30-case local run.

The draft paper figures **98.96% extraction recall** and **96.8% verification agreement** are not in this repository. No saved file reproduces them. Do not put them on a slide unless you find the exact run, model, date, and settings. The saved 30-case local run is lower, and that is the number the files support.

## This laptop, right now

Checked on this machine (not read from a result file):

- Fedora, Intel i5-11400H, 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM
- Ollama is installed and listening on `127.0.0.1:11434`
- Models present: `llama3.2:3b`, `qwen2.5-coder:3b`, `qwen2.5-coder:1.5b-base`, `nomic-embed-text`
- `qwen2.5:7b` is **not** installed here. A pull was started and parts of it stalled

The 30-case `qwen2.5:7b` file was produced on the other laptop (RTX 5050, 24 GB RAM). You can cite that file. You cannot re-run it here until `ollama pull qwen2.5:7b` finishes. `llama3.2:3b` fits entirely on this 4 GB GPU and is the model to use for a live demo on this machine. It is a weaker extractor. A short demo on it earlier kept only the year `1928` from "Alexander Fleming discovered penicillin in 1928" and then correctly marked that fragment unsupported.

Largest sensible local model for this GPU: `qwen2.5:7b`. It will not fit wholly in 4 GB, so Ollama puts some layers on the GPU and the rest in RAM. `llama3.1:8b` is a tighter fit. Leave it.

```bash
ollama pull qwen2.5:7b
ollama list
```

## What to say the system is, and what not to claim

Vishwas is a small, claim-level audit pipeline with an evidence-linked report and a gold set you labelled yourselves. Place it next to existing work. Do not say it is the first fact checker, the first retrieval checker, or a hallucination detector.

Systems to position against, by the job they do:

- **FActScore, SAFE, FacTool, VeriScore.** Break text into claims and check them. Vishwas is in this family. The difference you can actually defend is the report shape: per-claim status, a reason, source links, and three separate scores, plus a no-retrieval baseline once that run exists.
- **CRITIC, SelfCheckGPT.** Use the model again, or sample it many times, without an external document. Your no-retrieval mode is the same idea. Retrieval is the part that does not trust the model's memory. You have not measured self-consistency yet, so do not claim it.
- **HaluEval, RAGTruth, TruthfulQA, FEVER.** Benchmarks, not systems. Yours is 30 self-labelled cases. Say that. Do not compare your 56% or 68% to a published score on a different dataset.
- **RAG.** Retrieval used to help a model write an answer. You retrieve to audit an answer that already exists.

Honest limits, already visible in the files:

- 30 cases, partly self-labelled. A high score on this set is not a general accuracy claim.
- Web snippets are not ground truth. A page can be wrong, old, or off-topic.
- Outdated facts are the category the local model handles worst (2 of 9).
- The judge is still an LLM. The prompt says "use only the snippets." The model can ignore that. You have not proved it didn't.
- The extractor merges and drops claims (17 misses on the local run).
- The no-retrieval baseline on the same model and the same 30 cases does not exist yet.
- The error-cause table is an automatic rule, not a second annotator.
- Search needs the network. Only the model can be fully local.

## What to execute next

Do these in order. Each one should leave a file in `results/` before you talk about it. Two days is enough for the first four. It is not enough for a trained model or a J-Space paper.

### 1. Finish the local model on the demo machine

```bash
ollama pull qwen2.5:7b
ollama list
```

Until that name appears, demo with `llama3.2:3b` and say it is the small model, not the benchmark model.

### 2. Run the missing baseline

Same model as the 30-case file, retrieval off, all 30 cases. This is the comparison the paper is missing: does web evidence help, or is the model just remembering?

```bash
source venv/bin/activate
BENCHMARK_MODEL=qwen2.5:7b LLM_PROVIDER=ollama \
  python tests/run_gold.py --no-retrieval
```

Expect a new file `results/gold_qwen2.5_7b_no_retrieval_<timestamp>.json`. Read recall, verification accuracy, and end-to-end accuracy from that file. If retrieval-on end-to-end (56.25%) is not higher, say so. A result where retrieval does not help is still a result.

On this 4 GB GPU a full 30-case run can take longer than the 12 minutes recorded on the other laptop, because part of a 7B model sits in RAM.

### 3. Print the comparison table from saved files only

```bash
python scripts/compare_benchmarks.py
```

Keep the 1-case cache replay and the 5-case Groq smoke test out of the paper table. The paper row that is allowed today is one row: local `qwen2.5:7b`, retrieval on, 30 cases. Add the no-retrieval row only after step 2. Add a hosted 70B row only after a real 30-case file for that model exists. There is no such file now.

### 4. Three demo examples, inspector on

Use one correct case, one planted error, and one fabricated entity from `data/gold_claims.json`. Run them with `--inspect` so you can show the extracted claims, the query, the snippets, the exact judge prompt, and the raw JSON. Practice this on `llama3.2:3b` if the 7B pull is not finished. If the small model extracts a bad claim, show that. It supports the "small models drop facts" point already written in `docs/HOW_IT_WORKS.md`.

```bash
LLM_PROVIDER=ollama LLM_MODEL=llama3.2:3b \
  python evaluate.py --inspect \
  --question "..." \
  --answer "..."
```

The page is the same pipeline:

```bash
LLM_PROVIDER=ollama LLM_MODEL=llama3.2:3b \
  uvicorn backend.main:app --port 8000
```

Open `http://127.0.0.1:8000`. Leave the inspect box on.

### 5. Rewrite the result paragraph to match the files

Replace 98.96% and 96.8% with the 30-case local numbers, and state the machine that produced them. Add one sentence on the error heuristic. Add the no-retrieval numbers after step 2, including the case where evidence does not win.

## Later, after the baseline exists

Do not start these before step 2 is a saved 30-case file.

- **Hosted comparison.** One pinned Groq model, all 30 cases, retrieval on, `BENCHMARK_MODEL` set so it cannot fall back to another model. Save the JSON. Then the table has a real "big hosted model vs local 7B" row.
- **Rerun the error script on the new files** and say again that the labels are automatic.
- **Self-consistency and confidence.** Not implemented. Several samples at temperature above 0, agreement across samples as a confidence score. Only worth doing if the baseline table is already in the paper.
- **Source-reliability tiers.** Not implemented. Would rank a source (for example an encyclopaedia page above an unknown blog). Needs a written rule before any code, or it becomes an unmeasured claim.
- **SQLite for audit history.** The cache is already SQLite. A separate store of past reports is a product feature, not a result.
- **Passage reranking and an NLI cross-encoder.** Not in the repo. These replace or second-guess the LLM judge with a small local model (`all-MiniLM-L6-v2` for similarity, a cross-encoder such as `nli-deberta-v3-base` for entailment / contradiction / neutral). Useful only if you have time to run them on the same 30 cases and add rows to the table. Skip them if the presentation is in two days.
- **J-Space / J-Lens.** Your question, not a claim the system already tests: after the external pipeline works, do internal features from a Jacobian lens add anything about answer reliability beyond retrieval, the judge, and the no-retrieval baseline? A null result is a valid outcome. Do not describe Vishwas as using J-Space today. This session did not verify the Anthropic paper or the `idhantgulati/j-lens` code. Read those yourself before citing them, and do not invent page numbers or scores.

## Short viva path

If the questions stay on design, walk this and stop:

1. The input is someone else's answer, not a chat turn.
2. Claims are split so one wrong detail cannot hide inside a true sentence.
3. Retrieval is there because asking the same model "is this true?" reuses its memory. The no-retrieval run is how you show that difference, once it exists.
4. Unsupported is "no evidence," and an empty search never calls the judge.
5. Three scores, because a single accuracy number mixes "we found nothing" with "we found a contradiction."
6. The local model is a quantized GGUF file served by Ollama. The details are in `docs/HOW_IT_WORKS.md`.
7. The measured system, on the file you have, extracts about 82% of the gold claims and, end to end, agrees on about 56%. Outdated facts and search quality are the weak points. The benchmark is small and self-labelled.
