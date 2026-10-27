# Pitching Vishwas AI, and how the LLM inside it works

Use this when you have to explain the project out loud. `docs/README.md` says what is finished and which numbers you may quote. `docs/HOW_IT_WORKS.md` is the shorter note on Ollama and quantization. This file is the full explanation: the idea, the transformer, the words teachers will ask you to define, and the path one answer takes through the code.

## The pitch

People trust a fluent answer. Fluency is not evidence. A wrong year, a wrong person, or an invented organisation can sit inside a paragraph that is otherwise true. Asking the same model "is this right?" asks it to consult the same memory that produced the answer.

Vishwas AI does not write the answer. Some other model already did. Vishwas takes the question and that answer, splits the answer into separate checkable facts, fetches web snippets for each fact, and labels every fact:

- **supported**: the snippets state the fact
- **contradicted**: the snippets state something that conflicts with it
- **unsupported**: the snippets do not decide it

Unsupported means "no evidence in what we retrieved." It does not mean "false."

The output is a report: each claim, why it got that label, the links used, and three scores that keep "we found nothing" separate from "we found a contradiction."

You can say this in one sentence: Vishwas is an evidence-linked audit of an answer that already exists.

## What it is for, and where it sits

A chatbot's job is to continue your prompt with the next likely words. A RAG system retrieves documents so the chatbot can write a better answer. Vishwas runs after the answer exists. Retrieval here is for the auditor, so each claim can point at a source.

That puts it in the same family as claim-level checkers such as FActScore, SAFE, FacTool, and VeriScore. Sampling-based checks such as SelfCheckGPT ask the model again and look for agreement. Benchmarks such as FEVER, TruthfulQA, HaluEval, and RAGTruth are datasets, not products. Say you are building a small audit pipeline on a 30-case set you labelled. Do not say you invented fact-checking.

## Two different ways the project uses an LLM

The project never trains a model. It calls a model that already exists.

| Use | Who talks | What the model is allowed to do |
|---|---|---|
| The system being audited | Any LLM, outside Vishwas | Write the answer. Vishwas does not control it. |
| The extractor | The model configured in Vishwas | Read the question and answer. Return a JSON list of atomic claims. |
| The judge, retrieval on | The same Vishwas model | Read one claim plus the snippets. Return supported, contradicted, or unsupported. The prompt tells it not to use outside memory. |
| The judge, retrieval off | The same Vishwas model | Read the claim only. Judge from its own stored knowledge. This is the baseline. |
| Search | No LLM | Tavily, then DuckDuckGo, then Wikipedia. |

Hosted Groq and local Ollama are two transports to that same extractor and judge. `backend/llm.py` is the only place those calls are made. The rest of the pipeline asks `call_llm(system, user)` and gets text back.

On this Fedora laptop the model that fits the GPU completely is `llama3.2:3b`. The model in the saved 30-case file is `qwen2.5:7b`. Both are decoder-only transformers in GGUF files. The 7B file is larger than 4 GB of VRAM, so Ollama runs part of it on the GPU and part in RAM.

## How a transformer LLM actually produces text

A large language model here is a decoder-only transformer. "Decoder-only" means it does one job: given the tokens so far, score every token in the vocabulary for what comes next. It then appends one token and repeats. That loop is called autoregressive generation.

The published shape of `qwen2.5:7b`, which is the benchmark model, is the one already used in `docs/HOW_IT_WORKS.md`: each token is a vector of width 3584, the feed-forward layer expands that toward about 18944, and the final layer scores on the order of 152000 vocabulary items. The context window is 32768 tokens. Those are properties of that model family. This project did not measure them off the GPU.

### 1. Tokens

The model does not see letters or words. A tokenizer, built with byte-pair encoding, cuts text into pieces that occurred often in training. A common word may be one token. A rare word may be several. Each piece has an integer id.

The prompt you send and the JSON you hope to get back are both just more tokens. "Return only a JSON array" works when the model has seen enough JSON in training, and when the server also constrains the grammar. Ollama's `format: json` is that extra constraint. It blocks tokens that would break JSON. It does not make the facts true.

### 2. Embeddings

The integer id is looked up in a table and becomes a vector of 3584 numbers for Qwen2.5-7B. Vectors that the training run placed near each other tend to be tokens that appeared in similar contexts. That is a geometric habit of the model, not a dictionary definition.

Order is not in the id. Rotary position embeddings (RoPE) rotate pairs of dimensions by an angle that depends on the token's position, so "1928" before "Fleming" and "1928" after "Fleming" are different inputs.

### 3. One transformer block

The same block is stacked many times. Each block does two things, and each is wrapped in a residual connection: add the block's output back onto its input, so the original signal is never thrown away. A normalisation step (RMSNorm in Qwen) rescales the vector before the heavy math so training stays stable.

**Self-attention.** For every token the block builds three vectors:

- **Query**: what this token is looking for
- **Key**: what this token offers as a label
- **Value**: the content to copy if the match is strong

Scores are the dot products of queries and keys, divided by the square root of the key width so the values do not explode, then passed through softmax so they become weights that add up to 1. The output is the weighted sum of the values.

In symbols, for one head: softmax(Q K transpose / sqrt(d_k)) times V.

Multi-head attention runs several of these lookups with different learned projections, then concatenates them. A token such as a year can attend to the person and the verb in the same claim. Attention is how the claim "in 1928" stays tied to Fleming rather than to whatever other number is in the prompt. It is not a fact database. It is a weighted mix of the tokens that are already in this prompt, plus whatever the later feed-forward layer adds from the weights.

Qwen2.5 uses grouped-query attention: many query heads share fewer key and value heads. That cuts the size of the cache of past keys and values. The idea of attention stays the same.

**Feed-forward network.** After attention, each token is transformed on its own. SwiGLU is the variant used here: two linear layers, a gating activation on one of them, then a projection back to width 3584. Most of the model's parameters live in these layers. This is where associations stored during training get applied: names, dates, typical phrases.

Stack the blocks and the last vector at the newest position is a summary of the whole prompt, shaped by every earlier token.

### 4. The next token

A final linear layer, the language-model head, turns that vector into one score per vocabulary item. Those scores are logits. Softmax turns logits z into probabilities:

P(token i) = exp(z_i) / sum_j exp(z_j)

The chosen token is appended. The model runs again. It stops when it emits the end-of-message token or hits a length limit.

**Temperature** divides the logits before softmax. Temperature 0, which is what Vishwas sets, keeps the winning token and drops the rest. That is greedy decoding. The same prompt is meant to produce the same string, which is what you want from an auditor. A temperature like 0.7 flattens the distribution so less likely tokens get a chance. That is useful for varied writing. It is a bad default for a label that must be repeated in a benchmark.

**Context length** is the maximum number of tokens in the prompt plus the tokens still being generated. Attention compares the new token with every earlier token, and the keys and values of earlier tokens are kept in a KV cache so they are not recomputed. The cache grows with the length of the prompt. Vishwas cuts each snippet at 400 characters so a single claim check stays small. That is a product choice about memory and noise, not a limit of Qwen's 32768-token window.

### 5. Why a smaller model misses more

A 3B model and a 7B model are the same kind of machine with fewer or more of those weights. Fewer weights means a coarser store of associations and a weaker ability to follow a format. On a live run, `llama3.2:3b` reduced "Alexander Fleming discovered penicillin in 1928" to the claim `1928`. The judge then marked that fragment unsupported, which was reasonable for the fragment and useless for the original sentence. The extractor failed, and the judge did what it was told with a bad claim.

The saved 30-case run of `qwen2.5:7b` still missed 17 of 96 gold claims. Bigger is better at this job on that file. It is not reliable.

### 6. Quantization, GGUF, and Ollama

Training stores weights as 16-bit floats. A 7B model at 16-bit is on the order of 14 GB. Quantization buckets those weights into 4-bit codes plus a scale, which is why a Q4_K_M file is about 4.7 GB. The file format is GGUF: header, tokenizer, and quantized tensors in one file.

Ollama is the local server. It is already installed here and listens on `http://127.0.0.1:11434`. `ollama pull qwen2.5:7b` downloads that GGUF from Ollama's library into `/var/lib/ollama/models/blobs/`. After the pull, generation does not need the internet. The Python code only POSTs to `/api/chat`. The `ollama serve` process starts a runner subprocess that executes the model with CUDA on the RTX 3050, spilling leftover layers into system RAM when the file does not fit in 4 GB.

`qwen2.5:7b` is that local file. `qwen/qwen3.8-27b` in a Groq result is a different, hosted model. The similar names do not make them the same run.

## Words you will be asked to define

**LLM.** A network trained to predict the next token. In this project it is a tool with a text in, text out interface.

**Prompt.** The text of that request. Vishwas sends a system prompt (the rules) and a user prompt (the question, the answer, or the claim and the snippets). Both live in `backend/evaluation/prompts.py`.

**Hallucination.** A fluent statement that is not supported by the world or by the documents you gave the model. Vishwas does not detect "hallucinations" as a special internal state. It labels claims against snippets. A supported claim can still be wrong if the web page is wrong.

**Atomic claim.** One sentence with one checkable fact, names spelled out, no "he" or "it." "Fleming discovered penicillin in 1928 and won a Nobel Prize in 1945" must become two claims, because one half can be true and the other false.

**Evidence / snippet.** A title, a URL, and at most 400 characters of text from a search result. It is not the whole page.

**Retrieval.** Fetching those snippets. Retrieval on is the audit. Retrieval off is the memory baseline.

**Entailment.** The snippets state the claim, or state something from which the claim follows directly. The judge prompt calls the positive case "supported." A cross-encoder NLI model would use the word entailment for the same idea. That model is not in the repo.

**Gold claim.** A claim a human wrote as the correct split of a test answer, with the label they believe is right. The file is `data/gold_claims.json`.

**Recall (extraction).** Matched gold claims divided by all gold claims. Matching is token overlap, not an exact string compare, in `tests/run_gold.py`.

**Verification accuracy.** Among gold claims that matched an extracted claim, the share whose status also matched the gold label.

**End-to-end accuracy.** Gold claims whose extracted match also had the right status, divided by every gold claim, including the ones the extractor dropped. This is the number for the whole system.

**Baseline.** The same judge with no snippets. If you never run it, you cannot tell whether web evidence helped.

**Cache.** SQLite tables in `data/cache.db`. The key is a SHA-256 hash of the provider, model, prompts, temperature, and JSON flag for an LLM call, or of the query and result count for a search. The same input returns the stored string. A 0.02 second case in an old result file is a cache hit, not a fast model.

**Temperature 0.** Greedy next token. Set on every `call_llm` in the extractor and the judge.

**JSON mode.** Ollama is asked for `format: json`. The parser in `claims.py` and `verify.py` still strips fences and hunts for an array or object, and retries once, because local models still wrap or reshape JSON.

**Factuality, verifiable precision, unverifiable rate.** Defined in the next section. They are claim counts, not probabilities from the model.

**Provider.** The server behind `call_llm`: `ollama`, `groq`, `openai`, or `anthropic`.

**Benchmark model.** If `BENCHMARK_MODEL` is set, that exact name is the model, and the Groq code path does not try a list of substitutes.

## How the product works, stage by stage

```
question + answer
        |
        v
extract_claims          one LLM call, temperature 0, JSON array
        |
        v
for each claim, two at a time:
        gather_evidence     only if retrieval is on
        verify_claim        LLM judge, or an immediate "unsupported"
        |
        v
compute_factuality
        |
        v
JSON report  ->  CLI text, or the web page
```

Entry points are `evaluate.py` and `POST /api/evaluate` in `backend/main.py`. Both call `evaluate()` in `backend/evaluation/pipeline.py`. The page is static files from `frontend/`. Health is `GET /api/health`.

### Stage 1. Extract

`extract_claims` fills the user prompt with the question and the answer. The system prompt requires factual claims only, one fact each, pronouns replaced, and a JSON array of strings.

`call_llm` checks the cache, then POSTs to Ollama or the hosted SDK. The raw string is parsed. If it is a dict, the code looks for a list under a key such as `claims`, or takes string values. If parsing fails, it sends the broken text back once with a stricter prompt. That retry is a second LLM call and shows up in the cost log.

A bad extraction cannot be saved by a good judge. The judge only sees the claims the extractor emitted.

### Stage 2. Evidence

`gather_evidence` strips quotes and a trailing full stop and cuts a claim longer than 200 characters. That string is the query. `web_search` checks the search cache, then Tavily if a search key is configured, then DuckDuckGo, then the Wikipedia API. Each snippet is shortened to 400 characters. The pipeline asks for 3 results per claim.

Claims are processed two at a time so a laptop is not asked to run many model calls at once. Search can overlap. The Ollama runner itself is one model load.

If retrieval is off, this stage is skipped and the evidence list is empty on purpose.

### Stage 3. Judge

If retrieval is on and the evidence list is empty, `verify_claim` returns `unsupported` with the reason "No relevant search evidence was found for this claim." There is no LLM call.

Otherwise the snippets are numbered with title, URL, and text. The judge must return JSON:

```json
{"status": "supported", "reasoning": "...", "evidence_used": ["https://..."]}
```

The parser accepts a status that contains the words supported, contradicted, or unsupported. Anything else becomes unsupported. For an unsupported verdict the source list is cleared. For the other two, URLs are kept when they match a URL that was actually in the snippets.

Retrieval off uses a different prompt. The model is told to use its own knowledge, and `evidence_used` is empty. That run answers a different question: what does this model believe, with no page in front of it?

The prompt can say "use only the snippets." The weights can still lean on training memory. You have not measured how often that happens. Say that if you are asked whether the judge is purely extractive.

### Stage 4. Scores

From `backend/evaluation/scoring.py`, over the claims the pipeline produced:

- factuality = supported / total
- verifiable precision = supported / (supported + contradicted), or empty if both are zero
- unverifiable rate = unsupported / total

A worked example. Four claims: two supported, one contradicted, one unsupported.

- factuality = 2/4 = 0.50
- verifiable precision = 2/3 = 0.67
- unverifiable rate = 1/4 = 0.25

Factuality falls when a claim is merely unverifiable. Verifiable precision ignores those and asks, of the claims the evidence actually spoke about, how many were supported. An answer can have a low factuality score because search failed, while every claim that had evidence was fine. The three numbers exist so you do not hide that difference inside one accuracy percentage.

The gold runner's recall and end-to-end accuracy are a separate comparison against `data/gold_claims.json`. They are not the three scores above. The three scores describe one answer. Recall and end-to-end describe the system against the labelled set.

### Stage 5. The report and the inspector

The CLI prints each claim with a mark, the reason, and the links, then the three scores and the count of LLM and search calls.

`--inspect`, and the inspect switch on the page, also keep the extraction prompt, the raw extraction text, each search query, the snippets, the judge prompt, and the judge's raw JSON. That is the live demo. You show the middle, not only the final percentage.

The cost log counts calls. With the cache on, a repeated demo can show calls in the log while Ollama does no new work, because the call function still records the request and then returns the stored body. If a demo is instant, say it was cached.

## The gold set, in the language of the pitch

`data/gold_claims.json` has 30 answers and 96 gold claims in five buckets:

- **correct**: the answer's facts are true
- **planted_error**: a true-looking answer with a fact that was deliberately made false
- **fabricated_entity**: a name or organisation that should not survive a check
- **outdated_fact**: something that used to be true, or a current fact the web may still state badly
- **long_multi_fact**: many facts in one answer, so extraction has more chances to merge or drop one

The file you can quote is `results/gold_qwen2.5_7b_20261028_000521.json`. Local `qwen2.5:7b`, Ollama, temperature 0, retrieval on, 3 snippets, 2026-10-28 00:05, recorded on a Windows laptop with a Ryzen 7, 24 GB RAM, and an RTX 5050 4 GB. Extraction recall 82.29%. Verification accuracy 68.35% on the 79 matched claims. End-to-end 56.25%. Outdated facts agreed on 2 of 9. The error report for that file is a script, not a second human: every miss is called bad extraction, and an unsupported disagreement is called bad search. Details and the numbers you must not quote are in `docs/README.md`.

## What a teacher is probing when they ask "why"

**Why not one verdict for the whole answer?** Because a paragraph can be mostly true. The unit of failure is the claim.

**Why search, if the model knows things?** Because the model's memory is how the original answer was produced, and it goes stale. The baseline exists to measure that. The 30-case no-retrieval file for `qwen2.5:7b` is not in `results/` yet.

**Why can unsupported not mean false?** Because an empty search, a bad query, or a snippet about a different person is lack of evidence. Calling it false would punish the answer for your retrieval.

**Why temperature 0?** So a benchmark row is a property of the prompt and the weights, not of a random sample.

**Why a local model at all?** So you can point at the process: tokenizer, GPU runner, GGUF file on disk, and a server on port 11434. A hosted API hides that. The hosted model is still useful as a larger comparison, once a full 30-case file for it exists. The 5-case Groq file in `results/` is only easy true claims.

**Why is 56% end-to-end not a failure of the talk?** It is the measurement. The interesting part is where the misses sit: extraction on ordinary facts, search and judging on outdated and long answers. A project that only reports a near-perfect score on its own labels has less to explain.

## A two-minute version

Vishwas checks an answer another model already wrote. It splits that answer into single facts, searches the web for each one, and labels the fact supported, contradicted, or unsupported using the snippets. Unsupported means we did not find evidence. Three scores keep "no evidence" apart from "evidence of a contradiction." The checker itself is also an LLM, a quantized transformer run locally by Ollama, at temperature 0, with JSON constrained and cached. On the 30-case file, Qwen 2.5 7B with retrieval agreed end to end on 56% of the gold claims and extracted 82% of them. The set is small and self-labelled, search is not ground truth, and the judge may still lean on its training memory. I am not claiming a new kind of model. I am claiming a report you can open, claim by claim, and argue with.
