# Vishwas AI: Technical Architecture & Viva Study Guide

This document explains every core component, machine learning concept, and engineering decision in **Vishwas AI**. It is designed as an accessible, comprehensive reference for project reviews and oral examinations (viva).

---

## Stage A: Local LLM Execution & Architecture

### 1. What is Ollama?
**Ollama** is an open-source inference engine and local LLM runtime built on top of `llama.cpp`. 

* **Why we use it:** Instead of making paid API calls to remote cloud providers (Groq, OpenAI, Anthropic), Ollama runs the weights directly on your local hardware (GPU and system RAM).
* **How it works:** 
  1. Ollama runs a background service (daemon) listening on `http://localhost:11434`.
  2. It exposes REST API endpoints—specifically `/api/chat`—which accept standardized JSON payloads containing system prompts, user messages, and hyperparameter options (like `temperature`).
  3. **Hybrid Offloading:** If a model is too large to fit entirely into GPU VRAM (for example, a 4.7 GB model on a 4 GB GPU), Ollama automatically offloads as many transformer layers as possible to the NVIDIA GPU (via CUDA) and places the remainder in System RAM (via CPU AVX/AVX2 instructions).
  4. **Strict JSON Mode:** Ollama supports grammar-guided sampling via `"format": "json"`. It enforces token generation to strictly follow valid JSON syntax at the logit level, preventing malformed outputs.

---

### 2. What is a Quantized GGUF Model?
In deep learning, neural network weights are typically trained and stored as 16-bit or 32-bit floating point numbers (`FP16` or `FP32`):
* A 7-billion parameter model in full 16-bit precision requires:
  $$\text{Memory} \approx 7 \times 10^9 \text{ parameters} \times 2 \text{ bytes} \approx 14\text{ to }15\text{ GB of VRAM}$$
  This would exceed the 4 GB VRAM of consumer laptops.

* **Quantization:** Quantization maps high-precision floating point numbers to lower-precision integers (typically 4-bit integers, known as `INT4` or `Q4_K_M`):
  $$\text{Quantized Memory} \approx 7.6 \times 10^9 \text{ parameters} \times \approx 0.6 \text{ bytes/param} \approx 4.7\text{ GB}$$
  Quantization clusters weight values into bins and stores scale factors. Research shows that modern 4-bit quantization (`Q4_K_M`) preserves over 98% of the model's original reasoning accuracy while reducing memory usage by ~70% and accelerating memory bandwidth throughput.

* **What is GGUF?**
  **GGUF (GPT-Generated Unified Format)** is a binary file container specification created by Georgi Gerganov and the `llama.cpp` community. It packages everything needed to run an LLM into a single portable file:
  1. Header with model metadata (architecture, number of layers, attention heads).
  2. Complete tokenizer vocabulary and merges.
  3. Pre-quantized tensor weight data.

---

### 3. How a Transformer Turns a Prompt into Tokens and Generates Next Tokens
The model powering our local pipeline (`qwen2.5:7b`) is an autoregressive decoder-only Transformer. Here is the step-by-step lifecycle of a request:

```
User Prompt ("Alexander Fleming discovered penicillin")
                     ↓
         [1. Byte-Pair Encoding (BPE) Tokenizer]
                     ↓
      Token IDs: [15432, 42109, 3401, 8912]
                     ↓
        [2. Token & Position Embedding Layer]
                     ↓
      Vectors of dimension D = 3584
                     ↓
   [3. N Transformer Layers (Self-Attention + FFN)]
      - Multi-Head Self-Attention: Softmax(QK^T / √d_k) * V
      - SwiGLU Feed-Forward Network
                     ↓
      [4. Final Layer Norm & Language Model Head]
                     ↓
      Vocabulary Logits (Vector of size ~152,000)
                     ↓
        [5. Temperature Scaling & Softmax]
                     ↓
            Next Token Probability Distribution
                     ↓
            Sampled Token (e.g. " in")
```

#### Step 1: Tokenization
Computers cannot directly process words. A tokenizer splits text into sub-word units called **tokens** based on frequency statistics (Byte-Pair Encoding). For instance, the word `unverifiable` might be split into three tokens: `["un", "veri", "fiable"]`. Each token corresponds to a unique integer ID in a fixed vocabulary table (Qwen2.5 uses ~152,000 tokens).

#### Step 2: Embedding Layer
Each integer token ID is converted into a continuous, dense numerical vector (embedding) of dimensionality $D = 3584$. Rotary Positional Embeddings (RoPE) are injected to encode the relative position and order of words in the sentence.

#### Step 3: Multi-Head Self-Attention
Self-attention allows the model to compute dynamic relationships between all tokens in the context window.
For each token vector, the model computes three projections:
* **Query ($Q$):** What this token is looking for.
* **Key ($K$):** What information this token holds.
* **Value ($V$):** The actual content representation.

The attention score between token $i$ and token $j$ is calculated via scaled dot-product:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
If the claim is *"Fleming discovered penicillin in 1928"*, the attention mechanism enables the token `"1928"` to attend heavily to `"Fleming"` and `"discovered"`.

#### Step 4: Feed-Forward Network (FFN)
After tokens aggregate context from each other via attention, each token passes independently through non-linear feed-forward layers (SwiGLU) that project the token into a higher-dimensional space (e.g. 18,944 dimensions) and back. This is where factual associations and world knowledge stored in the weights are activated.

#### Step 5: Logits and Softmax
The output vector of the final layer is projected onto the entire vocabulary via the **Language Model (LM) Head**, producing an unnormalized score (logit $z_i$) for every possible token in the vocabulary. The **Softmax** function converts logits into a normalized probability distribution:
$$P(w_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

#### Step 6: Autoregressive Loop
The model selects the next token, appends it to the sequence, and feeds the sequence back into the model to predict the subsequent token. This continues token-by-token until an End-of-Sequence (`<|im_end|>`) token is generated.

---

### 4. What Temperature Does
**Temperature ($T$)** is a hyperparameter that controls the randomness and entropy of token selection by scaling the raw logits before applying Softmax:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

* **Temperature = 0.0 (Greedy / Argmax Decoding):**
  As $T \to 0$, the mathematical limit forces the highest logit to have probability $P = 1.0$ and all others to $0.0$. The model deterministically chooses the single most likely token at every step.
  * **Why we use $T = 0.0$ in Vishwas AI:** Verification and claim extraction require strict reproducibility and factual rigor. A claim should be audited identically every time it is checked.
* **Higher Temperature ($T \ge 0.7$):**
  Divides logits by a number $> 1$, flattening the differences between high and low probability tokens. This increases diversity and "creativity," but introduces randomness and hallucination risk, which is unacceptable for an auditing system.

---

### 5. Why Small Models Make More Mistakes
Comparing a 3B parameter model to a 7B or 70B parameter model illustrates key trade-offs in reasoning capacity:

1. **Parameter Capacity & Memory:** A 70B model has 10x more parameters than a 7B model. Parameters are the weights that store associative facts, world history, and language nuance.
2. **Multi-Step Deductive Reasoning:** Fact-checking often requires indirect inference:
   * *Claim:* "Neil Armstrong walked on the Moon in 1969."
   * *Evidence:* "Apollo 11 commander Neil Armstrong stepped onto lunar soil on July 20, 1969."
   * A larger model easily maps "commander stepped onto lunar soil" $\to$ "walked on the Moon". Smaller models ($\le$ 3B) often fail semantic entailment and misclassify such cases as `unsupported`.
3. **Instruction Following & Schema Adherence:** Small models struggle to output pure JSON without conversational fillers (e.g., *"Sure, here are the claims:"*), necessitating our robust parsing fallbacks and grammar enforcement.

---

### 6. What Context Length Means
**Context Length** (or Context Window) is the maximum number of tokens an LLM can process in a single inference call (combined prompt + generated response).
* For `qwen2.5:7b`, the native architecture supports up to **32,768 tokens**.
* **The Cost of Context (KV Cache):** As context length increases, the model must store the Key and Value vectors for every past token in memory (**KV Cache**). Memory usage for the KV cache scales linearly with sequence length:
  $$\text{KV Cache Memory} = 2 \times (\text{layers}) \times (\text{heads}) \times (\text{dim}) \times (\text{seq\_len}) \times (\text{bytes per value})$$
* **In Vishwas AI:** We bound evidence snippets to 400 characters each. This keeps total input tokens per claim check under ~800 tokens, ensuring negligible KV cache overhead, low latency, and zero risk of Out-Of-Memory (OOM) on a 4 GB GPU.

---

### 7. Disk Caching Architecture (SQLite)
To ensure research reproducibility and minimize API / compute waste, Vishwas AI incorporates an ACID-compliant disk cache in `data/cache.db`:
* **`llm_cache` Table:** Stores deterministic SHA-256 hashes of `(provider, model, system_prompt, user_prompt, temperature, format_type)`. If the exact prompt was audited previously, the cached verdict is returned in **< 0.02 seconds** without re-running model inference.
* **`search_cache` Table:** Stores SHA-256 hashes of `(query, max_results)` and serialized search results. This eliminates duplicate network calls to Tavily or encyclopedic providers across benchmark runs.
* **Concurrency:** The database uses Write-Ahead Logging (`PRAGMA journal_mode=WAL;`) with connection pooling context managers, enabling safe multi-threaded evaluation without table lock contention.
