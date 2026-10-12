# Vishwas AI

Vishwas AI is an evaluation layer designed to audit LLM answers: it splits an answer into factual claims, searches for evidence, and verifies each claim.

This repository contains the foundation layer and connectivity checks for the LLM and web search integrations.

---

## Project Structure

```
vishwas-ai/
├── backend/
│   ├── __init__.py
│   ├── config.py       # Environment variable loader & validator
│   ├── llm.py          # Unified LLM provider abstraction (Groq / OpenAI / Anthropic)
│   └── search.py       # Web search abstraction (Tavily)
├── scripts/
│   ├── __init__.py
│   ├── test_llm.py     # Connectivity test for LLM
│   └── test_search.py  # Connectivity test for Tavily web search
├── .env.example        # Environment configuration template
├── .gitignore          # Git ignore rules (protects .env, venv, caches)
├── requirements.txt    # Pinned dependency specifications
└── README.md           # Setup and usage guide
```

---

## Setup Instructions

### 1. Create a Virtual Environment

Ensure you have Python 3.11+ installed:

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

On Linux/macOS:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

Install pinned dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Open `.env` and fill in your actual credentials:
- `LLM_PROVIDER`: `groq`, `openai`, or `anthropic`
- `LLM_MODEL`: model identifier (e.g. `llama-3.3-70b-versatile`, `gpt-4o`, `claude-3-5-sonnet-20241022`)
- `LLM_API_KEY`: API key for the chosen LLM provider
- `SEARCH_API_KEY`: API key for Tavily search

> **Note:** Never commit `.env` or hardcode API keys.

---

## Running Connectivity Tests

Run the connectivity checks from the project root using module syntax:

### Test LLM Connectivity:
```bash
python -m scripts.test_llm
```

### Test Tavily Search Connectivity:
```bash
python -m scripts.test_search
```
