"""Offline unit tests for Vishwas AI.

No network and no LLM: call_llm and web_search are replaced with mocks.
Run: python -m pytest tests/test_offline.py -v
"""

import json
import sys
import types
from pathlib import Path

import pytest

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend import cache, llm, search
from backend.evaluation import claims, evidence, pipeline, scoring, verify
from tests.run_gold import (
    _claim_similarity,
    confusion_matrix,
    match_gold_to_extracted,
    per_class_prf,
)

OFFICIAL_RESULTS = root_dir / "results" / "gold_qwen2.5_7b_20261028_000521.json"


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    """Never read or write data/cache.db during tests."""
    monkeypatch.setenv("DISABLE_CACHE", "1")


class FakeLLM:
    """Returns queued responses and records every (system, user) prompt."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, system, user, temperature=0.0, json_mode=None):
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Claim JSON parsing
# ---------------------------------------------------------------------------


def test_claims_clean_array():
    assert claims._clean_and_parse_claims_json('["A is B.", "C is D."]') == ["A is B.", "C is D."]


def test_claims_fenced_array():
    raw = '```json\n["Paris is the capital of France."]\n```'
    assert claims._clean_and_parse_claims_json(raw) == ["Paris is the capital of France."]


def test_claims_dict_wrapped():
    raw = '{"claims": ["X was born in 1900.", "X died in 1980."]}'
    assert claims._clean_and_parse_claims_json(raw) == ["X was born in 1900.", "X died in 1980."]


def test_claims_dict_of_strings():
    raw = '{"claim_1": "Water boils at 100 C.", "claim_2": "Ice melts at 0 C."}'
    assert claims._clean_and_parse_claims_json(raw) == ["Water boils at 100 C.", "Ice melts at 0 C."]


def test_claims_array_inside_text():
    raw = 'Here are the claims: ["A is B."] hope this helps'
    assert claims._clean_and_parse_claims_json(raw) == ["A is B."]


def test_claims_malformed_raises():
    with pytest.raises(ValueError):
        claims._clean_and_parse_claims_json("I cannot answer that.")


def test_extract_claims_retry_includes_question_answer_and_bad_output(monkeypatch):
    fake = FakeLLM(["not json at all", '["Fleming discovered penicillin."]'])
    monkeypatch.setattr(claims, "call_llm", fake)
    tracker = {}
    out = claims.extract_claims("Who discovered penicillin?", "Fleming did.", cost_tracker=tracker)
    assert out == ["Fleming discovered penicillin."]
    assert tracker["llm_calls"] == 2
    retry_user = fake.calls[1]["user"]
    assert "Who discovered penicillin?" in retry_user
    assert "Fleming did." in retry_user
    assert "not json at all" in retry_user


# ---------------------------------------------------------------------------
# Verdict parsing and URL filtering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("supported", "supported"),
        ("Supported", "supported"),
        ("contradicted", "contradicted"),
        ("unsupported", "unsupported"),
        ("not supported", "unsupported"),
        ("NOT_SUPPORTED", "unsupported"),
        ("not contradicted", "unsupported"),
        ("supported (not contradicted)", "supported"),
        ("insufficient evidence", "unsupported"),
        ("", "unsupported"),
    ],
)
def test_parse_status(raw, expected):
    assert verify._parse_status(raw) == expected


def test_verification_json_not_supported_is_unsupported():
    raw = '{"status": "not supported", "reasoning": "r", "evidence_used": ["https://a.org"]}'
    out = verify._clean_and_parse_verification_json(raw, {"https://a.org"})
    assert out["status"] == "unsupported"
    assert out["evidence_used"] == []


def test_url_filter_drops_urls_not_in_evidence():
    raw = json.dumps(
        {"status": "supported", "reasoning": "r", "evidence_used": ["https://a.org", "https://made-up.com"]}
    )
    out = verify._clean_and_parse_verification_json(raw, {"https://a.org", "https://b.org"})
    assert out["evidence_used"] == ["https://a.org"]


def test_url_filter_no_fallback_to_unfiltered_urls():
    raw = json.dumps({"status": "contradicted", "reasoning": "r", "evidence_used": ["https://made-up.com"]})
    out = verify._clean_and_parse_verification_json(raw, {"https://a.org"})
    assert out["evidence_used"] == []


def test_verify_empty_evidence_skips_llm(monkeypatch):
    fake = FakeLLM([])
    monkeypatch.setattr(verify, "call_llm", fake)
    tracker = {}
    out = verify.verify_claim("X is Y.", [], cost_tracker=tracker, retrieval=True)
    assert out["status"] == "unsupported"
    assert fake.calls == []
    assert tracker == {}


def test_verify_retry_includes_claim_and_evidence(monkeypatch):
    good = '{"status": "supported", "reasoning": "ok", "evidence_used": ["https://a.org"]}'
    fake = FakeLLM(["garbage", good])
    monkeypatch.setattr(verify, "call_llm", fake)
    ev = [{"title": "T", "url": "https://a.org", "snippet": "Paris is the capital of France."}]
    out = verify.verify_claim("Paris is the capital of France.", ev, retrieval=True)
    assert out["status"] == "supported"
    retry_user = fake.calls[1]["user"]
    assert "Paris is the capital of France." in retry_user
    assert "https://a.org" in retry_user
    assert "garbage" in retry_user


# ---------------------------------------------------------------------------
# Scoring formulas
# ---------------------------------------------------------------------------


def test_scoring_empty_list():
    s = scoring.compute_factuality([])
    assert s["total"] == 0
    assert s["factuality_score"] == 0.0
    assert s["verifiable_precision"] is None
    assert s["unverifiable_rate"] == 0.0


def test_scoring_all_unsupported_has_no_verifiable_precision():
    s = scoring.compute_factuality([{"status": "unsupported"}] * 3)
    assert s["verifiable_precision"] is None
    assert s["unverifiable_rate"] == 1.0
    assert s["factuality_score"] == 0.0


def test_scoring_mixed():
    rows = [{"status": "supported"}] * 2 + [{"status": "contradicted"}] + [{"status": "unsupported"}]
    s = scoring.compute_factuality(rows)
    assert (s["supported"], s["contradicted"], s["unsupported"], s["total"]) == (2, 1, 1, 4)
    assert s["factuality_score"] == 0.5  # F = 2/4
    assert s["verifiable_precision"] == round(2 / 3, 4)  # V = 2/(2+1)
    assert s["unverifiable_rate"] == 0.25  # U = 1/4


def test_scoring_unknown_status_counts_as_unsupported():
    s = scoring.compute_factuality([{"status": "weird"}])
    assert s["unsupported"] == 1


# ---------------------------------------------------------------------------
# Claim similarity and greedy matching
# ---------------------------------------------------------------------------


def test_claim_similarity_worked_example():
    # gold:      {alexander, fleming, discovered, penicillin, st, mary, s, hospital, london} (9 words)
    # extracted: {alexander, fleming, discovered, penicillin, 1928} (5 words)
    # intersection 4, union 10 -> Jaccard 0.4, overlap 4/5 = 0.8 -> 0.4*0.4 + 0.6*0.8 = 0.64
    gold = "Alexander Fleming discovered penicillin at St. Mary's Hospital in London."
    ext = "Alexander Fleming discovered penicillin in 1928."
    assert _claim_similarity(gold, ext) == pytest.approx(0.64)


def test_claim_similarity_identical_and_empty():
    assert _claim_similarity("Paris is in France.", "Paris is in France.") == pytest.approx(1.0)
    assert _claim_similarity("the a an", "Paris") == 0.0


def test_greedy_matching_fixture():
    # Case 1 of the gold set: gold #1 takes extracted #0, so gold #2 (St. Mary's) cannot reuse it
    gold = [
        {"text": "Alexander Fleming discovered penicillin in 1928.", "label": "supported"},
        {"text": "Alexander Fleming discovered penicillin at St. Mary's Hospital in London.", "label": "supported"},
        {"text": "Alexander Fleming shared the Nobel Prize in Physiology or Medicine in 1945.", "label": "supported"},
    ]
    extracted = [
        {"text": "Alexander Fleming discovered penicillin in 1928."},
        {"text": "Alexander Fleming was awarded the Nobel Prize in Physiology or Medicine in 1945 for discovering penicillin."},
    ]
    matches = match_gold_to_extracted(gold, extracted)
    assert matches[0] == (0, pytest.approx(1.0))
    assert matches[1][0] == -1  # missed: best unused candidate scores below 0.35
    assert matches[1][1] < 0.35
    assert matches[2][0] == 1


def test_matching_reproduces_official_run():
    data = json.loads(OFFICIAL_RESULTS.read_text(encoding="utf-8"))
    matched = agreed = 0
    pairs = []
    for case in data["cases"]:
        ext = case["extracted_claims"]
        for g, (idx, _) in zip(case["gold_claims"], match_gold_to_extracted(case["gold_claims"], ext)):
            if idx != -1:
                matched += 1
                pred = ext[idx]["status"].lower()
                pairs.append((g["label"].lower(), pred))
                agreed += pred == g["label"].lower()
    assert matched == data["metrics"]["total_matched_claims"] == 79
    assert agreed == data["metrics"]["status_agreed"] == 54
    cm = confusion_matrix(pairs)
    assert [cm["supported"][p] for p in ("supported", "contradicted", "unsupported")] == [39, 3, 13]


def test_per_class_prf_and_zero_division():
    cm = confusion_matrix([("supported", "supported"), ("supported", "unsupported")])
    prf = per_class_prf(cm)
    assert prf["supported"]["precision"] == 100.0
    assert prf["supported"]["recall"] == 50.0
    assert prf["contradicted"]["precision"] is None  # nothing predicted contradicted
    assert prf["contradicted"]["recall"] is None  # no gold contradicted


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------


def test_llm_cache_key_deterministic():
    k1 = cache._compute_llm_key("ollama", "qwen2.5:7b", "sys", "user", 0.0, "json")
    k2 = cache._compute_llm_key(" Ollama ", "QWEN2.5:7B", "sys ", " user", 0.0, "JSON")
    assert k1 == k2
    assert len(k1) == 64
    assert k1 != cache._compute_llm_key("ollama", "qwen2.5:7b", "sys", "user", 0.1, "json")
    assert k1 != cache._compute_llm_key("ollama", "llama3.2:3b", "sys", "user", 0.0, "json")


def test_search_cache_key_deterministic():
    assert cache._compute_search_key("Paris France", 3) == cache._compute_search_key(" paris france ", 3)
    assert cache._compute_search_key("Paris France", 3) != cache._compute_search_key("Paris France", 5)


# ---------------------------------------------------------------------------
# Search fallback chain (mocked network)
# ---------------------------------------------------------------------------


class _FakeDDGS:
    def __init__(self, fail=False):
        self.fail = fail

    def text(self, query, max_results=5):
        if self.fail:
            raise RuntimeError("ddg down")
        return [{"title": "DDG hit", "href": "https://ddg.example", "body": "x" * 500}]


def test_search_duckduckgo_when_tavily_disabled(monkeypatch):
    monkeypatch.setattr(search, "SEARCH_API_KEY", "")
    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=_FakeDDGS))
    out = search.web_search("anything", max_results=3)
    assert out[0]["provider"] == "duckduckgo"
    assert out[0]["url"] == "https://ddg.example"
    assert len(out[0]["snippet"]) == 403  # 400 chars + "..."


def test_search_wikipedia_when_duckduckgo_fails(monkeypatch):
    monkeypatch.setattr(search, "SEARCH_API_KEY", "")
    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=lambda: _FakeDDGS(fail=True)))

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            body = {"query": {"search": [{"title": "Penicillin", "snippet": "<b>Penicillin</b> was found"}]}}
            return json.dumps(body).encode()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10.0: _Resp())
    out = search.web_search("penicillin", max_results=3)
    assert out == [
        {
            "title": "Penicillin",
            "url": "https://en.wikipedia.org/wiki/Penicillin",
            "snippet": "Penicillin was found",
            "provider": "wikipedia",
        }
    ]


def test_search_query_derivation_truncates_to_200_chars():
    q = evidence._derive_search_query('"' + "word " * 100 + '."')
    assert len(q) <= 200
    assert not q.endswith(".")


# ---------------------------------------------------------------------------
# Model fallback switch
# ---------------------------------------------------------------------------


def test_model_fallback_switch(monkeypatch):
    monkeypatch.delenv("BENCHMARK_MODEL", raising=False)
    monkeypatch.delenv("NO_MODEL_FALLBACK", raising=False)
    assert llm.model_fallback_disabled() is False
    monkeypatch.setenv("NO_MODEL_FALLBACK", "1")
    assert llm.model_fallback_disabled() is True


# ---------------------------------------------------------------------------
# End-to-end pipeline with mocks
# ---------------------------------------------------------------------------


def test_pipeline_end_to_end_mocked(monkeypatch):
    monkeypatch.setattr(claims, "call_llm", lambda **kw: '["Paris is the capital of France.", "Zorbia is a country."]')

    def fake_verify_llm(system, user, temperature=0.0, json_mode=None):
        status = "supported" if "Paris" in user else "contradicted"
        return json.dumps({"status": status, "reasoning": "r", "evidence_used": ["https://a.org"]})

    monkeypatch.setattr(verify, "call_llm", fake_verify_llm)

    def fake_search(query, max_results=5):
        if "Zorbia" in query:
            return []
        return [{"title": "T", "url": "https://a.org", "snippet": "Paris is the capital of France."}]

    monkeypatch.setattr(evidence, "web_search", fake_search)
    out = pipeline.evaluate("Q?", "Paris is the capital of France. Zorbia is a country.")
    statuses = [c["status"] for c in out["claims"]]
    assert statuses == ["supported", "unsupported"]  # empty evidence -> unsupported, no judge call
    assert out["cost_log"] == {"llm_calls": 2, "search_calls": 2}
    assert out["summary"]["factuality_score"] == 0.5
