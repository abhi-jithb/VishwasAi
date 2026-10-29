# Limitations: what they are and how to say them

Each item gives the limitation, the evidence for it, and one sentence to say it confidently.

1. **Small, author-written gold set.** 30 cases and 96 claims, written and labelled by one person, with no second labeller and no agreement score. The 95% confidence interval for end-to-end accuracy is 46–66%.
   *Say:* "This is a controlled 96-claim test set that I built to cover five specific failure types. The confidence intervals are wide, so I report them, and the numbers are evidence that the pipeline works, not a leaderboard score."

2. **One model.** All headline numbers come from qwen2.5:7b (Q4_K_M, local). The older no-retrieval file `gold_qwen_qwen3.8_27b_no_retrieval_*.json` used a different model on Groq, so it is not a fair comparison. That is why the same-model ablation (`RETRIEVAL_ABLATION.md`) exists.
   *Say:* "I fixed the model so the comparison isolates retrieval. The pipeline is model-agnostic, and swapping the model is one environment variable."

3. **The live web changes results.** Tavily, DuckDuckGo and Wikipedia return different snippets on different days. A rerun can flip verdicts even at temperature 0. The cache makes reruns repeatable on one machine, but it does not make them repeatable over time.
   *Say:* "Retrieval is live by design, so I store every query, snippet and URL in the result file (`trace`). Any verdict can be audited against the exact evidence the judge saw."

4. **Heuristic error causes.** The 17/19/6 split comes from string rules in `scripts/generate_error_analysis.py`. Every missed claim counts as "extraction", and every wrong "unsupported" counts as "search". No human opened the evidence.
   *Say:* "The automatic taxonomy is a first-pass triage. `ERROR_AUDIT.csv` has every failure with its evidence, and I have hand-labelled them [or: am hand-labelling them]."

5. **Lenient lexical matching.** Gold-to-extracted matching uses 0.4·Jaccard + 0.6·overlap ≥ 0.35. The overlap coefficient lets a short gold claim match a longer extracted claim that contains extra facts. Paraphrases with different words can be missed.
   *Say:* "Matching is deterministic and transparent: I can show the exact score for any pair. It favours recall, and a semantic matcher is the obvious upgrade."

6. **Overlapping extracted claims.** The extractor sometimes emits claims that overlap or repeat each other (for example "The unconditional surrender of the Axis powers occurred."). The 15 "extra" claims mix real ungolded facts, duplicates and vague fragments.
   *Say:* "Extra claims don't lower recall or accuracy. I report them separately so precision of extraction is visible, not hidden."

7. **Cache effects on timing.** The official 742 s / 24.7 s per case was measured with a warm cache: 20 of 30 cases took < 1 s, and 4 cases took 696 s. The official run was also on a different laptop (Windows, RTX 5050) from the new runs (Fedora, i5-11400H; Ollama ran mostly on CPU here).
   *Say:* "The headline time was a warm-cache run. The cold, uncached timings on this machine are in the ablation file, and I quote those for cost."

8. **Model fallback risk (now closed for benchmarks).** On Groq, a 429 rate limit used to switch silently to another model (`backend/llm.py`). A benchmark could then mix models without saying so. This never affects Ollama runs. Now every switch logs a warning, `NO_MODEL_FALLBACK=1` turns it off, `tests/run_gold.py` always turns it off, and each result file records `models_used`.
   *Say:* "Every benchmark file records the exact model that produced each answer, and the fallback cannot trigger during benchmarks."

9. **"Unsupported" is overused.** Per-class precision for Unsupported is only 34.5%: 13 true claims were judged Unsupported. The strict evidence-only rule plus 3 short snippets often lacks the key sentence.
   *Say:* "The judge is deliberately conservative. It prefers 'I can't confirm' to a wrong 'true', which is why Supported precision is 95%."

10. **Retries.** "5 LLM retries" means 5 attempts in total. Back-off is linear, not exponential. The search retry loop rarely fires, because `web_search` already catches provider errors and falls through to the next provider.
    *Say:* "Resilience comes mainly from the provider fallback chain, Tavily → DuckDuckGo → Wikipedia, which I demonstrated (`search_fallback_proof.txt`)."
