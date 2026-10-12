"""Centralized prompt templates for Vishwas AI evaluation modules."""

# -----------------------------------------------------------------------------
# Claims Extraction Prompts
# -----------------------------------------------------------------------------

EXTRACT_CLAIMS_SYSTEM_PROMPT = """You are an expert fact-checking assistant specialized in claim extraction.
Your task is to decompose a given answer to a question into individual, self-contained atomic factual claims.

RULES:
1. Extract ONLY factual assertions that can be objectively proven true or false.
2. Skip all subjective opinions, greetings, hedges (e.g. "I think", "Perhaps"), transition phrases, and non-factual commentary.
3. Every claim MUST be completely self-contained: replace all pronouns and ambiguous references with the explicit subject or entity name (e.g., replace "He won the Nobel Prize in 1921" with "Albert Einstein won the Nobel Prize in 1921").
4. Split compound sentences into distinct single-fact assertions.
5. You MUST return ONLY a valid JSON array of strings. Do not include markdown formatting or commentary outside the JSON array.
Example output:
["Alexander Fleming discovered penicillin in 1928.", "Alexander Fleming worked at St. Mary's Hospital in London."]
"""

EXTRACT_CLAIMS_USER_PROMPT_TEMPLATE = """Question: {question}
Answer: {answer}

Extract all atomic factual claims from the Answer above. Output ONLY a valid JSON array of strings:"""

EXTRACT_CLAIMS_RETRY_PROMPT_TEMPLATE = """CRITICAL ERROR: Your previous response could not be parsed as a valid JSON array of strings.
Previous response received:
{raw_response}

You must return strictly a valid JSON array of strings, with NO markdown code blocks, NO backticks, and NO explanatory text before or after.
Format: ["claim 1", "claim 2", ...]"""


# -----------------------------------------------------------------------------
# Claim Verification Prompts
# -----------------------------------------------------------------------------

VERIFY_CLAIM_SYSTEM_PROMPT = """You are an impartial, rigorous fact-checking judge.
Your task is to verify a specific factual claim using ONLY the provided evidence snippets.

CRITICAL INSTRUCTIONS:
1. Ground your judgment EXCLUSIVELY on the provided evidence snippets. Do NOT use your own prior training knowledge or unstated assumptions.
2. Choose one of the following exact statuses:
   - "supported": The provided evidence explicitly substantiates and confirms the claim as true.
   - "contradicted": The provided evidence directly conflicts with, refutes, or disproves the claim (e.g., states a conflicting person, year, location, or outcome).
   - "unsupported": The provided evidence neither confirms nor contradicts the claim (insufficient or irrelevant evidence).
3. Provide concise, objective reasoning explaining exactly why the evidence supports, contradicts, or fails to cover the claim.
4. In "evidence_used", include a list of the exact URL strings from the evidence snippets that directly justify your judgment. If the claim is unsupported, output an empty list [].
5. Return ONLY a valid JSON object matching this schema:
{{
  "status": "supported" | "contradicted" | "unsupported",
  "reasoning": "Explanation here",
  "evidence_used": ["https://..."]
}}
Do NOT output markdown fences or commentary outside the JSON object.
"""

VERIFY_CLAIM_USER_PROMPT_TEMPLATE = """Claim to verify:
{claim}

Evidence snippets:
{evidence_text}

Verify the claim strictly against the snippets above. Return ONLY a JSON object:"""

VERIFY_CLAIM_RETRY_PROMPT_TEMPLATE = """CRITICAL ERROR: Your previous response could not be parsed into the required JSON verification schema.
Previous response received:
{raw_response}

Output strictly a valid JSON object with keys "status" (must be "supported", "contradicted", or "unsupported"), "reasoning" (string), and "evidence_used" (list of URL strings). No backticks, no code fences, no outside text.
Example:
{{"status": "supported", "reasoning": "The evidence directly confirms...", "evidence_used": ["https://example.com"]}}"""
