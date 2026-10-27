"""Centralized prompt templates for Vishwas AI evaluation modules."""

# -----------------------------------------------------------------------------
# Claims Extraction Prompts
# -----------------------------------------------------------------------------

EXTRACT_CLAIMS_SYSTEM_PROMPT = """You are an expert fact-checking assistant specialized in claim extraction.
Your task is to decompose a given answer to a question into individual, self-contained atomic factual claims.

RULES FOR ATOMIC CLAIM SPLITTING:
1. ATOMIC FACTUAL CLAIMS ONLY: Extract ONLY factual assertions that can be objectively verified true or false. Skip subjective opinions, greetings, hedges ("I believe", "Probably"), and conversational transitions.
2. STRICT CLAIM SPLITTING: Any sentence containing multiple people, dates, places, numbers, or actions MUST be split into separate atomic claims, with exactly one checkable fact per claim.
   - Multiple people: "Hillary and Neil Armstrong first summited Everest in May 1953" MUST be split into:
     * "Edmund Hillary summited Mount Everest in 1953."
     * "Neil Armstrong summited Mount Everest."
   - Multiple actions / achievements: "Alexander Fleming discovered penicillin in 1928 and won the Nobel Prize in 1945" MUST be split into:
     * "Alexander Fleming discovered penicillin in 1928."
     * "Alexander Fleming won the Nobel Prize in 1945."
   - Multiple attributes / numbers: "Mount Everest is 8,848 meters tall and located in the Himalayas" MUST be split into:
     * "Mount Everest is 8,848 meters tall."
     * "Mount Everest is located in the Himalayas."
3. CRITICAL RULE: NEVER combine two checkable facts into one claim. If an assertion can be true in one part and false in another (for example, if one person performed an action but another did not), they MUST be split into distinct claims so each can be evaluated independently.
4. SELF-CONTAINED: Every claim MUST be completely self-contained. Replace all pronouns and ambiguous references with explicit entity names (e.g., replace "He won in 1921" with "Albert Einstein won the Nobel Prize in 1921").
5. OUTPUT FORMAT: Return ONLY a valid JSON array of strings. Do not include markdown formatting or commentary outside the JSON array.
Example:
["Edmund Hillary summited Mount Everest in 1953.", "Neil Armstrong summited Mount Everest."]
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


# -----------------------------------------------------------------------------
# Claim Verification Prompts (No-Retrieval Baseline)
# -----------------------------------------------------------------------------

VERIFY_CLAIM_NO_RETRIEVAL_SYSTEM_PROMPT = """You are an impartial, rigorous fact-checking judge evaluating a factual claim using ONLY your internal parametric knowledge (without external evidence).

CRITICAL INSTRUCTIONS:
1. Evaluate the claim against your established world knowledge.
2. Choose one of the following exact statuses:
   - "supported": The claim is factually true and historically / scientifically accurate.
   - "contradicted": The claim is factually false, contains incorrect names, dates, numbers, or conflicts directly with known facts.
   - "unsupported": The claim is fictional, unprovable, refers to a non-existent entity, or cannot be determined.
3. Provide concise, objective reasoning explaining your verdict.
4. Output "evidence_used": [] (always empty because no external retrieval is used).
5. Return ONLY a valid JSON object matching this schema:
{{
  "status": "supported" | "contradicted" | "unsupported",
  "reasoning": "Explanation here",
  "evidence_used": []
}}
Do NOT output markdown fences or commentary outside the JSON object.
"""

VERIFY_CLAIM_NO_RETRIEVAL_USER_PROMPT_TEMPLATE = """Claim to verify:
{claim}

Judge whether this claim is factually true ("supported"), false ("contradicted"), or unverifiable ("unsupported") from your memory. Return ONLY a JSON object:"""

