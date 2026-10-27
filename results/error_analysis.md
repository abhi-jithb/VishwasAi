# Vishwas AI: Benchmark Error Analysis Report

- **Model Evaluated:** `qwen2.5:7b` (`ollama`)
- **Timestamp:** `2026-10-28T00:05:21.114368`
- **Total Cases:** 30
- **Gold Claims:** 96
- **Matched Claims:** 79 (82.29% recall)
- **Verification Accuracy:** 68.35%
- **End-to-End Accuracy:** 56.25%

---

## 1. Error Taxonomy & Distribution

Every error in the evaluation pipeline belongs to one of four failure modes:
1. **`[Bad Extraction]`**: The claim extractor failed to isolate the atomic claim, merged multiple checkable assertions, or omitted the fact.
2. **`[Bad Search Results]`**: The web retrieval query was too vague or returned irrelevant / empty snippets, starving the judge of evidence.
3. **`[Judge Error]`**: Sufficient evidence was present, but the model failed semantic entailment or misapplied factual rigor.
4. **`[Ambiguous Gold Label]`**: The claim itself has debatable ground truth or relies on outdated temporal definitions.

### Summary of Failure Modes

| Failure Mode | Count | Share (%) | Primary Mitigation Strategy |
| :--- | :---: | :---: | :--- |
| **Bad Extraction** | 17 | 40.5% | Few-shot structured prompting & grammar-guided JSON schemas |
| **Bad Search Results** | 19 | 45.2% | Multi-query search expansion & Wikipedia fallback API |
| **Judge Error** | 6 | 14.3% | NLI Cross-Encoder verifier (Stage C) & Chain-of-Thought reasoning |
| **Ambiguous Gold Label** | 0 | 0.0% | Dataset curation & precision qualification |

---

## 2. Granular Case-by-Case Breakdown

### #1. Case 1 [correct] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'Alexander Fleming discovered penicillin at St. Mary's Hospital in London.' (Gold: supported)
- **Question:** *"Who discovered penicillin and when?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #2. Case 2 [correct] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'The Apollo 11 mission landed on the Moon in July 1969.' (Gold: supported)
- **Question:** *"What happened during the Apollo 11 mission?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #3. Case 2 [correct] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'Buzz Aldrin walked on the lunar surface.' (Gold: supported)
- **Question:** *"What happened during the Apollo 11 mission?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #4. Case 3 [correct] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'James Watson and Francis Crick's discovery relied on Rosalind Franklin's X-ray diffraction images.' (Gold: supported)
- **Question:** *"How was the structure of DNA discovered?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #5. Case 4 [correct] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'Guido van Rossum created the Python programming language.' (Gold: supported)
- **Question:** *"Who created the Python programming language?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #6. Case 4 [correct] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'Python was named after the comedy group Monty Python.' (Gold: supported)
- **Question:** *"Who created the Python programming language?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #7. Case 4 [correct] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Python was first released in 1991.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'The language was first released in 1991. The language was named after the British comedy group Monty Python.')
- **Question:** *"Who created the Python programming language?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #8. Case 5 [correct] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Albert Einstein published the general theory of relativity in 1915.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'Albert Einstein published his general theory of relativity in 1915.')
- **Question:** *"When was Albert Einstein's general theory of relativity confirmed?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #9. Case 7 [planted_error] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Neil Armstrong first summited Mount Everest in May 1953.' -> Pipeline: [unsupported] vs Gold: [contradicted] (Matched: 'Neil Armstrong first summited Mount Everest in May 1953.')
- **Question:** *"Who first summited Mount Everest and how tall is it?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #10. Case 8 [planted_error] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'World War II began in September 1939.' (Gold: supported)
- **Question:** *"When did World War II begin and end?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #11. Case 8 [planted_error] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Nazi Germany invaded France in September 1939.' -> Pipeline: [unsupported] vs Gold: [contradicted] (Matched: 'Nazi Germany invaded France.')
- **Question:** *"When did World War II begin and end?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #12. Case 9 [planted_error] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'The Sun is classified as a G-type main-sequence star.' (Gold: supported)
- **Question:** *"Can you provide key facts about our Solar System?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #13. Case 10 [planted_error] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'The United States Declaration of Independence was adopted in 1776.' (Gold: supported)
- **Question:** *"Who were the first presidents of the United States?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #14. Case 10 [planted_error] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'Thomas Jefferson served as the second US President.' (Gold: contradicted)
- **Question:** *"Who were the first presidents of the United States?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #15. Case 10 [planted_error] - Judge Error
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'George Washington served as the first US President.' -> Pipeline: [contradicted] vs Gold: [supported] (Matched: 'Thomas Jefferson served as the second US President.')
- **Question:** *"Who were the first presidents of the United States?"*
- **Root Cause Analysis:** Judge misapplied strictness criteria and marked true claim as contradicted.

### #16. Case 11 [planted_error] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'Dmitri Mendeleev published the periodic table in 1869.' (Gold: supported)
- **Question:** *"What are the chemical symbols for gold and silver?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #17. Case 11 [planted_error] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'The chemical symbol for silver is Si.' (Gold: contradicted)
- **Question:** *"What are the chemical symbols for gold and silver?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #18. Case 11 [planted_error] - Judge Error
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'The chemical symbol for gold is Au.' -> Pipeline: [contradicted] vs Gold: [supported] (Matched: 'The chemical symbol for silver is Si.')
- **Question:** *"What are the chemical symbols for gold and silver?"*
- **Root Cause Analysis:** Judge misapplied strictness criteria and marked true claim as contradicted.

### #19. Case 12 [planted_error] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'The RMS Titanic was owned and operated by the Cunard Line.' -> Pipeline: [unsupported] vs Gold: [contradicted] (Matched: 'The RMS Titanic was owned and operated by the Cunard Line.')
- **Question:** *"What happened to the Titanic ocean liner?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #20. Case 13 [fabricated_entity] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'Cafe Zyxvu is located in rural Iceland.' (Gold: unsupported)
- **Question:** *"What is Cafe Zyxvu known for in Iceland?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #21. Case 13 [fabricated_entity] - Judge Error
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Cafe Zyxvu served purple fermented moss stew to tourists in 2024.' -> Pipeline: [contradicted] vs Gold: [unsupported] (Matched: 'Cafe Zyxvu served purple fermented moss stew to tourists in 2024.')
- **Question:** *"What is Cafe Zyxvu known for in Iceland?"*
- **Root Cause Analysis:** LLM reasoning divergence during zero-temperature verification.

### #22. Case 17 [fabricated_entity] - Judge Error
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Over 25,000 attendees gathered for the Oakhaven Cryptid Festival in September 2023.' -> Pipeline: [supported] vs Gold: [unsupported] (Matched: 'Over 25,000 attendees gathered for the festival.')
- **Question:** *"What occurs at the Oakhaven Cryptid Festival?"*
- **Root Cause Analysis:** LLM reasoning divergence during zero-temperature verification.

### #23. Case 19 [outdated_fact] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'Clyde Tombaugh discovered Pluto in 1930.' (Gold: supported)
- **Question:** *"How is Pluto classified in astronomy?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #24. Case 19 [outdated_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Pluto is currently classified as a major planet of the solar system.' -> Pipeline: [unsupported] vs Gold: [contradicted] (Matched: 'Pluto is currently classified as a dwarf planet in the solar system.')
- **Question:** *"How is Pluto classified in astronomy?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #25. Case 21 [outdated_fact] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'The United Kingdom is a member state of the European Union.' (Gold: contradicted)
- **Question:** *"Is the United Kingdom a member state of the European Union?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #26. Case 21 [outdated_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'The European Union has headquarters in Brussels, Belgium.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'The European Union has its headquarters in Brussels, Belgium.')
- **Question:** *"Is the United Kingdom a member state of the European Union?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #27. Case 22 [outdated_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'The Puerto Rico Trench is the deepest oceanic trench on Earth.' -> Pipeline: [unsupported] vs Gold: [contradicted] (Matched: 'The Puerto Rico Trench is the deepest oceanic trench on planet Earth.')
- **Question:** *"What is the deepest known oceanic trench on Earth?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #28. Case 23 [outdated_fact] - Judge Error
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Element 118 on the periodic table is officially called Ununoctium and has no official name.' -> Pipeline: [supported] vs Gold: [contradicted] (Matched: 'Element 118 on the periodic table is officially called Ununoctium.')
- **Question:** *"What is element 118 on the periodic table called?"*
- **Root Cause Analysis:** Model incorrectly evaluated false claim as true due to semantic confusion or noisy snippet matching.

### #29. Case 24 [outdated_fact] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'Queen Elizabeth II is the reigning monarch of the United Kingdom.' (Gold: contradicted)
- **Question:** *"Who is the reigning monarch of the United Kingdom?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #30. Case 25 [long_multi_fact] - Bad Extraction
- **Error Type:** `Missed Claim`
- **Details:** • Missed gold claim: 'Sir Isaac Newton was born in 1643.' (Gold: supported)
- **Question:** *"What were Isaac Newton's major scientific contributions?"*
- **Root Cause Analysis:** Atomic fact was omitted during decomposition or merged into a compound sentence.

### #31. Case 25 [long_multi_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Isaac Newton served as the Master of the Royal Mint in London.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'He served as the Master of the Royal Mint in London.')
- **Question:** *"What were Isaac Newton's major scientific contributions?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #32. Case 26 [long_multi_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Marie Curie discovered the chemical elements polonium and radium.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'She discovered the chemical element polonium.')
- **Question:** *"Can you summarize the life and accomplishments of Marie Curie?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #33. Case 26 [long_multi_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Marie Curie was awarded the Nobel Prize in Literature in 1911.' -> Pipeline: [unsupported] vs Gold: [contradicted] (Matched: 'In 1911, she was awarded a second Nobel Prize.')
- **Question:** *"Can you summarize the life and accomplishments of Marie Curie?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #34. Case 26 [long_multi_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Marie Curie died in 1934 from aplastic anemia caused by radiation exposure.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'Curie died from aplastic anemia caused by radiation exposure.')
- **Question:** *"Can you summarize the life and accomplishments of Marie Curie?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #35. Case 28 [long_multi_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Ada Lovelace was the daughter of poet Lord Byron.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'Ada Lovelace was the daughter of poet Lord Byron.')
- **Question:** *"Who was Ada Lovelace?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #36. Case 28 [long_multi_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'Ada Lovelace died in 1852.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'Ada Lovelace was an English mathematician.')
- **Question:** *"Who was Ada Lovelace?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #37. Case 29 [long_multi_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'The International Space Station has been continuously inhabited since November 2000.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'The station has been continuously inhabited since November 2000.')
- **Question:** *"What are key facts about the International Space Station?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #38. Case 29 [long_multi_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'The ISS orbits Earth approximately once every 90 minutes.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'The ISS orbits Earth approximately once every 90 minutes.')
- **Question:** *"What are key facts about the International Space Station?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #39. Case 30 [long_multi_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'The Great Wall of China was built along northern borders of ancient China.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'The Great Wall of China was constructed along the northern borders of ancient Chinese states.')
- **Question:** *"What is the history of the Great Wall of China?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #40. Case 30 [long_multi_fact] - Judge Error
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'The Ming dynasty constructed well-preserved sections of the Great Wall.' -> Pipeline: [contradicted] vs Gold: [supported] (Matched: 'The Ming dynasty constructed the most famous and well-preserved masonry sections.')
- **Question:** *"What is the history of the Great Wall of China?"*
- **Root Cause Analysis:** Judge misapplied strictness criteria and marked true claim as contradicted.

### #41. Case 30 [long_multi_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'The Great Wall system measures over 20,000 kilometers in total length.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'The wall system spans over 20,000 kilometers in total length.')
- **Question:** *"What is the history of the Great Wall of China?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.

### #42. Case 30 [long_multi_fact] - Bad Search Results
- **Error Type:** `Status Disagreement`
- **Details:** ✗ 'The Great Wall was designated a UNESCO World Heritage Site in 1987.' -> Pipeline: [unsupported] vs Gold: [supported] (Matched: 'The Great Wall was declared a UNESCO World Heritage site in 1987.')
- **Question:** *"What is the history of the Great Wall of China?"*
- **Root Cause Analysis:** Search retrieved insufficient or irrelevant evidence, forcing the judge to default to 'unsupported'.
