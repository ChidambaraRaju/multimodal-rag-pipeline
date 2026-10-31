# Retriever evaluation: baseline (October 31, 2026)

The first evaluation of ReportLens's retriever on its own, with the pipeline as ingested on October 25, 2026.

- **Notebook:** `evals/notebooks/retriever_eval.ipynb`. This run's outputs are in the notebook as committed in `5b13479`; it has been re-run since.
- **Golden set:** `evals/goldens/retriever_goldens.json`, 15 questions
- **Metrics:** deepeval 4.3.2's Contextual Precision and Contextual Recall
- **Judge:** `deepseek-flash` (DeepSeek-V4.1-Flash) on DeepSeek's own API, through deepeval's `DeepSeekModel`

## What was tested

Each golden question goes straight to the agent's `search_document` tool, without the answering model, so it gets exactly the chunks the agent would get for that query:

- The vector store holds 377 text chunks and 12 chart descriptions, embedded with `jina-embeddings-v5-text-small`.
- A search returns the top 3 chunks (`RETRIEVED_CHUNK_COUNT`) and drops any scoring below 0.3 (`MIN_RELEVANCE_SCORE`). No search in this run dropped a chunk.
- Every test case's `actual_output` is `"generator not evaluated"`. Both metrics require one, but neither scores it.

The two metrics, both scored from 0 to 1 by the judge:

| Metric | What it asks |
|---|---|
| Contextual Recall | Can each sentence of the reference answer be found in the retrieved chunks? The score is the share of sentences found. |
| Contextual Precision | Are the chunks that help answer ranked above those that do not? A relevant chunk at rank 1 scores 1.00, at rank 2 alone 0.50, at rank 3 alone 0.33. |

## The golden set

- 15 questions with short reference answers, written from the PDF itself rather than from LlamaParse's output, so a parse error cannot slip into the answers.
- None repeats a `TEST_QUESTIONS` question in `generate.py`.
- Questions are worded the way a reader would ask, not copied from the page.
- The mix: 5 on text, 5 on tables, 3 on charts (pages 7, 24 and 57), and 2 whose answer spans two pages (44 to 45, and Note 7's table on 153 to 154).
- **Checks:** every number in a text or table answer was matched against the PDF's text layer by a script. Chart numbers are not in the text layer, so they were read from the page images. Page 7's chart has no value labels, so its answer is approximate ("more than 400", "about 200").

## Results

| Content | Questions | Mean precision | Mean recall |
|---|---|---|---|
| Chart | 3 | 1.00 | 0.89 |
| Text | 5 | 0.77 | 0.60 |
| Two pages | 2 | 0.79 | 0.25 |
| Table | 5 | 0.30 | 0.40 |
| **All** | **15** | **0.66** | **0.54** |

| # | Content | Question | Expected page | Retrieved pages, in rank order (score) | Precision | Recall |
|---|---|---|---|---|---|---|
| 1 | text | The December 2025 Groq agreement | 149 | **149** (0.62), 24 (0.49), 143 (0.47) | 1.00 | 1.00 |
| 2 | text | Share buyback authorization left | 121 | 150 (0.59), 24 (0.54), 67 (0.53) | 0.50 | 0.00 |
| 3 | text | Why gross margin fell in fiscal 2026 | 125 | 23 (0.71), **125** (0.63), 129 (0.62) | 1.00 | 1.00 |
| 4 | text | Workforce size and research staff | 99 | 124 (0.49), 93 (0.48), 82 (0.46) | 0.33 | 0.00 |
| 5 | text | The 2026 annual meeting | 20 | 25 (0.74), **20** (0.69), 29 (0.66) | 1.00 | 1.00 |
| 6 | table | PwC's fees | 78 | 79 (0.68), **78** (0.67), 137 (0.57) | 0.50 | 1.00 |
| 7 | table | Data Center revenue, compute and networking | 167 | 128 (0.66), 93 (0.65), 23 (0.65) | 0.00 | 0.00 |
| 8 | table | Revenue from customers in China | 166 | **166** (0.49), 129 (0.45), 167 (0.41) | 1.00 | 1.00 |
| 9 | table | Non-employee director pay | 52 | 74 (0.71), 57 (0.71), 61 (0.70) | 0.00 | 0.00 |
| 10 | table | Largest shareholders | 54 | 67 (0.58), 174 (0.56), 142 (0.56) | 0.00 | 0.00 |
| 11 | chart | Open-source AI repositories | 7 | **7** (0.66), **7** (0.63), 17 (0.47) | 1.00 | 1.00 |
| 12 | chart | Cash returned over 1, 3 and 5 years | 24 | **24** (0.73), **24** (0.68), 142 (0.57) | 1.00 | 1.00 |
| 13 | chart | The CEO's target pay mix | 57 | **57** (0.72), 65 (0.70), 62 (0.67) | 1.00 | 0.67 |
| 14 | two pages | Committee chairs and meetings | 44, 45 | 50 (0.68), **45** (0.60), **45** (0.60) | 0.58 | 0.50 |
| 15 | two pages | Cash equivalents and marketable securities, two years | 153, 154 | 141 (0.62), 143 (0.61), 23 (0.57) | 1.00 | 0.00 |

The full questions and reference answers are in the golden set, and the judge's reason for every score below 1 is in the notebook.

## Findings

### 1. The expected page was retrieved for 9 of 15 questions, and every miss is a ranking failure

For the 6 misses (questions 2, 4, 7, 9, 10 and 15), the answer is in a stored chunk, checked by reading the vector store directly: `page121-3`, `page99-2`, `page167-1`, `page52-1`, `page54-1`, and `page153-1` with `page154-1`. Other chunks simply outranked them:

| # | Answer chunk | What ranked above it |
|---|---|---|
| 2 | `page121-3`, prose on repurchases | The employee stock purchase plan (150), business highlights (24), stock ownership guidelines (67) |
| 4 | `page99-2`, human capital | The MD&A overview (124), the business description (93), a stockholder proposal (82) |
| 7 | `page167-1`, revenue by end market table | MD&A pages with growth rates but no dollar split (128, 93, 23) |
| 9 | `page52-1`, director compensation table | Executive pay pages (74, 57, 61) |
| 10 | `page54-1`, security ownership table | Stock ownership guidelines (67), the corporate information page (174), the shareholders' equity statement (142) |
| 15 | `page153-1` and `page154-1`, Note 7's table | The balance sheet (141), the cash flow statement (143), the business overview (23) |

Four of the six answer chunks are tables (`page167-1`, `page52-1`, `page54-1` and Note 7's halves), and in each case prose or a different table on the same topic ranked higher. Question 2 also shows a wording gap: the question says "buy back its own shares", while the page says "share repurchase authorization".

### 2. Tables are the weak spot; charts are the strong one

Tables scored 0.30 precision and 0.40 recall, the lowest of any content type, and the chart questions the highest (1.00 and 0.89). Chart content can be found two ways: through the chart's description, and through the page's text, where LlamaParse also writes a chart's values as a table. For questions 11 and 12 the page's text ranked first and the chart description second; for question 13 only page text was retrieved. Text questions sit between tables and charts.

### 3. The eval found two parse problems on pages that run over a page break

- **Pages 43 to 44 were wrongly merged.** LlamaParse merged page 44's Audit Committee box (its members and responsibilities) into page 43's Lead Director table, so it sits in chunk `page43-1` under page 43. The sentence "In Fiscal 2026, the AC met four times" is in no chunk at all. So question 14 cannot reach full recall, whatever the retriever does.
- **Pages 153 to 154 were not merged.** Each half of Note 7's table is its own chunk. Page 154's begins with only the running header and the date "Jan 26, 2025", with nothing saying what the table is, which may be why neither half was found.

### 4. The judge's scores held up when checked

- **Question 13's recall of 0.67 is a retrieval shortfall.** The retrieved page 57 chunk, `page57-2`, is the page's text, where LlamaParse wrote the CEO's pay mix as a table. The other NEOs' pay mix is in the next chunk, `page57-3`, which ranked 6th, so the judge was right that no retrieved chunk holds it. (Corrected on October 31, 2026: this file first called it a judge mistake, after a check that searched for "44%" when the table holds "44".)
- **Question 15's precision of 1.00 beside a recall of 0.00 is consistent.** Precision counts a chunk as relevant if it helps with any part of the answer, and the top chunk, the balance sheet, holds the marketable securities figures ($51,951 million and $34,621 million). Recall needs each whole sentence of the reference answer supported, and the balance sheet's cash and cash equivalents ($10,605 million and $8,589 million) are not the note's cash equivalents.
- **For question 4,** the judge correctly flagged that the retrieved stockholder proposal (page 82) gives a different headcount ("more than 35,000 people") from the human capital section's "approximately 42,000".

Single scores should still be checked against the retrieved text before acting on them, since the judge's verdicts also vary between runs (see `2026-10-31-retriever-chart-images.md`).

## Cost and time

- **Time:** 12 seconds of scoring, 34 seconds for the whole notebook. deepeval ran the 30 metric evaluations several at a time.
- **Judge tokens:** 82,995 input and 28,463 output.
- **Cost:** deepeval reports $0.059, using DeepSeek's peak list price as given to it ($0.30 a million input tokens and $1.20 a million output tokens, from third-party price lists). That is an upper bound: deepeval charges every input token at the full price, ignoring DeepSeek's cheaper cached input and off-peak hours.
- **Searches:** 15 Jina embeddings of about 15 tokens each.

## Limitations

- **15 questions, one run.** Each content type has only 2 to 5 questions, so a single question moves its mean a lot. The judge's scores were not repeated to see how much they vary.
- **Retrieval scores vary slightly.** Rerunning the searches gave the same pages in the same order, but one score changed in the second decimal (question 14's first chunk, 0.68 then 0.67).
- **The judge saw charts as their descriptions only,** not the images the answering model also gets. The re-run with the images (`2026-10-31-retriever-chart-images.md`) changed no chart score.
- **Precision is coarse with 3 chunks.** With one relevant chunk, it can only be 1.00, 0.50, 0.33 or 0.
- **Only the retriever's own query was tested.** The agent writes its own search queries and can search several times, which this stage leaves out on purpose.
